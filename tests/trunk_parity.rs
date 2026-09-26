//! Numerical parity: our trunk vs the checkpoint's own Python modeling code.
//!
//! The reference activations in `fixtures/trunk_reference.safetensors` were
//! produced by `tests/reference/dump_trunk_reference.py` running LiquidAI's
//! `modeling_lfm2_bidirectional.py` under transformers 4.56.2 in
//! float32/eager. Regenerate them with that script — never by hand, and
//! never by copying what our own implementation produced.
//!
//! These tests need the real weights, which are far too large to commit.
//! They fail loudly with a fetch command rather than skipping: a parity
//! test that quietly passes when it had nothing to compare is worse than no
//! test at all.

#[path = "support/memory_guard.rs"]
mod memory_guard;

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use lfm2_encoder::{Lfm2EncoderConfig, Lfm2Trunk};

const MODEL: &str = "LFM2.5-Embedding-350M";

/// Where the checkpoints live; override with `LFM2_MODELS_DIR`.
fn models_dir() -> PathBuf {
    match std::env::var("LFM2_MODELS_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".models"),
    }
}

fn checkpoint() -> PathBuf {
    memory_guard::arm();
    let dir = models_dir().join(MODEL);
    assert!(
        dir.join("model.safetensors").is_file(),
        "missing weights at {}\n\n  hf download LiquidAI/{MODEL} --local-dir {}\n\n\
         (or point LFM2_MODELS_DIR at a directory that has it)",
        dir.display(),
        dir.display(),
    );
    dir
}

fn reference() -> HashMap<String, Tensor> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/trunk_reference.safetensors");
    candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("load reference {}: {e}", path.display()))
}

fn load_trunk() -> Lfm2Trunk {
    let dir = checkpoint();
    let cfg = Lfm2EncoderConfig::from_json(&std::fs::read(dir.join("config.json")).unwrap())
        .expect("parse checkpoint config.json");
    // f32 even though the checkpoint ships bf16: the reference was dumped
    // in f32, and this is a CPU-first crate where f32 is the fast path.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[dir.join("model.safetensors")],
            DType::F32,
            &Device::Cpu,
        )
        .expect("mmap weights")
    };
    Lfm2Trunk::load(&cfg, vb).expect("load trunk")
}

/// Max absolute elementwise difference.
fn max_abs_diff(a: &Tensor, b: &Tensor) -> f64 {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64
}

/// Tolerance for a 16-block f32 forward pass. Reordered matmuls and a
/// different softmax accumulation make bit-equality impossible; anything
/// this small cannot be a wrong mask, a mis-shaped weight, or a causal
/// conv — those errors land in the 0.1–10 range on activations whose std
/// is ~2.2.
const TOL: f64 = 5e-4;

#[test]
fn trunk_matches_python_reference_unpadded() {
    let refs = reference();
    let trunk = load_trunk();

    let ids = refs["single.input_ids"].to_dtype(DType::U32).unwrap();
    let got = trunk.forward(&ids, None).expect("forward");
    let want = &refs["single.hidden_states"];

    let diff = max_abs_diff(&got, want);
    assert!(
        diff < TOL,
        "hidden states diverge from the Python reference: max|Δ| = {diff:e} (tol {TOL:e})"
    );
}

#[test]
fn cls_pooling_matches_python_reference() {
    let refs = reference();
    let trunk = load_trunk();

    let ids = refs["single.input_ids"].to_dtype(DType::U32).unwrap();
    let hidden = trunk.forward(&ids, None).expect("forward");
    let pooled = Lfm2Trunk::pool_cls(&hidden).expect("pool");

    let diff = max_abs_diff(&pooled, &refs["single.pooled_cls"]);
    assert!(diff < TOL, "CLS pooled vector diverges: max|Δ| = {diff:e}");
}

#[test]
fn padded_batch_matches_python_reference() {
    let refs = reference();
    let trunk = load_trunk();

    let ids = refs["batch.input_ids"].to_dtype(DType::U32).unwrap();
    let mask = refs["batch.attention_mask"].to_dtype(DType::U32).unwrap();
    let got = trunk.forward(&ids, Some(&mask)).expect("forward");
    let want = &refs["batch.hidden_states"];

    // Compare only real-token positions. Padded positions are genuinely
    // unconstrained — the reference leaves whatever the unmasked conv and
    // the softmax-over-all-suppressed-keys produced there — so asserting on
    // them would be pinning noise.
    let (_, seq, _) = got.dims3().unwrap();
    let real_lens = [seq, 4];
    for (row, len) in real_lens.iter().enumerate() {
        let g = got.narrow(0, row, 1).unwrap().narrow(1, 0, *len).unwrap();
        let w = want.narrow(0, row, 1).unwrap().narrow(1, 0, *len).unwrap();
        let diff = max_abs_diff(&g, &w);
        assert!(
            diff < TOL,
            "batch row {row} (first {len} tokens) diverges: max|Δ| = {diff:e}"
        );
    }
}

/// The trained behaviour we deliberately reproduce: because the short conv
/// is NOT masked, a right-padded sequence does not give bit-identical
/// hidden states to the same sequence run alone. This test exists so that
/// if someone "fixes" the conv by zeroing pad states, they find out here
/// rather than in a silent retrieval-quality regression.
#[test]
fn padding_perturbs_results_because_the_conv_is_unmasked() {
    let trunk = load_trunk();
    let dev = Device::Cpu;

    let bare = Tensor::new(&[[1u32, 777, 31, 4]], &dev).unwrap();
    let padded = Tensor::new(&[[1u32, 777, 31, 4, 0, 0, 0, 0]], &dev).unwrap();
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 0, 0, 0, 0]], &dev).unwrap();

    let a = trunk.forward(&bare, None).unwrap();
    let b = trunk
        .forward(&padded, Some(&mask))
        .unwrap()
        .narrow(1, 0, 4)
        .unwrap();

    let diff = max_abs_diff(&a, &b);
    assert!(
        diff > TOL,
        "padding made no difference (max|Δ| = {diff:e}) — did someone mask the \
         short conv? The checkpoints train with pad states flowing through it; \
         masking changes the numbers away from the reference."
    );
}
