//! Opt-in hardware gate. The runner requires BOTH CPU and the requested GPU.
//! Missing weights, driver, or compiled backend fail; no availability skips.

use std::path::PathBuf;

use clap::Parser;
use lfm2d::config::Cli;
use lfm2d::engine_real::RealEngine;
use lfm2d::types::EmbedKind;
use lfm2d::worker::InferenceEngine;

mod support;

fn arguments(device: &str) -> Cli {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let models = std::env::var_os("LFM2_MODELS_DIR").map(PathBuf::from).unwrap_or(root.join(".models"));
    Cli::parse_from([
        "lfm2d".to_string(), "--device".into(), device.into(),
        "--bind-addr=127.0.0.1:0".into(),
        format!("--embedder-dir={}", models.join("LFM2.5-Embedding-350M").display()),
        format!("--router-dir={}", models.join("LFM2.5-Encoder-350M-Prompt-Router").display()),
        format!("--token-classifier-dir={}", models.join("LFM2.5-Encoder-350M-PII-Detector").display()),
        "--dtype=f32".into(),
    ])
}

fn close(cpu: f32, gpu: f32) {
    assert!(cpu.is_finite() && gpu.is_finite(), "nonfinite CPU/GPU scores: {cpu}/{gpu}");
    assert!((cpu - gpu).abs() < 0.005, "CPU/GPU score drift: {cpu}/{gpu}");
}

#[test]
#[ignore = "run demo/test_devices.sh <rocm|cuda|metal> on a GPU host"]
fn all_heads_agree_between_cpu_and_explicit_gpu() {
    let backend = std::env::var("LFM2D_TEST_GPU").expect("set LFM2D_TEST_GPU to rocm/cuda/metal");
    assert!(["rocm", "cuda", "metal"].contains(&backend.as_str()), "GPU must be explicit, not auto/CPU");
    support::memory_guard::arm();
    let cpu = RealEngine::load(&arguments("cpu")).expect("load every head on CPU");
    let gpu = RealEngine::load(&arguments(&backend)).expect("load every head on GPU; no skip or fallback");
    assert_eq!(cpu.execution_metadata().device_type, "cpu");
    assert_eq!(gpu.execution_metadata().device_type, "gpu");
    assert_eq!(gpu.execution_metadata().backend, backend);
    let texts = vec!["A mutex protects shared memory from concurrent writes.".into(),
                     "日本語でも意味で検索する。".into()];
    for kind in [EmbedKind::Document, EmbedKind::Query] {
        let a = cpu.embed(&texts, kind).unwrap();
        let b = gpu.embed(&texts, kind).unwrap();
        assert_eq!(a.weight_hash, b.weight_hash);
        assert_eq!(a.vectors.len(), texts.len());
        assert_eq!(b.vectors.len(), texts.len());
        for (a, b) in a.vectors.iter().zip(&b.vectors) {
            assert_eq!(a.len(), b.len());
            assert!(a.iter().chain(b).all(|x| x.is_finite()));
            let similarity = lfm2_encoder::cosine_similarity(a, b);
            assert!(similarity > 0.999, "embedding CPU/GPU cosine {similarity}");
        }
    }
    let routes = vec!["shell".into(), "k8s".into(), "general conversation".into()];
    let a = cpu.route("kubectl get pods", &routes).unwrap();
    let b = gpu.route("kubectl get pods", &routes).unwrap();
    assert_eq!(a.routes.len(), routes.len());
    assert_eq!(b.routes.len(), routes.len());
    for (a, b) in a.routes.iter().zip(&b.routes) {
        assert_eq!(a.route, b.route);
        close(a.cosine, b.cosine);
    }
    let pii = vec!["Contact alice@example.com for the deployment review.".into()];
    let a = cpu.spans(&pii, None).unwrap();
    let b = gpu.spans(&pii, None).unwrap();
    assert!(!a.per_input[0].is_empty(), "PII fixture must exercise detection");
    assert_eq!(a.per_input.len(), b.per_input.len());
    for (a, b) in a.per_input.iter().zip(&b.per_input) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert_eq!((a.start, a.end, &a.entity), (b.start, b.end, &b.entity));
            close(a.score, b.score);
        }
    }
}
