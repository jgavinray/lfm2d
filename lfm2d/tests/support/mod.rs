//! Shared helpers for tests that load the real LFM2.5-8B-A1B GGUF. Every
//! such test builds its `Cli` with [`adjudicator_cli`] and loads through
//! [`load_adjudicator`], which:
//!
//! - names a GPU device explicitly (`LFM2D_TEST_GPU`, default `rocm`) and
//!   refuses `cpu` and `auto`. The CPU MoE dequantizes every expert to f32
//!   (~29 GB) and grows linearly with tokens; it is a reference, never a
//!   fallback, and gets tiny-fixture tests only;
//! - arms the host-memory watchdog (`memory_guard`, shared with the encoder
//!   crate's tests) before loading anything;
//! - bounds context to [`TEST_CONTEXT`].
//!
//! Encoder-only real-model tests call `memory_guard::arm()` themselves.
#![allow(dead_code)]

#[path = "../../../tests/support/memory_guard.rs"]
pub mod memory_guard;

use clap::Parser as _;
use lfm2d::adjudicator::Adjudicator;
use lfm2d::config::Cli;
use lfm2d::device::DeviceArg;

/// Which GPU backend real-model tests run on.
pub const GPU_ENV: &str = "LFM2D_TEST_GPU";
/// Context budget for real 8B tests: prompt plus output. The fixtures'
/// longest prompt (interleave's long email) is under 1k tokens; keep this
/// at what the tests need, since the ROCm caching allocator parks what a
/// prefill frees and never returns it (linear in context since the candle
/// pin's allocator size classes; `long_context_real.rs` is the one binary
/// that runs at 32768). A generate request's
/// `max_tokens` defaults to 2048 and counts against this, so real tests
/// name theirs (512 is ample for the fixture reports).
pub const TEST_CONTEXT: usize = 2048;

/// The GPU backend named by `LFM2D_TEST_GPU` (default `rocm`, the only
/// backend we run today). `cpu`, `auto` and anything else are refused.
pub fn gpu_backend_from(value: Option<&str>) -> Result<&str, String> {
    match value.unwrap_or("rocm") {
        backend @ ("rocm" | "cuda" | "metal") => Ok(backend),
        other => Err(format!(
            "{GPU_ENV}={other:?}: real 8B tests run on an explicit GPU backend (rocm, cuda or metal), \
             never cpu or auto: the CPU MoE dequantizes every expert to f32 (~29 GB) and grows \
             linearly with tokens"
        )),
    }
}

pub fn gpu_backend() -> String {
    let value = std::env::var(GPU_ENV).ok();
    gpu_backend_from(value.as_deref()).unwrap_or_else(|e| panic!("{e}")).to_string()
}

fn models_dir() -> String {
    // `strip_suffix`, not `trim_end_matches`: the latter strips BOTH
    // `/lfm2d`s off a checkout at `.../lfm2d/lfm2d`. A worktree has no
    // `.models`; `LFM2_MODELS_DIR` points at main's.
    std::env::var("LFM2_MODELS_DIR").unwrap_or_else(|_| {
        let crate_dir = env!("CARGO_MANIFEST_DIR");
        format!("{}/.models", crate_dir.strip_suffix("/lfm2d").expect("the crate is <repo>/lfm2d"))
    })
}

pub fn spec_path(name: &str) -> String {
    format!("{}/tests/fixtures/specs/{name}.json", env!("CARGO_MANIFEST_DIR"))
}

/// A `Cli` for the real adjudicator on the test GPU, with each named
/// fixture spec (`tests/fixtures/specs/<name>.json`) loaded at boot.
/// `LFM2D_ADJUDICATOR_MODEL`/`_TOKENIZER` override the checkpoint.
pub fn adjudicator_cli(specs: &[&str]) -> Cli {
    let model = std::env::var("LFM2D_ADJUDICATOR_MODEL")
        .unwrap_or_else(|_| format!("{}/LFM2.5-8B-A1B/LFM2.5-8B-A1B-Q5_K_M.gguf", models_dir()));
    let tokenizer = std::env::var("LFM2D_ADJUDICATOR_TOKENIZER")
        .unwrap_or_else(|_| format!("{}/LFM2.5-8B-A1B/tokenizer.json", models_dir()));
    let mut args = vec![
        "lfm2d".to_string(),
        "--bind-addr=127.0.0.1:0".into(),
        "--dtype=f32".into(),
        format!("--device={}", gpu_backend()),
        format!("--adjudicator-model={model}"),
        format!("--adjudicator-tokenizer={tokenizer}"),
        format!("--adjudicator-context={TEST_CONTEXT}"),
    ];
    args.extend(specs.iter().map(|s| format!("--opinion-spec={}", spec_path(s))));
    Cli::parse_from(args)
}

/// Load the adjudicator for a real-model test, or say why not. Refuses a
/// CPU or auto device before touching the checkpoint, arms the memory
/// guard, then checks the loaded backend.
pub fn try_load_adjudicator(cli: &Cli) -> Result<Adjudicator, String> {
    let named = cli.device.as_str();
    if matches!(cli.device, DeviceArg::Cpu | DeviceArg::Auto) {
        return Err(format!(
            "--device {named}: real 8B tests need an explicit GPU device; build the Cli with \
             support::adjudicator_cli"
        ));
    }
    memory_guard::arm();
    let adjudicator = Adjudicator::load(cli)?;
    let backend = adjudicator.info().backend;
    if backend != named {
        return Err(format!("asked for {named}, loaded on {backend}"));
    }
    Ok(adjudicator)
}

pub fn load_adjudicator(cli: &Cli) -> Adjudicator {
    try_load_adjudicator(cli).unwrap_or_else(|e| panic!("load the adjudicator: {e}"))
}

/// This process's device memory by the CUDA driver's own account: the
/// device's current allocation pool, from which cudarc allocates every
/// candle tensor (`cuMemAllocAsync`). What the libraries allocate for
/// themselves (cuBLAS workspaces, loaded modules) is not in it.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub struct CudaPoolBytes {
    /// Live allocations, byte-exact (`CU_MEMPOOL_ATTR_USED_MEM_CURRENT`):
    /// what a held tensor costs.
    pub used: u64,
    /// What the pool holds from the device (`..._RESERVED_MEM_CURRENT`):
    /// live allocations plus freed blocks kept for reuse, the counterpart
    /// of ROCm's KFD and DRM fdinfo accounts. It grows in chunks of
    /// megabytes, so it cannot size anything smaller.
    pub reserved: u64,
}

#[cfg(feature = "cuda")]
pub fn cuda_pool_bytes(device: &candle_core::Device) -> CudaPoolBytes {
    use candle_core::cuda_backend::cudarc::driver::{result, sys};
    let candle_core::Device::Cuda(cuda) = device else { panic!("not a CUDA device: {device:?}") };
    let context = cuda.cuda_stream().context().clone();
    context.bind_to_thread().expect("bind the CUDA context to this thread");
    // Without pools cudarc falls back to plain cuMemAlloc and this account
    // would read 0: refuse rather than report nothing as a measurement.
    let pools = context
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED)
        .expect("query memory pool support");
    assert_eq!(pools, 1, "this CUDA device has no memory pools; the pool account cannot see its allocations");
    let pool = unsafe { result::device::get_mem_pool(context.cu_device()) }.expect("the device's current memory pool");
    let read = |attribute| {
        let mut bytes = 0u64;
        unsafe { result::mem_pool::get_attribute(pool, attribute, (&mut bytes as *mut u64).cast()) }
            .expect("read a memory pool attribute");
        bytes
    };
    CudaPoolBytes {
        used: read(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT),
        reserved: read(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT),
    }
}
