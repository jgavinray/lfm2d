//! The adjudicator at a 32k-token context, on the real LFM2.5-8B-A1B.
//!
//! Chat keeps preserved reasoning in its history, which fills 4096 tokens
//! within a few turns (Amy, 2026-09-26: "raise context to 32k"). These tests
//! prove the engine at that depth, not just that the flag parses:
//!
//! - a ~30k-token prompt prefills and generates, and what it generates
//!   depends on the FIRST lines of that prompt (a needle retrieved across
//!   ~30k tokens: rope tables, the per-chunk causal mask and KV growth all
//!   have to be right for that);
//! - an opinion read takes a `facts` block past the old 8192 cap;
//! - `long_context_costs` measures what a deep context costs (prefill,
//!   decode rate, process memory, one state's size) and prints it. It
//!   asserts nothing about speed; the numbers are for reading.
//!
//! The filler text is this repository's own docs, with every line that
//! carries a chat control token dropped, so the rendered turn structure is
//! the one the test builds and nothing else.
//!
//! Ignored by default: each test loads and hashes a 6 GB GGUF. They load
//! through `support` (an explicit GPU from `LFM2D_TEST_GPU`, never cpu or
//! auto, and the host-memory guard armed first) at a 32768-token budget
//! instead of `support::TEST_CONTEXT`. On ROCm they need the candle
//! allocator's size classes: without them a cold 30k prefill parks memory
//! quadratic in depth and exhausted a 128 GB host.
//!
//!   LFM2D_TEST_GPU=rocm LFM2_MODELS_DIR=... cargo test -p lfm2d --release --features rocm \
//!     --test long_context_real -- --ignored --nocapture --test-threads 1
mod support;

use lfm2d::adjudicator::Generator;
use lfm2d::config::Cli;
use lfm2d::opinion_api::{OpinionRequest, SpecMenuEntry};
use lfm2d::probe_api::{ProbeMessage, ProbeRequest, render_messages};
use std::time::Instant;

const CONTEXT: usize = 32768;
const SPEC: &str = "email-triage-v1";
/// Where the needle's answer has to come from: the first line of the user
/// turn, ~30k tokens before the question.
const NEEDLE: &str = "Remember this for later: the courier's password is PERSIMMON-42.";
const QUESTION: &str = "What is the courier's password? Answer with the password only.";
/// The closed reasoning region the opinion prefill uses (measured best,
/// docs/lfm25-adjudicator.md), so the answer starts at once.
const CLOSED_THINK: &str = "<think>\n\n</think>\n";

fn repo_root() -> &'static str {
    env!("CARGO_MANIFEST_DIR").strip_suffix("/lfm2d").expect("the crate is <repo>/lfm2d")
}

/// Through `validate`, as `main` does: a context the config refuses must
/// fail this test, not slip past it because `Adjudicator::load` never
/// looked at the range.
fn cli() -> Cli {
    let mut cli = support::adjudicator_cli(&[SPEC]);
    cli.adjudicator_context = CONTEXT;
    cli.validate().expect("a 32768-token adjudicator context is a valid configuration");
    cli
}

/// Prose with no chat control tokens: the repo's docs, sorted by path so the
/// order is stable, lines carrying `<|`, `<think>` or `</think>` dropped.
fn corpus() -> String {
    let root = repo_root();
    let mut paths: Vec<_> = std::fs::read_dir(format!("{root}/docs"))
        .expect("docs directory")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    paths.sort();
    paths.push(format!("{root}/README.md").into());
    paths.push(format!("{root}/lfm2d/README.md").into());
    let mut out = String::new();
    for p in paths {
        for line in std::fs::read_to_string(&p).unwrap().lines() {
            if line.contains("<|") || line.contains("<think>") || line.contains("</think>") {
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// The corpus cut at a line boundary so it tokenizes to at most `tokens`
/// tokens on its own, repeated first if it is too short.
fn filler(tokenizer: &tokenizers::Tokenizer, tokens: usize) -> String {
    let mut text = corpus();
    let once = text.clone();
    loop {
        let enc = tokenizer.encode(text.as_str(), false).unwrap();
        if enc.get_ids().len() > tokens {
            let cut = enc.get_offsets()[tokens].0;
            let cut = text[..cut].rfind('\n').map_or(0, |at| at + 1);
            text.truncate(cut);
            return text;
        }
        text.push_str(&once);
    }
}

/// A single user turn, needle first, question last, then an assistant turn
/// opened past a closed reasoning region.
fn needle_prompt(tokenizer: &tokenizers::Tokenizer, filler_tokens: usize) -> Vec<u32> {
    let content = format!("{NEEDLE}\n\n{}\n{QUESTION}", filler(tokenizer, filler_tokens));
    let rendered = render_messages(
        &[ProbeMessage { role: "user".into(), content }],
        Some(CLOSED_THINK),
    );
    tokenizer.encode(rendered.as_str(), false).unwrap().get_ids().to_vec()
}

#[test]
#[ignore = "loads and hashes the 6 GB LFM2.5-8B-A1B GGUF; minutes on a GPU host"]
fn a_30k_token_prompt_prefills_and_retrieves_its_first_line() {
    let cli = cli();
    let mut adjudicator = support::load_adjudicator(&cli);
    let tokenizer = adjudicator.tokenizer_clone();
    let ok = || Ok(());

    // One past the budget is refused before any work.
    let over: ProbeRequest =
        serde_json::from_value(serde_json::json!({"ids": vec![1u32; 32769], "top_k": 0})).unwrap();
    over.validate().expect("shape is fine; only the context refuses it");
    let err = adjudicator.probe(&over, &ok).expect_err("32769 ids exceed a 32768 context");
    assert!(format!("{err:?}").contains("context"), "{err:?}");

    let ids = needle_prompt(&tokenizer, 29_900);
    assert!((29_000..=30_500).contains(&ids.len()), "prompt is {} tokens", ids.len());
    let request: ProbeRequest = serde_json::from_value(serde_json::json!({
        "ids": ids,
        "generate": 16,
        "top_k": 0,
        "timeout_ms": 120000
    }))
    .unwrap();
    request.validate().unwrap();
    let begin = Instant::now();
    let response = adjudicator.probe(&request, &ok).expect("probe a ~30k-token prompt");
    let wall = begin.elapsed().as_secs_f64();
    let answer: String = tokenizer
        .decode(&response.generated.iter().map(|s| s.token).collect::<Vec<_>>(), false)
        .unwrap();
    eprintln!(
        "needle: input_tokens={} prefill_ms={:.0} score_ms={:.0} wall_s={wall:.1} generated={} {answer:?}",
        response.input_tokens,
        response.prefill_ms,
        response.score_ms,
        response.generated.len()
    );
    assert_eq!(response.input_tokens, ids.len());
    assert!(!response.generated.is_empty(), "no token generated at depth {}", ids.len());
    assert!(
        response.generated.iter().all(|s| s.logprob.is_finite() && s.logprob <= 0.),
        "logprobs must be finite: {:?}",
        response.generated.iter().map(|s| s.logprob).collect::<Vec<_>>()
    );
    assert!(
        answer.contains("PERSIMMON-42"),
        "the answer has to come from the prompt's first line, {} tokens back: {answer:?}",
        ids.len()
    );
}

#[test]
#[ignore = "loads and hashes the 6 GB LFM2.5-8B-A1B GGUF; minutes on a GPU host"]
fn an_opinion_read_takes_a_facts_block_past_the_old_8192_cap() {
    let cli = cli();
    let mut adjudicator = support::load_adjudicator(&cli);
    let tokenizer = adjudicator.tokenizer_clone();
    let menu = adjudicator.menu();
    let entry: &SpecMenuEntry = menu.iter().find(|e| e.spec == SPEC).expect("spec on the menu");
    // Vocabulary from the menu, never from this file.
    let field = entry
        .fields
        .iter()
        .find(|f| !f.options.is_empty())
        .expect("the spec has a choice field")
        .field
        .clone();
    let ok = || Ok(());
    // As much of the state's byte budget as whole lines allow.
    let input = "I was charged twice for order #4471 and I want a refund today.";
    let mut facts = String::from("Background notes for the reader:\n");
    for line in filler(&tokenizer, 40_000).lines() {
        if facts.len() + line.len() + 1 + input.len() > 65_000 {
            break;
        }
        facts.push_str(line);
        facts.push('\n');
    }
    let request: OpinionRequest = serde_json::from_value(serde_json::json!({
        "spec": SPEC,
        "state": {"input": input, "facts": facts},
        "questions": [{"field": field}],
        "timeout_ms": 120000
    }))
    .unwrap();
    request.validate().expect("a 64 KiB state is within the byte cap");
    let question = entry.resolve(&request.questions[0]).unwrap();
    for (round, expect_described) in [(1, "miss"), (2, "hit")] {
        let begin = Instant::now();
        let response = adjudicator
            .opine(&request, std::slice::from_ref(&question), &ok)
            .expect("opinion read with a long facts block");
        let read = &response.answers[0].read;
        eprintln!(
            "facts round {round}: facts_bytes={} prompt_tokens={} cached={} described_tokens={} \
             prefill_ms={:.0} describe_ms={:.0} read_ms={:.0} wall_ms={:.0} cache={:?} \
             sequence_mass={:.4} first_token_mass={:.4} options={:?}",
            facts.len(),
            response.prompt_tokens,
            response.cached_tokens,
            response.described_tokens,
            response.prefill_ms,
            response.describe_ms,
            response.read_ms,
            begin.elapsed().as_secs_f64() * 1000.,
            response.cache,
            read.sequence_mass,
            read.first_token_mass,
            read.options.iter().map(|o| (o.option.as_str(), o.prob)).collect::<Vec<_>>()
        );
        assert!(
            response.prompt_tokens > 8192,
            "the facts block must carry the prompt past the old cap: {} tokens",
            response.prompt_tokens
        );
        assert_eq!(response.cache.described, expect_described, "round {round}");
        assert!(read.sequence_mass.is_finite() && read.sequence_mass <= 0.);
        assert!(read.options.iter().all(|o| o.prob.is_finite()));
    }
}

// ---- measurement ----------------------------------------------------------

/// This process's GPU memory in bytes, by the backend's own account, part by
/// part: DRM fdinfo's gtt and vram on ROCm, what the allocation pool holds
/// on CUDA (`support::cuda_pool_bytes`, reserved: parked blocks included,
/// as on ROCm).
fn gpu_parts(device: &candle_core::Device) -> Vec<(&'static str, u64)> {
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        return vec![("pool", support::cuda_pool_bytes(device).reserved)];
    }
    let _ = device;
    let (gtt, vram) = drm_fdinfo_bytes();
    vec![("gtt", gtt), ("vram", vram)]
}

fn gpu_bytes(device: &candle_core::Device) -> u64 {
    gpu_parts(device).iter().map(|(_, b)| b).sum()
}

/// This process's GPU memory from DRM fdinfo, in bytes: (gtt, vram). One
/// client can sit behind several fds, so clients are counted once.
fn drm_fdinfo_bytes() -> (u64, u64) {
    let mut seen = std::collections::BTreeSet::new();
    let (mut gtt, mut vram) = (0, 0);
    let Ok(dir) = std::fs::read_dir("/proc/self/fdinfo") else { return (0, 0) };
    for e in dir.flatten() {
        let Ok(text) = std::fs::read_to_string(e.path()) else { continue };
        let Some(client) = text.lines().find_map(|l| l.strip_prefix("drm-client-id:")) else { continue };
        if !seen.insert(client.trim().to_string()) {
            continue;
        }
        let kib = |key: &str| {
            text.lines()
                .find_map(|l| l.strip_prefix(key))
                .and_then(|v| v.trim().trim_end_matches(" KiB").trim().parse::<u64>().ok())
                .unwrap_or(0)
                * 1024
        };
        gtt += kib("drm-memory-gtt:");
        vram += kib("drm-memory-vram:");
    }
    (gtt, vram)
}

fn status_kib(file: &str, key: &str) -> u64 {
    std::fs::read_to_string(file)
        .unwrap()
        .lines()
        .find_map(|l| l.strip_prefix(key))
        .and_then(|v| v.trim().trim_end_matches(" kB").trim().parse().ok())
        .unwrap_or(0)
}

/// Samples RSS and GPU memory every 5 ms, keeping the peaks. The host
/// guard is `support::memory_guard`, armed before the checkpoint loads.
struct Sampler {
    peak: std::sync::Arc<std::sync::Mutex<(u64, u64)>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Sampler {
    fn start(device: &candle_core::Device) -> Self {
        let peak = std::sync::Arc::new(std::sync::Mutex::new((0u64, 0u64)));
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (p, s, device) = (peak.clone(), stop.clone(), device.clone());
        let thread = std::thread::spawn(move || {
            while !s.load(std::sync::atomic::Ordering::Relaxed) {
                let rss = status_kib("/proc/self/status", "VmRSS:") * 1024;
                let gpu = gpu_bytes(&device);
                {
                    let mut g = p.lock().unwrap();
                    g.0 = g.0.max(rss);
                    g.1 = g.1.max(gpu);
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });
        Self { peak, stop, thread: Some(thread) }
    }
    /// (peak RSS, peak GPU) since the last reset, then reset to now.
    fn take(&self) -> (u64, u64) {
        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut g = self.peak.lock().unwrap();
        let out = *g;
        *g = (0, 0);
        out
    }
}
impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            t.join().unwrap();
        }
    }
}

fn mib(b: u64) -> f64 {
    b as f64 / (1024. * 1024.)
}

/// Prefill time, decode rate, memory and one state's size at 8k, 16k and
/// ~30k tokens, straight on the model with the daemon's 128-token prefill
/// chunks and one-token decode steps. Cold every time (a fresh state), one
/// run per depth; prints one JSON line per depth. Greedy argmax on the host,
/// no repetition penalty: this measures the model, not the sampler.
#[test]
#[ignore = "loads the 6 GB GGUF and runs ~2 minutes of GPU work; prints numbers, asserts none"]
fn long_context_costs() {
    // Straight on the model, not through the adjudicator, so the device
    // and the guard are this test's to get right: the same explicit GPU
    // and the same watchdog `support::load_adjudicator` uses.
    let cli = support::adjudicator_cli(&[]);
    support::memory_guard::arm();
    let checkpoint = lfm2d::adjudicator::Checkpoint::load(
        cli.adjudicator_model.as_ref().unwrap(),
        cli.adjudicator_tokenizer.as_ref().unwrap(),
        cli.device,
        cli.device_index,
    )
    .expect("load the checkpoint");
    assert_eq!(checkpoint.execution.backend.as_str(), cli.device.as_str(), "loaded on the asked GPU");
    let model = &checkpoint.model;
    eprintln!("device: {:?}", checkpoint.execution.backend);
    let sampler = Sampler::start(model.device());
    let ids = needle_prompt(&checkpoint.tokenizer, 30_500);
    let (load_rss, load_gpu) = sampler.take();
    let mut loaded = serde_json::json!({"phase": "loaded", "peak_rss_mib": mib(load_rss), "peak_gpu_mib": mib(load_gpu)});
    for (part, bytes) in gpu_parts(model.device()) {
        loaded[format!("{part}_mib")] = mib(bytes).into();
    }
    eprintln!("{loaded}");
    const CHUNK: usize = 128;
    const DECODE: usize = 128;
    let argmax = |t: &candle_core::Tensor| -> u32 {
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
    };
    // A warm-up at a short depth so kernel compilation is not billed to 8k.
    {
        let mut state = model.new_state();
        let logits = model.forward(&ids[..256], &mut state).unwrap();
        let mut token = argmax(&logits);
        for _ in 0..8 {
            token = argmax(&model.forward(&[token], &mut state).unwrap());
        }
        logits.device().synchronize().unwrap();
        sampler.take();
    }
    // `LFM2D_LC_DEPTHS=4096,8192` measures fewer, shallower depths.
    let depths: Vec<usize> = std::env::var("LFM2D_LC_DEPTHS")
        .map(|v| v.split(',').map(|d| d.trim().parse().expect("a depth")).collect())
        .unwrap_or_else(|_| vec![128, 8192, 16384, 30_000]);
    for depth in depths {
        let base = gpu_bytes(model.device());
        let mut state = model.new_state();
        let begin = Instant::now();
        let mut logits = None;
        for chunk in ids[..depth].chunks(CHUNK) {
            logits = Some(model.forward(chunk, &mut state).unwrap());
        }
        let logits = logits.unwrap();
        logits.device().synchronize().unwrap();
        let prefill_s = begin.elapsed().as_secs_f64();
        let (prefill_rss, prefill_gpu) = sampler.take();
        let held = gpu_bytes(model.device());
        let mut token = argmax(&logits);
        let begin = Instant::now();
        for _ in 0..DECODE {
            let l = model.forward(&[token], &mut state).unwrap();
            token = argmax(&l);
        }
        let decode_s = begin.elapsed().as_secs_f64();
        let (decode_rss, decode_gpu) = sampler.take();
        eprintln!(
            "{}",
            serde_json::json!({
                "depth": depth,
                "prefill_ms": prefill_s * 1000.,
                "prefill_tok_s": depth as f64 / prefill_s,
                "decode_tokens": DECODE,
                "decode_tok_s": DECODE as f64 / decode_s,
                "peak_rss_mib_prefill": mib(prefill_rss),
                "peak_gpu_mib_prefill": mib(prefill_gpu),
                "peak_rss_mib_decode": mib(decode_rss),
                "peak_gpu_mib_decode": mib(decode_gpu),
                "state_gpu_delta_mib": mib(held.saturating_sub(base)),
                // 6 attention layers x K and V x 8 KV heads x 64 wide x f32,
                // at the power-of-two capacity the KV allocator rounds to.
                "state_kv_capacity_mib": mib(6 * 2 * 8 * 64 * 4 * (depth.max(128).next_power_of_two().min(model.context_length())) as u64),
            })
        );
        drop(state);
        drop(logits);
        // What the process still holds with no state alive: the ROCm
        // allocator parks freed blocks per size bucket and never returns
        // them, and prefill's score tensors take a new size every chunk.
        let after = gpu_bytes(model.device());
        eprintln!(
            "{}",
            serde_json::json!({"depth": depth, "gpu_mib_after_drop": mib(after),
                               "parked_growth_mib": mib(after.saturating_sub(base))})
        );
    }
    // The host builds each chunk's causal mask as a Vec<u8> of
    // 128 x (past + 128): quadratic in depth summed over a prefill.
    let begin = Instant::now();
    for past in (0..30_000).step_by(CHUNK) {
        let m = candle_transformers::utils::build_causal_mask(CHUNK, past, &candle_core::Device::Cpu).unwrap();
        std::hint::black_box(m);
    }
    eprintln!(
        "{}",
        serde_json::json!({"phase": "host_masks_30k_cpu_only", "ms": begin.elapsed().as_secs_f64() * 1000.})
    );
}
