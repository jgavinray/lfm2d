//! The byte bound the state caches charge (`StateSize::bytes`) against what
//! a held state actually costs on the device, on the real LFM2.5-8B-A1B.
//!
//! Measured through the kernel's own per-process account of device memory
//! (`/sys/class/kfd/kfd/proc/<pid>/vram_*`, which on gfx1151 is host memory
//! the GPU maps). The candle ROCm allocator parks freed blocks by exact size
//! and hands them back to the next allocation of that size, so a block freed
//! earlier in a process hides the cost of a later state. Each case therefore
//! runs in a process of its own and frees nothing before it measures: build
//! a base state, hold 4 states of the case's shape, then 4 more, and divide
//! the second difference by 4 (scratch of the same shapes comes back out of
//! the pool by then and cancels).
//!
//! Two shapes, both how the caches get their states:
//! - `prefill`: a fresh state forwarded `len` tokens in 128-token chunks
//!   (a ready prompt, a checkpoint);
//! - `fork`: a clone of the held base forwarded 8 more tokens one at a time
//!   (a described state, a tail prefix): past the first, a clone cannot
//!   append into the base's storage and copies into its own.
//!
//!   LFM2_MODELS_DIR=... flock ~/.cache/zorak-heavy.lock cargo test -p lfm2d \
//!     --release --features rocm --test state_bytes_real -- --ignored --nocapture
mod support;
use candle_transformers::models::quantized_lfm2_moe::State;
use lfm2d::adjudicator::Checkpoint;

const CASE_ENV: &str = "LFM2D_STATE_BYTES_CASE";

fn device_bytes() -> u64 {
    let dir = format!("/sys/class/kfd/kfd/proc/{}", std::process::id());
    std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{dir}: {e}"))
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("vram_"))
        .map(|e| std::fs::read_to_string(e.path()).unwrap().trim().parse::<u64>().unwrap())
        .sum()
}

/// One case in this process: prints `shape len per_state bound`.
fn measure(shape: &str, len: usize) {
    let cli = support::adjudicator_cli(&[]);
    support::memory_guard::arm();
    let checkpoint = Checkpoint::load(
        cli.adjudicator_model.as_ref().unwrap(),
        cli.adjudicator_tokenizer.as_ref().unwrap(),
        cli.device,
        cli.device_index,
    )
    .expect("load");
    assert_eq!(checkpoint.execution.backend.as_str(), "rocm", "the account read here is KFD's");
    let model = &checkpoint.model;
    let text = "The desk arrived with a cracked leg and the courier left it in the rain. ".repeat(400);
    let all = checkpoint.tokenizer.encode(text.as_str(), false).unwrap().get_ids().to_vec();
    let prefill = |len: usize| -> State {
        let mut state = model.new_state();
        for chunk in all[..len].chunks(128) {
            model.forward(chunk, &mut state).unwrap();
        }
        model.device().synchronize().unwrap();
        state
    };
    let base = prefill(len);
    let make = || match shape {
        "prefill" => prefill(len),
        "fork" => {
            let mut state = base.clone();
            for &t in &all[len..len + 8] {
                model.forward(&[t], &mut state).unwrap();
            }
            model.device().synchronize().unwrap();
            state
        }
        other => panic!("unknown shape {other}"),
    };
    let mut held: Vec<State> = (0..4).map(|_| make()).collect();
    let four = device_bytes();
    held.extend((0..4).map(|_| make()));
    let eight = device_bytes();
    let bound = checkpoint.state_size.bytes(held[0].len()) as u64;
    println!("CASE {shape} {} {} {bound}", held[0].len(), eight.saturating_sub(four) / 4);
}

#[test]
#[ignore = "loads the 6 GB LFM2.5-8B-A1B GGUF once per case; ROCm host only"]
fn the_state_byte_bound_holds_on_the_device() {
    if let Ok(case) = std::env::var(CASE_ENV) {
        let (shape, len) = case.split_once(':').unwrap();
        measure(shape, len.parse().unwrap());
        return;
    }
    let mut rows = Vec::new();
    for len in [200usize, 1000, 2000, 3000] {
        for shape in ["prefill", "fork"] {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--ignored", "--exact", "the_state_byte_bound_holds_on_the_device", "--nocapture"])
                .env(CASE_ENV, format!("{shape}:{len}"))
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout);
            let line = stdout
                .lines()
                .find_map(|l| l.strip_prefix("CASE "))
                .unwrap_or_else(|| panic!("{shape}:{len}: {stdout}\n{}", String::from_utf8_lossy(&out.stderr)));
            let v: Vec<&str> = line.split_whitespace().collect();
            let (state_len, per_state, bound): (usize, u64, u64) =
                (v[1].parse().unwrap(), v[2].parse().unwrap(), v[3].parse().unwrap());
            rows.push((v[0].to_string(), state_len, per_state, bound));
        }
    }
    eprintln!("| shape | len | measured bytes/state | bound | measured / bound |");
    eprintln!("|---|---|---|---|---|");
    for (shape, len, per_state, bound) in &rows {
        eprintln!("| {shape} | {len} | {per_state} | {bound} | {:.3} |", *per_state as f64 / *bound as f64);
    }
    for (shape, len, per_state, bound) in rows {
        assert!(per_state > 0, "{shape} at {len}: the account did not move");
        assert!(per_state <= bound, "{shape} at {len} tokens holds {per_state} bytes, over the {bound}-byte bound");
    }
}
