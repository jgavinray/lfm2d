# Building and testing

How to build lfm2d, fetch the checkpoints, and run the tests. The
project overview is the [root README](../README.md); the daemon's flags
and API are in [`lfm2d/README.md`](../lfm2d/README.md).

Build (CPU is the default; ROCm is the one GPU backend we run and measure.
The `cuda` and `metal` features exist but have not been ported or
measured — see `lfm2d/README.md`):

```sh
cargo build --release                    # library + examples, CPU
cargo build --release -p lfm2d           # the daemon, CPU
cargo build --release -p lfm2d --features rocm
```

A C compiler is needed: candle-core pulls in oniguruma through
`tokenizers`.

Checkpoints are never downloaded at runtime. Fetch them into `.models/`
(gitignored) with the Hugging Face CLI:

```sh
for m in LFM2.5-Embedding-350M LFM2.5-ColBERT-350M \
         LFM2.5-Encoder-350M-PII-Detector LFM2.5-Encoder-350M-Prompt-Router; do
  hf download "LiquidAI/$m" --local-dir ".models/$m"
done
# the opinion engine: the GGUF, plus the tokenizer from the base repo
hf download LiquidAI/LFM2.5-8B-A1B-GGUF LFM2.5-8B-A1B-Q5_K_M.gguf \
  --local-dir .models/LFM2.5-8B-A1B
hf download LiquidAI/LFM2.5-8B-A1B tokenizer.json --local-dir .models/LFM2.5-8B-A1B
```

Tests that need weights **fail loudly** rather than skipping, naming the
missing file (most print the `hf download` command too), so a plain
`cargo test --workspace` wants all of the above except the GGUF:

| weights | tests |
|---|---|
| none | config parsing, guards, the daemon's API/stub/shutdown suites |
| `LFM2.5-Embedding-350M` | `trunk_parity`, `embedding_parity`, `retrieval_quality` |
| `LFM2.5-ColBERT-350M` | `colbert_parity` |
| `LFM2.5-Encoder-350M-PII-Detector` | `pii_parity`, `lfm2d/tests/integration_real_spans` |
| `LFM2.5-Encoder-350M-Prompt-Router` | `router_parity` |
| `LFM2.5-8B-A1B/tokenizer.json` | `lfm2d/tests/tokenize_api`, `probe_tokenize_telemetry_safety`, and unit tests in `lfm2d/src/{adjudicator,tokenize_api}.rs` |
| the GGUF (`#[ignore]`d; minutes on a GPU, hours on CPU) | `opinion_real`, `probe_real`, `spec_registry_real`, `constrained_decoding` |
| Embedding + Router + PII and a GPU (`#[ignore]`d) | `lfm2d/tests/device_real`, via `demo/test_devices.sh` |

`LFM2_MODELS_DIR` and `LFM2_TOKEN_CLF_DIR` point the tests elsewhere
(a git worktree has no `.models/`). `demo/test_devices.sh <rocm|cuda|sycl|metal>`
runs the CPU-vs-GPU agreement gate on a GPU host.

