# lfm2d

A System 1 service around the LFM2.5 suite, on candle: the bidirectional
encoder heads, plus a resident LFM2.5-8B-A1B opinion engine.
`README.md` is the what; this file is the working context.

Workspace of two crates: `lfm2-encoder` (library, repo root) and `lfm2d`
(HTTP daemon). The daemon serves whichever encoder heads a deployment
configures (embedder, Prompt-Router, token classifiers such as the
PII-Detector; ColBERT and sequence classification live in the library)
and, beside them, the **opinion engine**: LFM2.5-8B-A1B (MoE, GGUF) with reusable prefix state
and schema-constrained output. `/v1/opinion` is the fast read,
`/v1/adjudicate` the generative continuation, `/v1/probe` and
`/v1/tokenize` the instruments.

**Nothing in lfm2d knows what any one domain is.** The questions come from
prompt specs, and each spec names its own input (`input_label`). Consumers
own their specs and upload them at runtime (`POST /v1/opinion/specs`,
content-addressed). Shell judgement moved out on 2026-09-24: kaijutsu owns
the shell specs, the gate and the hook; `~/src/kaish-training-data` holds
the shell training and eval material. Its history is still in this repo's
git.

## Where things are

| thing | path |
|---|---|
| consumer contract (numbered invariants, never renumbered) | `docs/integration.md` |
| daemon API, deploy notes, known problems | `lfm2d/README.md` |
| deploy examples | `lfm2d/deploy/` — generic k8s manifests (encoder heads, `lfm2d-system1` on one GPU) and a quadlet unit; host-specific manifests live outside the repo |
| opinion engine: build, API, validation | `docs/lfm25-adjudicator.md` |
| writing a spec: what loads, what the fields do, how to measure one | `docs/writing-a-spec.md` |
| opinion API types and spec menu | `lfm2d/src/opinion_api.rs`; engine in `adjudicator.rs` |
| engine performance record | `docs/lfm25-*.md` (kernels, cache, fusion, GQA, prefill), numbers in `benchmarks/lfm25/results/` |
| lens / routing / knockout tooling | `benchmarks/lfm25/examine/`, bins in `lfm2d/src/bin/` |
| demos (System 1 acts, search, keyphrases) | `demo/` — the acts run on the `email-triage-v2` prop spec |
| checkpoint fixtures | `tests/fixtures/` — REAL Hub configs (refresh with curl from `https://huggingface.co/LiquidAI/<model>/raw/main/config.json`) plus the parity references `tests/reference/dump_*.py` regenerates |
| test specs for the opinion engine | `lfm2d/tests/fixtures/specs/` |

**Adaptation source**: `candle-transformers/src/models/lfm2.rs` in a candle
checkout (659 lines, the CAUSAL branch — config, blocks and attention are
what we adapt; drop the causal mask and KV cache). Dual MIT/Apache-2.0;
keep attribution in adapted files. Upstreaming is a decision Amy makes
after reading whatever AI policy candle has at the time.

**Python reference**: HF transformers `models/lfm2/modeling_lfm2.py` is the
causal one; the bidirectional variants live in each checkpoint's `auto_map`
custom code on the Hub. READ the checkpoint's own modeling file for head
shapes before implementing a head — never guess.

## GPU backends

ROCm (gfx1151, zorak) is the only GPU backend we run and measure.
**NVIDIA (CUDA) and Metal are future ports**; Intel is hypothetical until
System 1 is dialled in and demoed (Amy, 2026-09-24: maybe "a direct
backend on whatever Intel's ideal sdk is"). What keeps a port cheap:

- lfm2d itself is backend-neutral: the only backend-gated code is
  `lfm2d/src/device.rs`. The porting work lives in our candle fork
  (`tobert/candle`, pinned in `Cargo.toml`).
- ROCm compiles candle's own `candle-kernels/src/*.cu` through hipcc, so a
  kernel change goes in the shared source with arch guards
  (`RDNA2`/`RDNA3`), not in a ROCm-only copy. The generic path stays the
  default; a vendor fast path is opt-in (e.g. the MoE's grouped prefill
  behind `supports_grouped`, with `indexed_moe_forward` everywhere else).
- An unsupported backend fails loudly ("not implemented for …"); the CPU
  MoE is a reference, never a GPU fallback.
- The fork's CUDA side has never been compiled by nvcc (zorak has none).
  The first CUDA step is `cargo build --features cuda` plus the real-model
  tests (`LFM2D_TEST_GPU=cuda`, `demo/test_devices.sh cuda`) on the DGX
  Spark (tenchi, arm64).
- Numbers are per backend and per target: re-measure on the new stack,
  never carry a threshold across. `snapshot_id` hashes the device identity
  (`rocm:gfx1151:hip7.2`) and the candle revision; a CUDA port should
  give its device the compute capability the same way (today it reports
  the bare `cuda`).

## Checkpoint facts (fixture-verified — trust these over docs)

- Hybrid stack via `layer_types` (conv-heavy, interleaved full_attention).
  The 230M base has **14** layers, the 350M family 16. hidden 1024,
  16 heads / 8 kv-heads, vocab 65536.
- `"full_attn_idxs"` ships as a present key with a **null** value across
  the family — absent is not the same as null here.
- PII detector: BIOES over 40 entity types, **161 labels**, five of them
  `credential.*` (api_key, jwt, private_key, password, connection_string)
  — it is a secrets detector too.
- Prompt-Router: `Lfm2BidirForSequenceRouting`, `rule_proj_dim` 256 — it
  scores prompts against rule projections (zero-shot-shaped), NOT a fixed
  label softmax.
- rope theta appears BOTH flat (`rope_theta`) and structured
  (`rope_parameters.rope_theta`) across checkpoints;
  `Lfm2EncoderConfig::rope_theta()` resolves precedence, default 1e6.

## Read vocabulary at runtime

Never hard-code a label name, count or index, a spec's field or option
names, or an `input_label` — not in a consumer, not in a test fixture's
expectations, not in a doc example. Read them from `GET /v1/models` and
`GET /v1/opinion/specs`. The vocabulary has changed wholesale between
checkpoints and specs and will again. `docs/integration.md` is the
contract version of this.

## Measuring the opinion engine

Rules the adjudicator work paid for; they hold for any spec:

- Measure on **our** stack, per backend. llama.cpp is a cheap
  cross-check; the same GGUF gives different distributions on ROCm, CPU
  and llama.cpp, and prefix-cache results do not transfer.
- An eval harness renders the prompt **exactly** as the daemon does
  (`rendered: true`, or `/v1/probe` with `decode_from`), and a replay
  orders fields by the spec's `required`, which is emission order. Hash the
  rendered prompt into the results.
- Log the raw probability mass in the answer set beside every
  constrained read. Near-zero mass means the model was never asked, not
  that it answered badly.
- A pass-through rate and a recall number are separate instruments; a
  recall without its false-alarm count beside it is not a result.
- A looked-at split stops being a test. Confirm on one that was not.
- Most inputs are ordinary and the engine's job is to pass them through;
  a spec that flags often is wrong before it is measured.

## Conventions

Amy's global CLAUDE.md applies (TDD, 改善, loud failures). Verify against
fixtures, not documentation — the fixtures have contradicted plausible
assumptions repeatedly, starting on day 0. Corpora never live in the repo;
tools print aggregates, never raw rows. `CLAUDE.md` is a symlink to this
file.

Every test that loads real weights calls `memory_guard::arm()`
(`tests/support/memory_guard.rs`: aborts under `LFM2_TEST_MEM_FLOOR_GIB`,
default 8, of MemAvailable, which sees GTT where RSS does not); real 8B
tests load through `lfm2d/tests/support` (explicit GPU, never CPU or auto).
Wrap every cargo build/test and GPU or >8 GiB run in
`flock ~/.cache/zorak-heavy.lock <cmd>`, one build or run per hold.
