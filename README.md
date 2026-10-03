# lfm2d

**A System 1 for software, built on LiquidAI's LFM2.5 models.** Rust on
[candle] with no Python in the serving path, serving from one resident
process on an AMD, NVIDIA or Intel GPU.

Before an agent runs a command, an app sends an email, or a person acts on
an idea, something should take a quick look: a fast, cheap first read that
lets the ordinary majority through and flags the rest for a slower judge (a
bigger model or a person, the System 2). lfm2d is that first read.

You bring the question as a **spec** (a system prompt and a small JSON
schema). lfm2d loads it into a resident **LFM2.5-8B-A1B**, a mixture of
experts with about 1B parameters active per token, and answers with
**odds, not prose**: the model describes the input in the spec's fields, then
every option of the field you asked about is scored from that one
description, each option's tokens teacher-forced.
The response carries each option's probability *and* the raw probability
mass the model put on the whole answer set, so a caller can tell an answer
from a model that was never really asked. lfm2d never picks a winner; your
code sets the threshold.

Beside the opinion engine, the same daemon serves LiquidAI's LFM2.5 encoder
heads: embeddings, PII and secrets detection, and prompt routing.

## What an opinion looks like

Upload a spec (content-addressed: its id is the sha256 of its bytes), then
ask:

```sh
curl -s localhost:8088/v1/opinion/specs --data-binary @demo/web/static/life-decision-v2.json
curl -s localhost:8088/v1/opinion -H 'content-type: application/json' -d '{
  "spec": "0995af933c7763ff7b6de306cd3542e8ad39ad93fe0e0cdd510430f7020df042",
  "state": {"input": "microwave a fork to see what happens"},
  "questions": [{"field": "verdict"}]
}'
```

```jsonc
{
  "described": [
    {"field": "effect", "value": "The fork will melt and lose its structural integrity, potentially causing burns or damage to the microwave."},
    {"field": "scope",  "value": "someone else"},
    {"field": "undo",   "value": "hard"}
  ],
  "answers": [{
    "field": "verdict",
    "options": [
      {"option": "go",   "prob": 0.00006, "logprob": -9.77},
      {"option": "wait", "prob": 0.339,   "logprob": -1.08},
      {"option": "stop", "prob": 0.661,   "logprob": -0.41}
    ],
    "sequence_mass": -0.000016,   // log of the raw mass on {go, wait, stop}: 99.998%
    "margin": 0.32
  }],
  "cache": {"prefix": "hit", "state": "miss", "described": "miss"},
  "prefill_ms": 171, "describe_ms": 1161, "read_ms": 102
}
```

(trimmed; the full response also names the model, weights, device and
candle revision that produced it.)

- **The spec's prefix is prefilled once**, when you upload it, and kept
  resident as a snapshot of the hybrid model's state. Each request only
  prefills its own input.
- **Describe, then read.** The fields in front of the asked one are the
  model's only reasoning; it writes them greedily under a JSON grammar
  compiled from your schema. Asking again reuses the description, so a
  repeat costs one read.
- **Facts** (`state.facts`) are trusted context placed before the input,
  such as a project's rules or what the user asked for. One line of facts
  can flip an answer.
- **Deterministic.** Greedy decoding on a fixed stack: thirty inputs read
  twice from cold gave bit-identical descriptions and probabilities.

The model has no domain built in. The questions come from the specs, and
[writing a spec](docs/writing-a-spec.md) covers what a spec may contain,
what the engine does with it, and how to measure one.

## How well it reads

Measured 2026-09-25 through the running daemon on sets it was not tuned on
([`benchmarks/system1/`](benchmarks/system1/README.md) has the method,
every table, and the scripts). On 213 everyday proposals (61 ordinary, 80
worth sleeping on, 72 physically dangerous):

| spec | ordinary → go | dangerous → go | dangerous → stop |
|---|---|---|---|
| life-decision **v1** | 3 / 61 | 0 / 72 | 14 / 72 |
| life-decision **v2** | 46 / 61 | 5 / 72 | 37 / 72 |
| **gate** variant | 53 / 61 | 1 / 72 | 2 / 72 |

The model is the same in every row; only the spec changes. v1's rules
said anything touching someone else is a wait, and the model decided
nearly everything touches someone else ("make a cup of tea": wait, 93%).
Rewording the rules fixed pass-through. Where v2 still errs, its
description was already wrong ("put water on the grease fire" was described
as extinguishing it safely): the verdict follows what the model believes
happens next.

On support email the same engine is weaker: routing between "a template
can close it" and "a person must read it" gives an AUC of 0.76-0.81.

**Speed**, one question on a busy workstation (upper bounds): a fresh
short input in about 0.6 s p50, a support email in 0.85 s, and a repeat of
either in 54-65 ms.

## Demos

Browser pages that play themselves against a live daemon, sized for
recording ([`demo/web/`](demo/web/README.md)):

- **The Sour Note**: LFM2.5 reads a passage and every token plays a note
  whose dissonance is the model's surprise. Change one word and you hear
  it.
- **Would LFM Let You?**: a game show. Everyday ideas go through the
  life-decision spec, and the traffic light is lit by the odds.
- **Two Worlds**: the same command read alone, then with one line of facts
  from two different worlds.
- **House Rules**: an agent's `AGENTS.md`; the daemon's own embedder finds
  the rule that governs a command, and quoting that one rule changes the
  answer more usefully than quoting the whole file.
- **Everything Is a Command**: what happens when the input doesn't fit the
  spec. The mass stays high, so the harness has to choose the inputs.

[`demo/show.py`](demo/README.md) runs the same ideas in a terminal against
an email-triage spec.

## Run it

The opinion engine runs on a GPU. Each backend is a cargo feature over the
same code; pick one at build time and name it with `--device`:

| vendor | feature | hardware-tested on | status |
|---|---|---|---|
| AMD | `rocm` | Radeon 8060S (Strix Halo, gfx1151) | where we run and measure; production |
| NVIDIA | `cuda` | DGX Spark (GB10, sm_121, CUDA 13.0, arm64) | encoder parity, e2e and the real 8B suite pass |
| Intel | `sycl` | Arc Pro B70 (Xe2, oneAPI 2026.1) | encoder parity and every endpoint exercised; one open opinion bug ([daemon README](lfm2d/README.md#problems-noted-not-fixed)) |
| Apple | `metal` | — | a candle backend; not built or verified here |

The CPU path is a slow reference for the opinion engine; the encoder heads
run well on CPU. Numbers do not transfer between backends: the same weights
read differently on each, so `snapshot_id` names the device. Building needs
that backend's toolchain (ROCm, the CUDA toolkit, or oneAPI with `icpx`)
and a C compiler (candle-core links oniguruma through `tokenizers`).

```sh
# the model and its tokenizer
hf download LiquidAI/LFM2.5-8B-A1B-GGUF LFM2.5-8B-A1B-Q5_K_M.gguf --local-dir .models/LFM2.5-8B-A1B
hf download LiquidAI/LFM2.5-8B-A1B tokenizer.json --local-dir .models/LFM2.5-8B-A1B

cargo build --release -p lfm2d --features rocm   # or cuda, sycl
./target/release/lfm2d --device rocm --bind-addr 127.0.0.1:8088 \
  --adjudicator-model .models/LFM2.5-8B-A1B/LFM2.5-8B-A1B-Q5_K_M.gguf \
  --adjudicator-tokenizer .models/LFM2.5-8B-A1B/tokenizer.json
```

Add `--embedder-dir`, `--router-dir` or `--token-classifier-dir` to serve
encoder heads from the same process. `lfm2d/Containerfile.rocm` builds a
container; `lfm2d/deploy/` has Kubernetes and quadlet examples.

| where | what |
|---|---|
| [`lfm2d/README.md`](lfm2d/README.md) | the daemon: flags, every endpoint, deployment, known problems |
| [`docs/writing-a-spec.md`](docs/writing-a-spec.md) | writing and measuring a spec |
| [`docs/integration.md`](docs/integration.md) | the consumer contract, as numbered invariants |
| [`docs/encoders.md`](docs/encoders.md) | the encoder library: heads, parity, retrieval quality, costs |
| [`docs/development.md`](docs/development.md) | building, fetching checkpoints, and which tests need which weights |
| [`docs/lfm25-adjudicator.md`](docs/lfm25-adjudicator.md) | the opinion engine's build and validation record |

## The encoder heads

The repo root is also `lfm2-encoder`, a library that implements LiquidAI's
*bidirectional* LFM2.5 encoders (upstream candle has only the causal
LFM2).
Every head a LiquidAI checkpoint ships has a parity test against
activations dumped from the real weights by LiquidAI's own modeling code.
Sequence classification, which no LiquidAI checkpoint ships, is tested on
synthetic weights.

| checkpoint | head |
|---|---|
| `LFM2.5-Embedding-350M` | pooled 1024-dim embeddings (served at `/embed`) |
| `LFM2.5-ColBERT-350M` | late interaction, MaxSim (library) |
| `LFM2.5-Encoder-350M-PII-Detector` | token classification: 161 BIOES labels, credentials and secrets included |
| `LFM2.5-Encoder-350M-Prompt-Router` | zero-shot routing against caller-written lanes |
| your own fine-tune | sequence classification over the shared trunk (library) |

Details, numbers and caveats (the embedder is asymmetric; batching changes
embeddings) are in [docs/encoders.md](docs/encoders.md).

## License & attribution

MIT OR Apache-2.0, matching candle. Trunk block implementations are
adapted from [candle-transformers]' `lfm2.rs` (© the candle authors, MIT
OR Apache-2.0); attribution retained in source where adapted. The opinion
engine runs on a [fork of candle](https://github.com/tobert/candle) carrying
the ROCm backend, the CUDA MoE work and a SYCL backend for Intel GPUs
(Petr Gadorek's backend, brought up on the LFM2.5 engine by
[J. Gavin Ray](https://github.com/jgavinray)), pinned in `Cargo.toml`.

[candle]: https://github.com/huggingface/candle
[candle-transformers]: https://github.com/huggingface/candle/tree/main/candle-transformers
