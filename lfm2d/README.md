# lfm2d

HTTP sidecar/daemon serving `lfm2-encoder` heads (embedder, Prompt-Router,
token classifiers such as the PII-Detector) and the LFM2.5-8B-A1B opinion
engine ("Resident causal adjudicator" below) over a Unix domain socket
and/or TCP, from ONE process. Built because:

- candle's `from_mmaped_safetensors` copies every tensor into private
  anonymous RSS on load — N processes loading the same 350M checkpoint cost
  N × ~1.4 GiB, not one shared mapping.
- A single forward pass already saturates ~13.6 of this box's cores
  (measured) — per-request concurrency has no headroom to spend, so this
  daemon serves every request through ONE serial inference worker thread by
  design, not as an unoptimized bottleneck.
- `kaibo` cannot link `candle-core` at all (it hard-wires
  `tokenizers`+`onig`, breaking kaibo's musl static-build invariant) — an
  HTTP boundary is the only way kaibo can reach these models.

See `src/lib.rs`'s module docs for the full architecture writeup (worker
thread / channel design, the `InferenceEngine` trait that decouples router
tests from candle, and the two response conventions this API uses).

## Building

```sh
cargo build --release -p lfm2d
```

The default build is CPU-only. A GPU build selects a device at startup:

```sh
# AMD, with the ROCm toolchain/runtime installed
cargo build --release -p lfm2d --features rocm
target/release/lfm2d --device auto --threads 8 \
  --embedder-dir '.models/LFM2.5-Embedding-350M' --bind-addr '127.0.0.1:8088'
```

Optional `cuda`, `sycl` and `metal` features build the other Candle
backends. One code path serves all of them; only `src/device.rs` is
backend-gated, and the porting work lives in the candle fork.

| backend | build needs | hardware-tested | MoE prefill | device identity |
|---|---|---|---|---|
| `rocm` | ROCm toolchain (hipcc, rocBLAS) | Radeon 8060S, gfx1151: everything; production | grouped (shared kernels) | `rocm:gfx1151:hip7.2` |
| `cuda` | CUDA toolkit (nvcc) | DGX Spark, GB10 sm_121, CUDA 13.0, aarch64, since 2026-09-27: encoder parity, `demo/e2e.py`, real 8B suite | grouped (shared kernels, [docs/lfm25-grouped-prefill.md](../docs/lfm25-grouped-prefill.md)) | `cuda:sm_121:nvcc13.0:drv580.173.02` |
| `sycl` | oneAPI 2026.1+ (`icpx` on `PATH`), Level Zero, Xe2+ GPU | Arc Pro B70 since 2026-10-02: encoder parity, every endpoint ([docs/lfm2d-b70-endpoint-exercise.md](../docs/lfm2d-b70-endpoint-exercise.md)) | grouped inside `indexed_moe_forward` at `batch*topk >= 64` ([docs/lfm25-sycl-b70.md](../docs/lfm25-sycl-b70.md)) | `sycl:2026.1:Intel(R) Arc(TM) Pro B70 Graphics` |
| `metal` | macOS | not verified | — | bare `metal` |

The SYCL path differs from ROCm and CUDA in where the grouped prefill is
chosen: those two report `supports_grouped_moe` and take routing the engine
packs once per layer, while SYCL reports `false` and switches to its grouped
GEMM inside each indexed call. F8E4M3 is unimplemented on SYCL and errors
loudly. The `sycl` build compiles `candle-sycl-kernels` with `icpx`; a
plain `cargo build` never touches oneAPI. After a candle revision bump,
clear `target/release/build/candle-sycl-kernels-*` (see the problems
list below).

Candle core/nn/transformers are pinned together to one revision of our fork
(`tobert/candle`, branch `lfm25-batch`) because crates.io 0.11 carries no
ROCm or SYCL backend and none of the MoE kernels.

`auto` tries compiled backends in ROCm/CUDA/Metal/SYCL order, then uses CPU if
device initialization is unavailable. Every failed probe/reason is logged,
and execution metadata reports the **selected** backend. Use `--device cpu`
to force CPU or `--device rocm` (etc.) to require a GPU; an explicit GPU never
falls back. `--device-index` selects the backend's device ordinal.
The LFM2.5-8B-A1B adjudicator is the exception to `auto`'s CPU step: its CPU
MoE is a reference (~29 GB of f32 experts, then linear in tokens), so `auto`
resolving to CPU refuses to load it; pass `--device cpu` for a reference run.

Fallback is startup-only: checkpoint/configuration errors, model smoke-test
failures, and inference failures remain errors, without retrying on CPU.
GPU builds still require their linked runtime libraries even in CPU mode;
missing shared libraries prevent startup before selection can run. The
ordinary CPU build remains portable to machines without those libraries.
The container recipes below build the ordinary CPU variant; a GPU container
also needs the runtime libraries and device access.

Changing the execution device can change scores and flip near-tied
readings. Validate deployment calibration before moving a head a decision
depends on; the numerical parity gate is not a replacement for traffic
evaluation.

Run the mandatory device gate when changing backend selection/execution:

```sh
bash demo/test_devices.sh rocm   # or cuda, sycl, metal
```

It builds once with the chosen backend, compares real embedding/router/PII
heads on CPU and GPU, then runs the real HTTP suite under explicit
CPU, explicit GPU, automatic GPU selection, and automatic CPU fallback after
an invalid GPU ordinal. A missing GPU/driver/weight file fails this gate.
The hardware Rust test is opt-in (`--ignored`, invoked by this script); normal
CPU-only tests cover selection success/failure through an injected probe.

After proving the pinned ROCm patch set locally, polish it for upstream as
separate work. Amy will review the upstream policy and handle posting; the
local service work does not submit changes upstream.

Container image (build context must be the repo ROOT, since `lfm2d`
depends on the parent crate via `path = ".."`):

```sh
podman build -t lfm2d:latest -f lfm2d/Containerfile .
```

See `deploy/lfm2d.container` for an example podman quadlet unit (models
volume-mounted, `CPUWeight=`/`Nice=` set, both socket and TCP examples,
OTLP env wiring) and `deploy/k8s.yaml` for a complete Deployment + Service
(probes, a measured memory recipe, CPU requests without limits, OTLP env
wiring — see "Deploying on k8s" below for the reasoning behind each).

## Configuration

CLI flags, each with an env-var fallback (`clap`'s `env` feature):

| Flag | Env var | Meaning |
| --- | --- | --- |
| `--embedder-dir` | `LFM2D_EMBEDDER_DIR` | `Lfm2Embedding`-shaped checkpoint dir; backs `/embed` |
| `--router-dir` | `LFM2D_ROUTER_DIR` | Prompt-Router checkpoint dir; backs `/v1/route` |
| `--token-classifier-dir` (repeatable) | `LFM2D_TOKEN_CLASSIFIER_DIR` (comma-separated) | `Lfm2TokenClassifier`-shaped checkpoint dir(s) — REPEATABLE, unlike the two heads above; backs `/v1/spans`, `/v1/spans/credentials` |
| `--dtype` | `LFM2D_DTYPE` | `f32` (default), `f16` or `bf16`, for every head. Read the flag's own help before reaching for `f16`: these checkpoints ship f32 natively, so f16 is a real loss of resolution at ~1.6× latency, and `LFM2.5-Embedding-350M` ships bf16, where bf16→f16 can produce inf/0 rather than rounding. It is a memory lever, not a speed one |
| `--device` | `LFM2D_DEVICE` | `auto` (default), `cpu`, `rocm`, `cuda`, `sycl`, or `metal`; requires corresponding compiled GPU feature |
| `--device-index` | `LFM2D_DEVICE_INDEX` | GPU ordinal (default `0`); values exceeding the driver's signed 32-bit range are refused |
| `--log-input-hash` | `LFM2D_LOG_INPUT_HASH` | `true`/`false`, must be spelled out (not a bare flag); default `false`. Attaches a hash of `/v1/spans`/`/v1/spans/credentials` request text — never the text itself — to that call's trace/log span; see "Observability" below |
| `--socket-path` | `LFM2D_SOCKET_PATH` | Unix domain socket to serve on |
| `--bind-addr` | `LFM2D_BIND_ADDR` | TCP address to serve on, e.g. `127.0.0.1:8088` |
| `--threads` | `LFM2D_THREADS` | Size of rayon's global thread pool (candle's matmul runs on it transitively), set BEFORE any model load. Defaults to `std::thread::available_parallelism()` |
| `--adjudicator-model` | `LFM2D_ADJUDICATOR_MODEL` | LFM2.5-8B-A1B GGUF; with `--adjudicator-tokenizer`, enables the opinion engine (`/v1/opinion`, `/v1/adjudicate`, `/v1/chat`, `/v1/opinion/specs`, `/v1/adjudicator`, `/v1/probe`). Each alone is refused |
| `--adjudicator-tokenizer` | `LFM2D_ADJUDICATOR_TOKENIZER` | The matching Hugging Face `tokenizer.json`, checked against the GGUF's vocabulary at load |
| `--adjudicator-context` | (none) | Context budget including output, default `4096`, range `128`–`32768` (`docs/lfm25-adjudicator.md`, "Context budget") |
| `--adjudicator-repeat-penalty` | (none) | Sign-aware repetition penalty, default `1.05` (the checkpoint author's), range `1.0`–`2.0`; part of every spec's `snapshot_id` |
| `--opinion-spec` (repeatable) | `LFM2D_OPINION_SPECS` (comma-separated) | A boot-time spec, named by its file stem. Optional: with none the menu starts empty and fills by upload. Needs the adjudicator. No spec is a default |
| `--opinion-spec-capacity` | `LFM2D_OPINION_SPEC_CAPACITY` | How many uploaded specs stay resident (LRU), default `8`; boot specs don't count and are never evicted |
| `--no-probe` | `LFM2D_PROBE` (`0` disables) | Removes `/v1/probe`; `/v1/opinion` and `/v1/adjudicate` are unaffected |
| `--chat-checkpoint-budget-mib` | `LFM2D_CHAT_CHECKPOINT_BUDGET_MIB` | Memory for chat checkpoints (`/v1/chat`, tail reads), default `16384`; least recently used evicted first; must hold a turn's two checkpoints at the full context |
| `--state-cache-budget-mib` | `LFM2D_STATE_CACHE_BUDGET_MIB` | Memory for every cached model state (ready prompts, described states, chat tail prefixes), default `16384`; least recently used evicted first across specs; the per-spec counts still apply inside it; must hold one full-context state |

Standard OTEL env vars also apply (`OTEL_EXPORTER_OTLP_ENDPOINT`,
`OTEL_SERVICE_NAME` default `lfm2d`, ...) — see "Observability" below; none
of these are `clap` flags, they're read directly by the OTLP exporters and
`main.rs`.

At least one of `--socket-path`/`--bind-addr` and at least one of
`--embedder-dir`/`--router-dir`/`--token-classifier-dir`/
`--adjudicator-model` are required (an opinion-engine-only daemon is
valid), and a retired env var (`LFM2D_ADJUDICATOR_PROMPT`,
`LFM2D_CASCADE_*`, `LFM2D_CLASSIFIER_DIR`,
`LFM2D_CANDIDATE_CLASSIFIER_DIR`) is refused by name — startup
fails loudly (`exit 2`, config problem named) otherwise. Model loading
itself is synchronous and happens before the socket is ever bound: a bad
checkpoint path or an incompatible config also fails loudly at startup
(`exit 1`), never lazily on the first request.

Only ONE model per head kind is supported for the embedder and router
heads — `/embed` and `/v1/route` don't take a model-selection parameter, so
a deployment serving two routers, say, needs two `lfm2d` processes. Token classifiers are the one exception: `--token-classifier-dir`
is repeatable, and `/v1/spans`/`/v1/spans/credentials` take an optional
`"model"` field to pick among them — see the API section below.

## API (v1)

The semantic contract a consumer programs against — numbered invariants,
cited by number from outside this repo — is `docs/integration.md` at the
repo root. It never renumbers. This section is the reference for the
endpoints themselves; where the two overlap, the contract wins.

```
GET  /healthz                 liveness
GET  /readyz                  readiness
GET  /v1/models               [{id, kind, weight_hash, labels?, hidden_size}]
POST /embed                   TEI-compat  {"inputs": str|[str], "kind"?: document|query}
POST /v1/route                {"input": str, "routes": [str]}  -> RAW cosines
POST /v1/spans                {"inputs": str|[str], "model"?: str}
POST /v1/spans/credentials    same shape; credential.* entities only
POST /v1/tokenize             {"model": str, "text": str, "context"?: str}
POST /v1/probe                {"text": str} | {"messages": [...]}  -- adjudicator only, on by default
GET  /v1/adjudicator          checkpoint identity                   -- adjudicator only
GET  /v1/opinion/specs        the spec menu (boot + uploads)        -- adjudicator only
POST /v1/opinion/specs        body: a spec's exact bytes            -- adjudicator only
DELETE /v1/opinion/specs/{id}                                       -- adjudicator only
POST /v1/opinion              {"spec", "state": {"input", "facts"?}, "questions": [...], "context"?: {"checkpoint"}}
POST /v1/chat                 {"system"?, "tools"?} | {"from"}, "messages": [...], "stream"?  -- adjudicator only
POST /v1/adjudicate           {"spec", "input", ...}                -- spec required
```

The input field is `inputs` (TEI's spelling), not `texts`. `/v1/route`
takes singular `input` because it scores one text against many routes.

- `GET /healthz` → `200 "ok"` — process alive, never touches the worker.
- `GET /readyz` → `200` once every configured model has loaded, `503`
  before.
- `GET /v1/models` → `[{id, kind, weight_hash, labels?, hidden_size}]` —
  every loaded model. `kind` is `embedder`/`router`/`token_classifier`/
  `adjudicator`. `labels` is present for a token classifier (its distinct
  `entity_types()`, BIOES prefix
  stripped and deduped — NOT the raw `id2label`, which on the PII detector
  is 161 entries wide for only 40 distinct entities). `weight_hash` is
  sha256 over the checkpoint's `model.safetensors`, 64 lowercase hex chars
  — the audit trail this whole daemon exists partly to satisfy (kaish
  approval-chain rulings).
- `POST /embed` — TEI-compatible-ish. `{"inputs": "text"}` or
  `{"inputs": ["a", "b"]}`, optional `"kind": "query"|"document"` (default
  `document`; this is the E5-style asymmetric-embedding side selector —
  see `candle_lfm2_encoder::TextKind`'s docs on why picking wrong quietly
  degrades retrieval). Response: `[[f32, ...]]`, one pooled vector per
  input, in input order. `model_id`/`weight_hash` travel as
  `X-Model-Id`/`X-Model-Weight-Hash` response headers, not in the JSON body
  — keeping the body wire-compatible with plain TEI clients that don't
  know these headers exist.
- `POST /v1/route` — `{"input": str, "routes": [str, ...]}` → `{"model_id",
  "weight_hash", "routes": [{"route", "cosine"}, ...]}`. **RAW COSINE
  ONLY** — this router's softmax is route-count arithmetic and carries no
  confidence information (see `candle_lfm2_encoder::routing`'s module
  docs), so it is never exposed on this endpoint.
- `POST /v1/spans` — TEI-style, `{"inputs": str | [str]}` plus an OPTIONAL
  `"model"` (which loaded `--token-classifier-dir` head answers; required
  as a 400 naming every loaded id when 2+ are loaded, implicit when
  exactly 1 is) → `[[{"start", "end", "entity", "score"}, ...], ...]`, a
  bare array of arrays, one span list per input, same TEI-shaped
  convention as `/embed`. Audit pair travels as `X-Model-Id`/
  `X-Model-Weight-Hash` headers, same reason.
  - **`start`/`end` are UTF-8 BYTE offsets, not codepoints/chars, not
    UTF-16 code units.** A Rust caller (kaibo, the first consumer) can
    slice `&text[start..end]` directly; a Python/JS caller must re-encode
    to UTF-8 bytes first — indexing a Python `str` or a JS string with
    these numbers directly is wrong on any non-ASCII input.
  - **The matched text is NEVER returned, by design, not as a missing
    feature** — no `word`/`quote`/`text` field, not even opt-in. The
    caller already has the text it just sent; echoing a credential back
    would only create a second copy of it in every log this response
    passes through. Stricter than GCP DLP on purpose (DLP ships a
    no-matched-text mode as an option; here it's the only mode).
  - **`score` is the MINIMUM softmax probability across the span's
    tokens**, passed straight through from `lfm2_encoder::Span::score`.
    Minimum rather than mean because a span is a conjunction of per-token
    decisions — it is wrong if any one token is wrong — so a single
    coin-flip token inside an otherwise confident credential is exactly
    the signal averaging would hide. **These numbers therefore read
    systematically lower than tools that average** (Hugging Face's
    grouped-entity pipeline) or that report a recognizer's own confidence
    (Presidio); do not compare them across tools, and do not threshold
    them against a fixed cutoff, same as every other number this API
    returns.
- `POST /v1/spans/credentials` — identical contract to `/v1/spans`, server-
  side filtered (via `Lfm2TokenClassifier::credentials`) to only the
  `credential.*` entity family. A separate endpoint, not a query flag on
  `/v1/spans` — AWS's precedent of shipping "contains PII" as its own API.
- `POST /v1/tokenize` — `{"model": str, "text": str, "context"?: str}` →
  `{"model", "tokenizer_hash", "ids", "tokens": [{"id","token","start","end"}],
  "context"?: {"ids","tokens","suffix_stable"}}`. `model` is any id
  `GET /v1/models` lists, OR the adjudicator's own id
  (`GET /v1/adjudicator`) — each has its own tokenizer. `start`/`end` are
  UTF-8 BYTE offsets into `text` (same convention as `/v1/spans`); `token`
  is the tokenizer's own byte-level BPE piece spelling, not a decoded
  string (a leading space reads as `"Ġ..."`). Unknown `model` is `404`.
  With `context`: the `context` block reports the tokens `text` occupies
  once `context` precedes it, and `suffix_stable` — whether `text`,
  tokenized alone, is exactly the tail of `context + text`'s tokenization.
  `false` means a BPE merge crossed the boundary, which is exactly the
  hazard that makes teacher-forcing arbitrary text at a slot unsafe (see
  `docs/lfm25-adjudicator.md` "The opinion API"). Answered on the request
  handler over a tokenizer CLONED out at startup — it shares no queue with
  either worker, so it answers even while the adjudicator or an encoder
  head is mid-job.
- `POST /v1/probe` — adjudicator only (absent, not present-but-403, unless
  `--adjudicator-model` is configured), on by default; `--no-probe` (or
  `LFM2D_PROBE=0`/`false`) removes the route. Raw inference over exact
  text on the daemon's own stack — full details, response shape, and its
  bit-identical/drift certification are in
  [the adjudicator guide](../docs/lfm25-adjudicator.md)'s "Probe and
  tokenize" section. **It is an instrument, not a judgement API**: no
  calibration contract, and `docs/integration.md` invariants 8–11 do not
  apply to it.

Errors: `{"error": {"message", "type"}}`. `type` is one of:

- `"bad_request"` (400) — malformed/empty input, a call against a head
  this instance never loaded, or `/v1/adjudicate` without a `spec`;
- `"not_found"` (404) — `/v1/tokenize`'s unknown `model`, an unknown
  `spec` on `/v1/opinion` or `/v1/adjudicate`, `DELETE
  /v1/opinion/specs/{id}` on nothing loaded, or a chat checkpoint
  (`/v1/chat`'s `from`, `/v1/opinion`'s `context`) that was evicted or
  never made. `/v1/probe` names no spec,
  so it never 404s that way; with `--no-probe` the route is absent, a
  plain unmatched-route 404;
- `"forbidden"` (403) — deleting a boot spec;
- `"unprocessable"` (422) — an uploaded spec that parses but cannot be
  served (the load-time refusal text is the message);
- `"cancelled"` (408) — the client went away before the answer, or the
  daemon is shutting down;
- `"overloaded"` (503) — the opinion engine's queue is full;
- `"deadline"` (504) — the request's `timeout_ms` ran out;
- `"internal"` (500) — a loaded model's forward pass failed, or a worker
  thread died.

A consumer switching on `type` must treat an unknown value as an error.
Never a silently-wrong `200`.

## Calling lfm2d from another program (non-normative)

Design notes for a client. The promises are in `docs/integration.md`;
these are the consequences of how the daemon is built.

**Assume a queue, not a pool.** One forward pass already saturates most of
a 16-core box, so client-side fan-out buys nothing — requests serialize
through one worker by design. Batching within a request is the lever;
concurrent connections are not. The `lfm2d.worker.queue_depth` gauge makes
overload visible before it is painful.

**Budget ~1.4 GiB of resident memory per head**, and do not expect heads to
share a trunk unless they were trained on a common frozen one: the base
encoder and the shipped Prompt-Router have 0 of 148 trunk tensors in
common, and a full finetune diverges from both.

**Never threshold a score you did not fit.** No endpoint ships a
`flagged` boolean or a cutoff. The retired shell severity head showed why:
its benign and severe score ranges overlapped by measurement. If your
policy needs a yes/no, that decision belongs in your code, fitted on your
data, where it is visible as your choice.

**Log the `weight_hash` beside any decision you record.** It is the audit
trail: it ties a verdict to the exact weights that produced it, and it is
how a rollback is told apart from a regression after the fact.

**Keep failures loud.** An unknown spec is a `404` and a bad one is
refused by name at load; a dead worker exits the process rather than serving a healthy-looking 200. Do not
paper over a 5xx with a permissive default — but do not block on the
daemon either. `docs/integration.md` invariant 6 is the rule: proceed with
your own baseline controls and record that you skipped.

### Standards this API follows, and where it stops

`/embed` is TEI-shaped, so a TEI-conformant client works against it
unchanged. The `/v1/*` endpoints have no standard to follow:
TEI has no token-classification endpoint at all, and KServe V2/OIP — the
only real standard here — is tensor-clunky and was deliberately not
adopted. The span shape follows the PII-service convention (Presidio,
Amazon Comprehend, GCP DLP) instead, which is the closest prior art for
this job.

## Being a good k8s/k3s container citizen

**Graceful shutdown.** On SIGTERM or SIGINT: `/readyz` flips to 503
immediately (so a Service's endpoint controller stops routing new traffic
here), every listener (Unix socket and/or TCP) stops accepting new
connections via axum's `with_graceful_shutdown`, in-flight requests are
allowed to finish (a forward pass here is sub-second), then the process
exits 0. The drain is bounded — `shutdown::DEFAULT_DRAIN_TIMEOUT` (10s) —
so a wedged request can't hang shutdown forever; `serve()` returns `Ok(())`
either way, exit code 0 regardless. Rust runs as PID 1 in the container, so
this handler is installed explicitly (`tokio::signal`) — there's no init
process to translate the signal for us. See `src/shutdown.rs` and
`src/main.rs`'s `install_signal_handlers`; `tests/graceful_shutdown.rs`
drives the sequence programmatically (via `server::begin_shutdown`) and
asserts all three parts: `/readyz` flips, the in-flight request completes
200, `serve()` resolves.

**Exit waits for the engine drop.** After `serve()` returns, `main.rs`'s
`exit_after_serve` waits up to `shutdown::WORKER_EXIT_TIMEOUT` (5s) for the
worker thread to drop its engine (`worker::WorkerExit`), and only then exits
0. `serve()` returning drops the last `WorkerHandle`, which ends the worker
loop, and the engine drops on the worker thread. A GPU engine's drop frees
device memory through the driver, so exiting underneath it runs
libamdhip64's atexit teardown mid-free; a ROCm build segfaulted exactly
there (see Known problems). If a request outlived the drain cap, its
connection task still holds a `WorkerHandle`, so the wait times out, a
warning is logged, and the exit is still 0. That keeps 10s + 5s inside
Kubernetes' default 30s grace period. It also means the fix NARROWS the
race to that already-pathological path rather than eliminating it: exit
then runs under a live engine, and a GPU build can still fault there.
`tests/shutdown_exit.rs` proves the ordering through the real binary with
no GPU: a stub engine whose drop sleeps and then writes a marker must have
written it before the process exits. `demo/e2e.py` (run by
`demo/test_devices.sh`) fails the device gate on any nonzero SIGTERM exit.

**Crash-only on worker death.** A worker-thread panic today would leave the
HTTP server answering 500s forever while `/healthz` still said 200 —
alive-but-useless, and kubelet never restarts a pod whose process hasn't
exited. `main.rs`'s production path spawns the worker via
`WorkerHandle::spawn_crash_on_panic` instead of the plain `spawn` the test
suite uses: a monitor thread calls `std::process::exit(1)` if (and only if)
the worker thread terminates via panic — NOT on the clean channel-closure
exit a graceful shutdown produces (see `worker_thread_outcome_is_a_crash`
in `src/worker.rs`, unit-tested directly). `tests/worker_crash_monitor.rs`
proves this through the REAL compiled binary: it spawns `lfm2d` itself
with `LFM2D_TEST_CRASH_ON_WORKER_PANIC=1` (a test-only env-gated branch in
`main.rs` that swaps in a panic-on-call stub engine instead of loading real
checkpoints — real models aren't needed to prove the process-exit
mechanism), fires one request over a real HTTP connection, and asserts the
CHILD PROCESS itself exits nonzero.

**Startup observability + `--threads`.** Startup logs (via `tracing`, not
`eprintln!`) the full effective config, each loaded model's `{id, kind,
weight_hash}`, `std::thread::available_parallelism()`, and a hand-rolled
container-runtime probe (`/.dockerenv` → docker, `/run/.containerenv` →
podman, `KUBERNETES_SERVICE_HOST` → kubernetes, `container` env var,
`/sys/fs/cgroup/cpu.max` when readable) — see `src/probe.rs`, unit-tested
with injectable root dir + env snapshot, no new dependency. `--threads N`
(env `LFM2D_THREADS`) sizes rayon's GLOBAL thread pool — which candle's
matmul runs on transitively — via
`rayon::ThreadPoolBuilder::num_threads(n).build_global()`, called BEFORE
any model load (rayon's global pool can only be built once, so this has to
win the race against candle's own lazy default); a build failure exits
loudly rather than silently falling back to the default pool size.

**OpenTelemetry: metrics + traces + logs over ONE pipeline** (`tracing` +
OTLP, not bare Prometheus). Human-readable stderr logging is always on
(`tracing-subscriber`'s `fmt` layer); when `OTEL_EXPORTER_OTLP_ENDPOINT` is
set, OTLP export layers/providers are added on top and ONE loud line is
logged either way ("OTLP export disabled (... unset)" or "OTLP export
enabled"). A missing/unreachable collector never crashes the daemon or
slows inference — every exporter is a BATCH exporter (bounded queue, drops
under backpressure) over gRPC/tonic with rustls (never openssl). One span
per HTTP request (`method`, `route`, `status`); within it, a `worker_call`
span carrying `queue_wait_ms` and `inference_ms` — the queue-wait vs.
compute split is the whole point of a serial worker (see `src/worker.rs`'s
module docs on how a `tracing::Span` is created request-side and recorded
onto from the worker OS thread).

Incoming `traceparent` and `tracestate` are extracted using W3C Trace Context.
The HTTP server span adopts the remote parent, and the worker span carries
that trace context across its OS-thread boundary. Remote sampling is honored;
absent/invalid parent headers start a local trace, duplicate `traceparent`
headers are refused as parents, and split `tracestate` headers retain order.
This extraction does not depend on setting a global propagator. Upstream
clients still need to inject their headers; lfm2d has no downstream inference
HTTP calls to inject into.

HTTP spans also carry `http.request.method`, `http.route`, and integer
`http.response.status_code`, with server span kind and error status on 5xx.
Unknown paths use `<unmatched>` so arbitrary path text never enters trace or
metric labels. Export-capture tests in `telemetry.rs` verify actual parent IDs,
tracestate, sampling, worker timing, HTTP errors, and execution resources.

**`/v1/spans`/`/v1/spans/credentials` telemetry is held to a stricter bar.**
This endpoint exists to find live credentials, so request text contains
them by definition. No span, log event, or metric anywhere in this crate
ever records input text, span offsets, or a matched substring — the
`worker_call` span for these two operations carries only `operation`,
`queue_wait_ms`, `inference_ms`, and (opt-in) `input_hash`, same as every
other operation's span, nothing request-shaped added. `--log-input-hash`
(default OFF) attaches a sha256 hash of the request's input text — never
the text itself — to that call's span as a trace/log attribute ONLY; it is
never threaded onto an OTLP METRIC label (unbounded per-input cardinality
would wreck VictoriaMetrics). `tests/spans_telemetry_safety.rs` is the
load-bearing proof: it embeds a known secret string in a request, captures
every field tracing emits via a custom `Layer` (no OTLP collector
required), and asserts the secret — and the opt-in hash's raw bytes —
never appear; a mutation test during development (temporarily logging the
raw input) confirmed this test actually fails when it should.

Resource attributes: `service.name`
(`OTEL_SERVICE_NAME`, default `lfm2d`), `service.version` (crate version),
`lfm2d.execution.device_type` (`cpu`/`gpu`), `lfm2d.execution.backend`
(`cpu`/`rocm`/`cuda`/`sycl`/`metal`), `lfm2d.execution.dtype`, and
`lfm2d.model.<kind>_hash` per loaded model — set once, from
`main.rs`, AFTER models finish loading (weight hashes aren't known any
earlier; see `src/telemetry.rs`'s module docs). The same resource is attached
to traces, metrics, and logs. Device metadata comes from the loaded engine,
not host hardware inventory; `lfm2d.execution.device_name` carries the
selected device's identity where the backend can name its target (ROCm:
`rocm:gfx1151:hip7.2`; CUDA: `cuda:sm_121:nvcc13.0:drv580.173.02`; SYCL:
`sycl:2026.1:Intel(R) Arc(TM) Pro B70 Graphics`) and is omitted elsewhere, and `lfm2d.candle_rev` names
the candle build. Metrics:
`lfm2d.worker.queue_depth` (observable gauge over an `AtomicUsize`,
incremented on send, decremented when the worker picks a command up),
`lfm2d.request.duration` (histogram, by route+status),
`lfm2d.inference.duration` (histogram, by operation kind), `lfm2d.requests`
(counter). The two histograms carried a
`_duration_ms` name until the OTel exporter's unit suffix made it
`..._ms_milliseconds`; `src/telemetry.rs` is the source of truth for the
name, and a running image may still be exporting the old one until it is
rebuilt. Verified against a real OTLP/gRPC capture server with a
sequence classifier (since removed) and the Prompt-Router loaded: spans arrived correctly nested
(`http_request` parenting `worker_call`) carrying `queue_wait_ms`/
`inference_ms`, all four metrics arrived, logs arrived with the effective
config and per-model weight hashes, and SIGTERM still drained and exited 0
cleanly despite transient export failures observed against one candidate
endpoint (see "Problems noted, not fixed" below — worth knowing, not a
code defect).

**Deploying on k8s** — `deploy/k8s.yaml` is a complete Deployment +
Service: `readinessProbe`/`livenessProbe` against `/readyz`/`/healthz`, a
`startupProbe` budgeted for a COLD PVC (local-disk load measured at
~1.4-3.2s for two heads here; a cold network-backed PVC can take far
longer, which is what the startup probe's generous budget is for — the
readiness/liveness probes only start once it succeeds),
`terminationGracePeriodSeconds` comfortably above the 10s drain cap,
`resources.requests` with a MEASURED memory number (not an estimate — see
the manifest's comment for the exact `/proc/<pid>/status` reading), CPU
`requests` WITHOUT `limits` (rationale in-manifest: rayon's thread pool
auto-sizes to the visible core count at startup and doesn't re-check a
limit applied later — a CPU limit just throttles the same thread count
against fewer cores, no benefit; this workload scales by replica count,
not by starving one replica), and OTLP env wiring. lfm2d is STATELESS — N
replicas behind the Service is the horizontal scaling path; a previous
version of this doc's k8s example pinned `replicas: 1` in a way that read
as a requirement, which was wrong and has been fixed (see `deploy/lfm2d.container`'s
trailing comment and `deploy/k8s.yaml` itself).

## Judgment calls worth knowing about

The original design brief left a few things implicit; here's what was
decided and why (also in the relevant doc comments):

1. **Audit headers vs. audit body fields.** Every inference response
   carries `{model_id, weight_hash}`, but `/embed` and `/v1/spans` are bare
   TEI-shaped arrays with no room for those fields. Resolved by putting
   them in `X-Model-Id`/`X-Model-Weight-Hash` response headers there, and
   directly in the JSON body for `/v1/route` (which is "our full
   contract," not TEI-compat).
2. **One model per head kind — except token classifiers.** No `/embed` or
   `/v1/route` request carries a model-selection parameter, so this daemon
   serves at most one embedder and one router at a time. `/v1/spans`/`/v1/spans/credentials` are the deliberate
   exception: `--token-classifier-dir` is repeatable and the request takes
   an optional `"model"` field, because "N secrets/PII detectors behind one
   sidecar" (a general PII head plus a narrower secrets-only head, say) is
   a realistic deployment shape in a way that "N embedders" or "N routers"
   is not for this daemon's current consumers.
3. **`/v1/spans`'s `score` is the weakest token, not the average.** A
   span is a conjunction of per-token decisions, so its trustworthiness is
   its weakest token's; averaging would hide the coin-flip token that
   indicates a misplaced boundary. The consequence for a caller is that
   these numbers are not comparable with other PII services' — see the
   API section's `/v1/spans` entry and `types::SpanResult::score`'s doc
   comment.

4. **No observe-only endpoint.** Asked for a way to report actions a
   consumer handled WITHOUT scoring them — read-only calls that bypass the
   daemon — so coverage is countable. Deliberately not added: an endpoint
   that records without scoring turns a model server into an event sink,
   a different service with different retention and scaling properties,
   and the right sink already exists. This daemon exports traces, metrics
   AND logs over OTLP (see Observability), so a consumer should emit its
   bypassed actions as spans carrying the attributes it would have sent,
   marked `lfm2d.skipped=read_only`. Scored and unscored actions then live
   in one store, coverage accounting is a query rather than a second write
   path, and lfm2d keeps one job.

## Problems noted, not fixed

- **`snapshot_id` names the GPU target on ROCm and CUDA, and only part of
  it on SYCL.** It hashes the device identity and the candle revision
  (`CANDLE_REV`, read from `Cargo.lock` by `lfm2d/build.rs`). ROCm's
  (`rocm:gfx1151:hip7.2`, since 2026-09-24) comes from the fork's
  `RocmDevice::arch()`/`hip_version()`. CUDA's
  (`cuda:sm_121:nvcc13.0:drv580.173.02`, since 2026-09-27) is the compute
  capability, the nvcc release in the header of candle's own PTX, and the
  driver release from `/proc/driver/nvidia/version`: the driver
  JIT-compiles that PTX, so a driver update can move the numbers. SYCL's
  (`sycl:<icpx>:<device name>`, since 2026-10-02) reads the compiler release
  from `CANDLE_SYCL_ICPX_VERSION` at run time, defaulting to `2026.1`, not
  from the build that made `libcandle_sycl.so`; and the kernels build to
  SPIR-V that the GPU driver JIT-compiles, so, as on CUDA, the driver
  release belongs in the identity and is not there yet. Metal still
  reports the bare backend name.
- **SYCL: the 8B engine's router bias and conv weights alias freed
  memory** (found in review 2026-10-03, code-read; not yet reproduced on a
  device). The fork's `QSyclStorage` hands `dequantize` callers a
  non-owning view of a per-storage cache, and `quantized_lfm2_moe.rs`
  dequantizes `exp_probs_b.bias` and `shortconv.conv.weight` from
  temporary QTensors, so the cache entry returns to the pool while the
  model still reads it. Encoder parity cannot see it (no QTensor there).
  Treat 8B numbers measured on SYCL before the fix as suspect.
- **SYCL: a cold `/v1/opinion` describe runs until the client deadline.**
  Seen on the B70 (2026-10-02, [docs/lfm2d-b70-endpoint-exercise.md](../docs/lfm2d-b70-endpoint-exercise.md)):
  with the described-state cache cold, the read holds the worker at one
  core until `timeout_ms`, then returns `cancelled`; warm reads take 79-97
  ms. It answers cancellation, so the worker is walking, not deadlocked:
  `describe_then_read` decodes up to `context_limit` with no budget of its
  own and logs nothing until it finishes, and the two cold describes that
  did finish took 17.8 and 25.7 s. Not seen on ROCm or CUDA. The aliased
  router bias above is a candidate contributing factor; re-run after it.
- **SYCL: a stale kernels library survives a candle bump.**
  `lfm2d/build.rs` rpaths the newest-named
  `target/<profile>/build/candle-sycl-kernels-*/out` that holds
  `libcandle_sycl.so`; with several hash dirs present, name order is not
  build order. Clear those dirs after a bump until the kernels crate hands
  its path down directly.
- **Spec registration telemetry has no source address.** The daemon has
  no `ConnectInfo` wiring, so a registration or eviction log line cannot
  say who uploaded the spec.
- **The cancellation check after a spec registration is covered only by a
  test double** (`handle_menu_tests`), not by a real `Adjudicator` load
  racing a cancel.
- Each head loads its own trunk; there is no trunk sharing across heads.
  This is not a gap to close in the general case — the base encoder and
  the shipped Prompt-Router have 0 of 148 trunk tensors in common, and
  only heads trained on a *common frozen trunk* can share one, which costs
  measured accuracy. Budget memory per head.
- The quadlet's `HealthCmd` only proves the binary still execs, not that
  `/healthz` answers — the runtime image ships no `curl`/`wget` on purpose
  (minimal image; see the Containerfile). A real HTTP healthcheck should
  run from OUTSIDE the container against `/healthz`/`/readyz`.
- **An endpoint that accepts TCP is not an endpoint that forwards gRPC.**
  During live OTLP verification a port that connected instantly timed out
  every actual export, even at a 20 s `OTEL_EXPORTER_OTLP_TIMEOUT`, while
  the standard OTLP/gRPC port on the same box captured everything from the
  identical exporter code. The daemon behaved correctly either way — the
  failed exports neither crashed it nor blocked SIGTERM's clean exit 0 —
  but pointing `OTEL_EXPORTER_OTLP_ENDPOINT` at a genuinely reachable
  collector is a precondition this daemon cannot check for you.
- `WorkerHandle::spawn_crash_on_panic`'s monitor mechanism is proven
  end-to-end via `tests/worker_crash_monitor.rs`, but that test needs a
  `LFM2D_TEST_CRASH_ON_WORKER_PANIC`-gated stub-engine branch in `main.rs`
  to avoid depending on real checkpoints being present in CI. The pure
  decision logic (`worker_thread_outcome_is_a_crash`) is ALSO unit-tested
  directly, so the crash-vs-clean-exit distinction has a fast, model-free
  test in addition to the slower real-binary one.
- **`lfm2d.model.token_classifier_hash` collides across 2+ token
  classifiers.** `telemetry::resource()` sets one OTLP resource attribute
  per LOADED KIND (`lfm2d.model.<kind>_hash`), a scheme that predates
  `--token-classifier-dir` being repeatable — with 2+ token classifiers
  loaded, only the last one's hash survives in that attribute (the others
  are silently overwritten by `Resource::builder().with_attribute`'s
  last-write-wins semantics). Every loaded model's real hash is still
  correctly reported per-model in `GET /v1/models` and in every
  `/v1/spans` response's `X-Model-Weight-Hash` header — this only affects
  the OTLP resource-attribute shortcut, not the audit trail itself. Worth
  fixing (e.g. `lfm2d.model.token_classifier.<id>_hash` per head) before
  anyone relies on that specific attribute with 2+ token classifiers
  loaded; not fixed here because it needed a resource-attribute shape
  decision, not a spans-endpoint one.
- **FIXED 2026-09-13: the final OTLP flush never ran.**
  `telemetry::TelemetryGuard` flushes every batch exporter when dropped, but
  `main` has always left through `std::process::exit`, which never unwinds
  its frame. So the guard never dropped, and whatever the exporters had
  buffered was lost on every shutdown. A kaibo review found it.
  `exit_after_serve` now drops the guard explicitly, after the worker wait,
  bounded by `shutdown::TELEMETRY_FLUSH_TIMEOUT` (5s) so a dead collector
  cannot eat the grace period. `tests/shutdown_exit.rs` asserts the flush
  runs, and runs after the engine drop.
- **FIXED 2026-09-13: a ROCm build crashed after a clean SIGTERM
  shutdown.** On 2026-09-12 a ROCm daemon logged `shutdown complete,
  exiting 0` and the harness reported exit 1. On 2026-09-13 the same shape
  reproduced as SIGSEGV in 1 of 11 paired CPU+ROCm shutdowns. The core dump
  put the fault on the `lfm2d-worker` thread: the worker loop ended, dropped
  `RealEngine`, and its last tensor freed the ROCm allocator
  (`RocmAllocator::release_all` -> `hipFree`) while `main`'s `exit(0)` ran
  libamdhip64's atexit teardown. Two contributing factors, two fixes:
  - lfm2d exited without waiting for the engine drop. `main` now waits for
    it (see "Exit waits for the engine drop").
  - The candle fork's exit guard read its flag once, before the free loop.
    tobert/candle `d9748a8f` makes the atexit hook wait for releases
    already under way; its `rocm_exit_race` test went from 24 of 24
    SIGSEGV to 24 of 24 exit 0.
  After the lfm2d fix, 80 of 80 paired shutdowns exited 0 (at the old rate
  a clean 80 is ~0.05% luck). `demo/test_devices.sh` now asserts the exit
  status.

## Resident causal adjudicator

LFM2.5-8B-A1B GGUF inference, reusable hot-prefix state, and validated JSON
reports are available as a development integration. See
[the adjudicator guide](../docs/lfm25-adjudicator.md) for build/run commands,
HTTP semantics, measured limitations, and the published Candle dependency.

Beside `/v1/adjudicate` the same worker serves `/v1/opinion`, the System 1
read: state in (`{"input", "facts"?}`), a typed distribution over a spec's
choice field out, no text generated past the question's slot and no winner
named. Each spec names its own input with a required `input_label`, and
the user turn renders `{facts}{input_label}:\n{input}`; the menu
(`GET /v1/opinion/specs`) carries the label so a client reads it at
runtime. The guide's
"The opinion API" section has the shapes and the numbers;
`docs/integration.md` invariants 8–14 are its contract, and
`docs/writing-a-spec.md` is the guide to writing one. It deploys as its
own GPU service, `lfm2d-system1` (`Containerfile.rocm`,
`deploy/k8s-system1.yaml`), beside this encoder sidecar rather than
inside it.

A spec can also be registered at runtime — `POST /v1/opinion/specs` (body:
the spec's bytes) and `DELETE /v1/opinion/specs/{id}` — beside the ones
loaded at boot via `--opinion-spec` (repeatable; zero is valid, and the
menu then starts empty). An id is the content hash of the exact bytes, so
re-uploading the same spec is free; an unknown or evicted `spec` is `404`,
the cue to upload and retry. There is no default spec: `/v1/opinion` and
`/v1/adjudicate` both require `spec`, and `/v1/adjudicate` without one is
a `400` naming the menu. `GET /v1/adjudicator` reports the checkpoint
and where it runs (`model_id`, `weight_hash`, `tokenizer_hash`,
`context_limit`, `backend`, `device`, `candle_rev`, `dtype`, `sampling`,
`weight_dtypes`); per-spec identity is each menu entry's `snapshot_id`,
which hashes `device` and `candle_rev` too. See the guide's "Runtime spec
registration" and `docs/integration.md` invariants 12 and 14.

Two more routes are answered without ever entering a spec's judgement
contract: `POST /v1/tokenize` (any loaded model, encoder head or
adjudicator, over a cloned tokenizer — never queues behind either worker;
every clone has truncation/padding cleared, so a >512-token request
through an embedder-style tokenizer never silently reports a truncated
count) and `POST /v1/probe` (raw inference over exact text — or exact
token `ids` — on the adjudicator's own stack: top-k logprobs,
teacher-forced continuations, greedy `generate`, and an explicit warm/cold
resume report that always picks the LONGEST matching resident prefix,
never just the first one checked. An optional `decode_from` byte offset
(or, with `ids`, `decode_from_token`) replays part of the schedule one
token at a time — the daemon's own decode-loop forward call, reused — so
a caller can reproduce `/v1/opinion`'s exact schedule, not just the
bulk-prefill one; prefer the `ids` form when replaying a generation, since
`text`/`decode_from` re-tokenizes fresh and BPE is not injective in that
direction). Both are instruments for inspecting the daemon's own numbers,
not decision APIs; see the guide's "Probe and tokenize" section.
