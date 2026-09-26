# LFM2.5 resident adjudicator

The daemon can load `LFM2.5-8B-A1B` directly through our Candle fork, prefill
each prompt spec's system prompt once, and evaluate independent inputs from
that immutable hybrid-state snapshot. `POST /v1/opinion` reads a spec's
choice field as a distribution; `POST /v1/adjudicate` returns a validated
JSON report and the original model output. It executes nothing and decides
nothing: the consumer owns every decision.

> **Specs moved out on 2026-09-24.** lfm2d ships no production specs: each
> consumer owns and uploads its own, every spec names its input with
> `input_label`, and no spec is a default. The measurements below were
> taken on the shell spec `command-verdict-enum-v1` (now kaijutsu's; in
> this repo's git at `f9ca081`) and are kept as the dated record of what the
> engine did, with that spec named beside each number.

The [first optimization pass](lfm25-optimizations.md) adds device-side
selection, merged expert projections, and fused convolution updates while
preserving the initial outputs. The [second pass](lfm25-grouped-prefill.md)
adds grouped expert prefill. That pass changes floating-point accumulation
order and can change generated text; see its separate hardware checks and
prompt/model observations.

The [third pass](lfm25-qkv-input-cache.md) packs compatible QKV projections
and adds an exact complete-input checkpoint with saved next-token logits.

The [fourth pass](lfm25-kv-cache.md) adds append-efficient KV buffers with
immutable snapshot prefixes and transactional failure behavior.

The [fifth pass](lfm25-gqa-routing.md) removes repeated KV-head materialization
and shares prepared expert-routing maps across compatible projections.

The [sixth pass](lfm25-moe-fusion.md) fuses expert selection and weighted output
reduction, removing 154 kernel launches per measured decode step.

The [seventh pass](lfm25-expert-buffers.md) removes redundant expert-buffer
initialization, eliminating another 88 fill dispatches per measured decode step.

## Main integration validation — September 15

The published dependency pin passed 269 workspace CPU tests, with one existing
ignored GPU test, including the real encoder-model regression fixtures. The ROCm
release build passed the 64-token cache/deadline/reuse/SIGTERM smoke; all twelve
outputs match the saved full-run reference prefixes. This short smoke checks
integration behavior, not report quality or new performance gains.

Workspace Clippy completes with existing style/type warnings; `-D warnings`
remains blocked by the recorded lint backlog. No runtime code changed when
replacing the sibling Cargo overrides with the published dependency.

## Build and run

Clone the `main` branch of
[tobert/lfm2d](https://github.com/tobert/lfm2d).
Cargo pins a revision of the published Candle fork on
[lfm25-trace](https://github.com/tobert/candle/tree/lfm25-trace) (the rev is
in `Cargo.toml`; `GET /v1/adjudicator` reports it as `candle_rev`): the seven
optimization passes plus the read-only observer and per-call steering seams.
No sibling checkout or local Cargo patches are required. The ROCm build
requires the ROCm development toolchain and a supported AMD GPU. Fetch the
GGUF and the tokenizer as [development.md](development.md) shows:

```bash
cargo build -p lfm2d --release --features rocm
./target/release/lfm2d \
  --device rocm --threads 8 --bind-addr '127.0.0.1:18152' \
  --adjudicator-model '.models/LFM2.5-8B-A1B/LFM2.5-8B-A1B-Q5_K_M.gguf' \
  --adjudicator-tokenizer '.models/LFM2.5-8B-A1B/tokenizer.json' \
  --opinion-spec 'demo/specs/email-triage-v2.json'
```

`--adjudicator-model` and `--adjudicator-tokenizer` together enable the
engine. `--opinion-spec` (repeatable, or `LFM2D_OPINION_SPECS`
comma-separated) puts a spec on the boot menu with its own resident prefix,
named by file stem; zero is valid, and the menu then holds only uploads.
There is no default spec: `/v1/opinion` and `/v1/adjudicate` both name one.
`--opinion-spec-capacity` (`LFM2D_OPINION_SPEC_CAPACITY`, default 8) bounds
how many runtime-uploaded specs (`POST /v1/opinion/specs`, below) stay
resident at once — boot-time specs above don't count against it and are
never evicted.

### Deploying it: `lfm2d-system1`

The adjudicator deploys as its own service beside the encoder pod, never
inside it: `lfm2d/Containerfile.rocm` builds the `rocm` feature on
`rocm/dev-ubuntu-24.04:<host ROCm version>` (the binary links the ROCm
runtime dynamically, and candle compiles its kernels with `hipcc` at first
run for the GPU it finds, cached under `CANDLE_ROCM_CACHE_DIR`), and
`lfm2d/deploy/k8s-system1.yaml` runs it with one GPU from the AMD
device plugin, the GGUF, tokenizer and embedder from a mounted volume, and
only the demo prop specs baked into the image (consumers upload theirs).
The manifest's comments
carry the measured memory numbers and every deliberate difference from
the encoder pod.

Download the matching `tokenizer.json` from
[LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/tree/main).
The loader checks its vocabulary against the GGUF, checks the control-token
IDs, and refuses an unrecognized embedded chat template. It hashes the full
weights and tokenizer at startup. The supported template is the checkpoint's
single-system/single-user text format. Literal model control delimiters in
input are rejected. Tool definitions remain available as a comparison mode;
the initial report contract uses `output_schema`.

### The reasoning region

A prompt spec's `reasoning` field says whether the assistant's turn opens with
a *completed* reasoning region. It defaults to `closed`, which prefills
`<think>\n\n</think>\n`.

The checkpoint's chat template never emits a reasoning region — `<think>` is a
token the model writes, and after `<|im_start|>assistant\n` it writes it at
p=1.00. Under an `output_schema` the grammar's first legal byte is the object's,
so the v1 template masked the model off its own manifold at step 0: the forced
`{"` scored logprob -17.8 to -21.8 and every later token was conditioned on a
prefix the model considers impossible.

The prefill's exact bytes were measured on our ROCm stack, five candidates over
three inputs, reading `{"`'s standing at that slot:

| after `<|im_start|>assistant\n` | `{"` |
|---|---|
| nothing (the v1 template) | logprob -17.8 to -21.8, rank 6 or worse |
| `<think></think>` | rank 5 |
| `<think></think>\n` | rank 2-3 |
| `<think>\n</think>\n` | rank 2 |
| `<think>\n\n</think>\n` | **rank 1, p 0.42-0.71** |

The template's own dialect for a completed region — `"<think>" + thinking +
"</think>"`, no surrounding newlines — is the one in the table that does worst
of the four, so the bytes are measured rather than derived. A second round
varied one newline at a time and every neighbour was worse.

`reasoning: "open"` leaves the turn as the template opens it and the model
reasons. Combined with an `output_schema` it is refused at load: the grammar
would have to admit a free-text region it cannot bound and then start the
object after `</think>`, which is not built. The one measurement we have says
this model's reasoning argues severity *down*, so it is a decision rather than
a default.

`template_version` carries the mode (`lfm25-single-user-v2-closed`), and each
spec's `snapshot_id` is computed from it, so a consumer sees the template
change rather than inferring it. A spec with `tools` reads `v3` (2026-09-26):
its tools are listed as the template's own `tojson` writes them, through
`lfm2d::chat`, where `v2` wrote them compact and key-sorted. Specs without
tools render the same bytes and keep `v2`.

```bash
curl --fail-with-body 'http://127.0.0.1:18152/v1/adjudicate' \
  -H 'Content-Type: application/json' \
  --data '{"spec":"email-triage-v2","input":"Email:\nwhere is my order 4471?"}'
```

## Response contract

`GET /v1/adjudicator` identifies the checkpoint and where it runs
(`model_id`, `weight_hash`, `tokenizer_hash`, `context_limit`, `backend`,
`device` such as `rocm:gfx1151:hip7.2`, `candle_rev`, `dtype`, `sampling`,
`weight_dtypes`); per-spec identity (`snapshot_id`,
`template_version`, `prefix_tokens`) is on each `GET /v1/opinion/specs`
entry. `/v1/models` also lists the adjudicator. Each generation response
includes:

- Model, weights, tokenizer, template, snapshot, backend, compute dtype,
  weight formats, and decoding-policy identifiers.
- `report`: the validated report object, or `null`.
- `report_error`: an explicit validation/truncation error, or `null`.
- `output`: the generated text, unmodified. Under `reasoning: "closed"`
  that is the report and nothing else — the reasoning region is in the prompt,
  and a reasoning delimiter reaching this field would mean something went
  wrong, which `validate_report` reports rather than strips.
- `resumed_tokens`: present only when the generation resumed from a
  described state `/v1/opinion` left for these exact prompt bytes (below,
  "Escalation"): how many of `completion_tokens` came from it. The report
  is the one a fresh generation writes; only the time differs.
- `finish_reason`: `stop` or `length` (`opinion` on an opinion read, below);
  token counts and queue/prefill/decode durations. Prefill timing
  synchronizes the device before stopping the clock.

HTTP 200 means generation completed, **not that a valid report exists**.
Consumers must require `report != null` and `report_error == null`. Validation
checks structure, not truth: free-text explanations still require evaluation.
No verdict is substituted for an invalid report. There is no retry or repair
loop, and none is needed for *format*: with an `output_schema`, generation is
constrained by a JSON grammar compiled from that schema (`lfm2d/src/constrain.rs`),
so a malformed or off-schema report is unreachable rather than rejected after
the fact. `report_error` is then only ever truncation — the grammar will not
invent a field value to close the object, so a completion that exhausts
`max_tokens` mid-document still returns `finish_reason: "length"` and no report.

The supported schema is deliberately small: one closed object with all fields
required, string/boolean field types, and optional enums. Strings must be
nonempty. Unsupported schema constraints, duplicate fields, unknown fields,
coercions, Markdown fences, trailing prose, and truncated output are rejected.
Under `output_schema` the
whole completion is the document — a reasoning region before the object is
rejected, not stripped, because the region the model expects is supplied
already closed by the prompt (see above) and `<think>` is masked
unconditionally, so one appearing in a completion means something went wrong.
Keys come out in the schema's `required`
order — and the schema **as stated in the system prompt** lists them in that
same order, because `serde_json`'s sorted rendering used to state one order and
then mask the model into another — separators are spaced the way the model
writes them (`": "` and `", "`; compact ones corrupted 15 of 16 reports), and
`\uXXXX` escapes are not admitted (every character they can spell is reachable
literally as UTF-8). A schema
`validate_schema` accepts but the grammar cannot honour — today, an `enum` with
a blank string value, which no valid report could contain — is a loud error,
never a silent fall-through to free generation. The grammar is compiled against
the tokenizer once, at load, so such a schema stops the daemon from starting
and never reaches a request; its mask plans are shared by every report after
the first that needs them.

Requests accept `input`, `max_tokens` (default/max 2048), `timeout_ms`
(default 30000, max 120000), `use_cache` (default true), `distributions` and
`opinion` (both below). `use_cache:false` explicitly measures a cold prefill. Prompt plus output must fit the server's
context budget (default 4096, range 128–32768; see "Context budget"). Inputs are at most 66560 bytes: `/v1/opinion`'s 64 KiB state plus 1 KiB of headroom for what the opinion render adds around it (the `input_label` itself is capped at 64 bytes), so an escalation always fits.
Bad requests return 400; overload 503; deadlines 504; cancellation 408;
inference failures 500. Deadline time includes queueing.

Decoding is deterministic greedy with a sign-aware repetition penalty of
1.05, applied once per token occurring anywhere in the full prompt and this
evaluation's continuation. `--adjudicator-repeat-penalty 1.0` disables it.
The [model card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) recommends
1.05 along with stochastic sampling; our deterministic policy is recorded
explicitly in each response. Under `reasoning: "closed"` the model does not reason before answering, so
the report is the whole completion and a few hundred output tokens is ample;
under `"open"` it reasons first and 512 or 1024 often truncate the report.

### Distributions

A request may add `distributions`; one that does not gets the response above,
byte for byte. Unknown keys, an empty or repeating token set, an id outside the
vocabulary and a `top_k` over 20 are 400s, never clamped.

```json
{"input": "...", "distributions": {"top_k": 5,
  "token_sets": {"verdict": [101, 202, 303]}, "constrained": true}}
```

The response gains `distributions`: one entry per generated token, in order.
Each carries the selected `token`, its `text` and `logprob`, the `top_logprobs`,
and `set_mass` — per named set, the `logprob` and `prob` of the mass the model
put on those ids.

**Every number is raw**: the model's own log-softmax over the full vocabulary,
read before the output grammar's mask and before the repetition penalty. A set's
mass is never renormalised over the set. Low mass means the model was never
steered toward that vocabulary — *unasked*, not *wrong* — and it is the only
thing that tells those apart (`docs/field-requests.md` (git f9ca081), decision 5). Read
logprobs, not `prob`: values saturate, and the ordering survives only in the log.

`constrained: true` adds a `constrained` object to each step, describing what
the grammar left the sampler to choose from. It needs an output schema; asking
without one is a 400. `legal_tokens` is how many rows the grammar admitted,
`legal_mass` the raw mass the model put on them, `top_logprobs` the best legal
rows by raw logprob, and `forced` is true when the model's raw first choice was
not legal. A low `legal_mass` is the grammar overruling the model — worth
counting, since a forced token is conditioned on by everything after it. Leave
the closing end-of-text step out of such a count: once the document is complete
the legal set is that one token, so `forced` there only says the model would
have kept writing. There
is deliberately no renormalised number on the wire: the conditional logprob of a
legal token is `logprob - legal_mass.logprob`, computed by whoever wants it, with
the raw mass necessarily in hand. Token ids come from the checkpoint's
tokenizer; nothing here names a label.

### Opinion reads

A prompt spec may carry an `opinion` block — `prefill` (the assistant text
the options continue, e.g. `{"verdict": "`), `options` (the answer set, at
least two, distinct, non-empty) and `close` (the bytes written after each
option, e.g. `",` or `"}`; it makes an option that prefixes another score as
a whole word and is part of every score). A request with `opinion: true`
then does one prefill and teacher-forces every option's canonical tokens
plus the close, decoding nothing. The tokenization is split where the
options' full encodings first diverge, so a BPE merge across the prefill
boundary cannot put an option on an off-canonical path; that split depends
on the tokenizer and the spec, never on the input, and a close that cannot
separate the options stops the daemon at load. `opinion` with
`distributions` is a 400; `opinion: true` against a spec without the block is
a 400.

The response then carries `opinion` and differs from a generation:
`output` is empty, `finish_reason` is `opinion`, `completion_tokens` is 0,
`prompt_tokens` counts the prefilled shared prefix and `decode_ms` is the
option scoring. Inside `opinion`: `options` in spec order, each with its
`tokens`, raw sequence `logprob` (full-vocabulary log-softmax summed over the
continuation, close included) and `prob` renormalised over the options;
`sequence_mass`, the log probability that the model writes exactly one of
the options and its close; `first_token_mass`, the raw log mass on the
options' distinct first tokens; `shared_tokens` and `scored_tokens`; and
`rendered_sha256` of the text the options continue. **No winner is picked.**
Read `prob` beside `sequence_mass`: a low mass means the model was never
asked this question here, and a renormalised number over it is noise that
looks like an answer (decision 5, again). The option names are the spec's,
echoed; nothing in the daemon knows a verdict word. The block is part of
`snapshot_id`, so changing it changes the identity a consumer pins.

The continuation is scored as one block where generation decodes it a token
at a time. On ROCm the two schedules can take different kernels
(`lfm25-chunk-kernels.md`), so an opinion's numbers agree with a generation's
verdict slot in token identity, not to the last nat; the CPU equivalence test
(`opinion.rs`) certifies the alignment, not the serving backend. Measure on
the backend you serve.

Two shell specs (kaijutsu's now; in git at `f9ca081`) read the verdict
enum: `command-verdict-opinion-v1.json` asked only the verdict, so
`{"verdict": "` was its canonical first field;
`command-verdict-enum-v1-opinion.json` read verdict-first off the
describe-first schema, an arm that was measured (it collapsed toward `ask`)
rather than a path the model would write. The fixture specs
`email-verdict-opinion-v1.json` and `email-triage-opinion-v1.json` under
`lfm2d/tests/fixtures/specs/` carry the same two shapes for the tests.

### The opinion API: describe-then-read (`/v1/opinion`)

The read above stands the model at a slot the spec fixes. On this checkpoint
a verdict read *before* the model has described the command carries nothing
(F9 gold: AUC 0.46–0.55 at 99% mass); the generative path's own verdict
slot, read *after* `effect`/`scope`/`undo`, is the judgement (AUC 0.74,
25/74 recall at 4 false alarms). `/v1/opinion` serves that slot directly.
It is its own endpoint with its own contract (`docs/integration.md`
invariants 8–11): the typed-decision surface, in the README's words the
System 1 read; in code, the opinion read. How to write the spec it reads
is [`writing-a-spec.md`](writing-a-spec.md).

```json
POST /v1/opinion
{"spec": "email-triage-v2",
 "state": {"input": "my order 4471 still says preparing after six days",
           "facts": "Facts from the order system:\n..."},
 "context": null,
 "questions": [{"field": "verdict", "options": ["auto_close", "human_read"]}],
 "rendered": false, "use_cache": true, "timeout_ms": 30000}
```

- `spec` names a loaded spec; `GET /v1/opinion/specs` lists them with every
  schema field in emission order, its `kind` (`text`, `choice`, `boolean`)
  and a choice field's `options`. **Read the menu at runtime; never
  hard-code a field name or an option.**
- `state` is rendered into the user turn as `{facts}{input_label}:\n{input}`
  — the `facts` block verbatim (an app builds it; the daemon never does),
  then the spec's own `input_label` and the input. The label comes from
  the spec (required, top-level; the menu repeats it), so a paired
  generative run that sends the same bytes as `/v1/adjudicate`'s `input`
  sees the same prompt. `state.command` is a `400`: it was the shell-only
  spelling before 2026-09-24.
- `questions` names one or more `choice` fields of the spec; each
  question's `options` may narrow its enum and is scored in the spec's
  order. Several questions share ONE description: the engine walks the
  grammar once, stands at each asked slot as it passes it, scores the
  options there and walks on, so asking `scope`, `undo` and `verdict`
  costs one description plus three reads, not three descriptions.
  `answers` come back in emission order whatever order they were asked
  in, each with its own `rendered_sha256` over the text up to its own
  slot; `rendered` and `described` are the text and fields up to the
  last one. Each answer is bit-identical to the same question asked
  alone on the same cache path (`lfm2d/tests/opinion_real.rs`), and the
  walk leaves a described-cache entry at every slot it passed. Questions come from the spec on disk, never
  from request text: framing words in a rendered prompt move label tokens
  by an order of magnitude, and a prompt the instruments never scored is not
  one the daemon serves.
- `context` is reserved and must be `null`.
- `rendered: true` adds `rendered` to the response: the exact text the
  options continue — the spec prefix and user turn with the chat
  template's control tokens, then the description up to and including
  the last asked slot's `"<field>": "`. Each answer's `rendered_sha256`
  hashes this text cut at the end of its own slot, so a consumer checks an
  answer against the prefix ending at `"<its field>": "` — for a single
  question, the whole string. Off by default, and the
  key is absent rather than null: the hash is the audit trail, and the text
  repeats the whole spec prefix on every request. Under the SAME
  condition, `rendered_token_ids` carries the model's own token ids for
  that exact text — prefix + user turn (plain tokenization: never
  generated, so unambiguous) followed by every field the model actually
  sampled before the last slot, never a re-tokenization of the decoded
  `rendered` string. Added 2026-09-23 (kaibo review, F3) specifically to
  close `POST /v1/probe`'s replay loop: `rendered` alone is not safe to
  feed back through `/v1/probe`'s `text`/`decode_from` (BPE is not
  injective — a caller re-encoding generated text can silently land on a
  different token path than the model took), but `rendered_token_ids` fed
  through `/v1/probe`'s `ids`/`decode_from_token` is exact by
  construction. See "Probe and tokenize" below.

The daemon renders the prompt, generates every field before the question
under the output grammar (greedy, the repetition penalty, exactly the
generative path), stops when the generated text ends with `"<field>": "`,
and teacher-forces every option plus its close there (`",` when another
field follows, `"}` when the field is last). Each option and its close sit
on their own pre-tokens after the slot — the tokenizer's pre-tokenizer
decides that, checked at load on a probe and held at request time — so the
continuations are the tokens the model would have written.

A response, recorded on the shell spec `command-verdict-enum-v1` before it
moved out (its fields are that spec's, not the daemon's):

```json
{"model_id": "...", "snapshot_id": "...", "spec": "command-verdict-enum-v1",
 "described": [{"field": "effect", "value": "Removes build artifacts ..."},
               {"field": "scope", "value": "project"}, {"field": "undo", "value": "easy"}],
 "answers": [{"field": "verdict",
   "options": [{"option": "allow", "logprob": -0.100908, "first_logprob": -0.100907, "prob": 0.905, "tokens": [13537, 1377]},
               {"option": "ask", "logprob": -2.556361, "first_logprob": -2.556359, "prob": 0.078, "tokens": [1767, 1377]},
               {"option": "review", "logprob": -4.028756, "first_logprob": -4.028737, "prob": 0.018, "tokens": [63202, 1377]}],
   "sequence_mass": -0.0006, "first_token_mass": -0.0006, "margin": 0.827,
   "shared_tokens": 383, "scored_tokens": 6, "rendered_sha256": "..."}],
 "cache": {"prefix": "hit", "state": "miss", "described": "miss"},
 "prompt_tokens": 344, "cached_tokens": 327, "described_tokens": 39,
 "queue_ms": 1.1, "prefill_ms": 105.8, "describe_ms": 511.6, "read_ms": 46.6}
```

- **No winner.** Nothing in the response names a choice; the caller picks,
  from its own thresholds on its own data, per spec, and refits when
  `snapshot_id` changes.
- `described` is the fields the model wrote before the last asked slot,
  in emission order (a list, because `serde_json` sorts object keys). The
  read is conditioned on it, so it is part of the answer and is echoed; an
  earlier answer was read on a prefix of it, and that prefix includes the
  values the model wrote at each earlier asked slot.
- Per option: `logprob` is the raw sequence logprob (option plus close,
  full-vocabulary denominators), `first_logprob` the raw logprob of the first
  token alone (the number the F9 slot score reads), `prob` renormalised over
  the options asked. `sequence_mass` and `first_token_mass` are the raw log
  mass on the answer set: read `prob` and `margin` beside them, because both
  are blind to "never asked".
- `margin` is `prob[top] - prob[runner-up]`.
- `cache` names each layer's outcome, `hit`, `miss`, `bypass`
  (`use_cache: false`) or `skipped` (a described hit never consults the
  prompt checkpoint): the spec's resident prefix; the exact rendered
  prompt (one checkpoint per spec, exact repeat); and the *described*
  state — the prompt plus its generated description at a slot, kept per
  spec in a small LRU (`described_cache_capacity` on the menu, 16). The
  capacity counts ENTRIES, one per asked slot: a three-question request
  spends three, so the LRU holds about five such commands. Greedy
  decoding under the grammar is a pure function of the rendered prompt on a
  fixed backend, so the description and the model state after it are
  reusable computation, never a cached answer: a hit skips the prefill and
  the description and scores the options fresh, in tens of milliseconds.
  Every layer is keyed under the spec's identity, so a weight, tokenizer,
  template or spec change empties it.

Measured 2026-09-22 on ROCm, warm, `command-verdict-enum-v1`, over the F9
gold set (236 rows, inputs with facts blocks) paired on the F9 generative
run (the shell gold and run logs are in the author's private notes, not
published):

| | PT false alarms /96 | recall /74 | twin FA /61 | AUC (challenge) |
|---|---|---|---|---|
| generative slot score, raw P(allow) < 0.8 (F9) | 4 | 25 | 7 | 0.7357 |
| `/v1/opinion`, raw first-token P(allow) < 0.8 | 4 | 25 | 7 | 0.7355 |
| both, < 0.9 | 12 | 39 | 13 | — |

The endpoint *is* the slot score: on every row the described fields equal
the generative report's, the top option is the generative verdict, and the
read's `first_logprob` sits within the paired run's 4-decimal rounding of
the slot the generative path recorded (6e-5 nats; the Rust test reads
both unrounded and sees ~5e-6, the block-vs-token kernel difference above). Latency, p50
(p90): miss 888 ms (1405) — prefill 239 (586), description 575 (798) for
49 tokens, read 41 (57) — against 1248 ms (1726) for the generative
report; an immediate repeat hits the described cache in 46 ms (91) and
returns bit-identical numbers, 236 of 236. The two runs reproduce the
instruments exactly across daemon restarts. Bare commands without facts
read in 490–740 ms on a miss. A sweep longer than the cache evicts it: a
second pass over 236 rows is 236 misses, which is why the harness measures
the hit path by an immediate repeat.

`lfm2d/tests/opinion_real.rs` (ignored: it loads the 8B) pins the read to
the generative slot and the hit to the miss; `demo/xray.py` and
`demo/asked.py` show the read against a running daemon. The harnesses that
measured it on the F9 gold set (`benchmarks/lfm25/opinion_demo.py`,
`prompts/describe_read_eval.py`, `score_instruments.py`) left with the shell
material; they are in git at `f9ca081`.

**Escalation is the same forward pass, continued.** The generative
adjudicator is the same resident model on the same state. When
`/v1/adjudicate` receives the exact prompt bytes an opinion read left in
the described cache (`state.render()` is `input`), it resumes from that
state: the cached description is walked through the grammar and the
penalty's history as if this generation had written it, and only what
follows the slot — the question's value, `reason`, the close — is decoded.
The response carries `resumed_tokens`, `cached_tokens` equals
`prompt_tokens`, and `output`/`report` are byte-identical to a fresh
generation (`lfm2d/tests/opinion_real.rs`). So the cascade an app builds on
the opinion — read, decide, escalate the unsure ones — pays for the
description once. A cold request (`use_cache: false`) or one asking for
`distributions` never resumes.

A spec whose field name contains a quote or a backslash is refused at
load: slot detection could not tell its escaped key from a later field's.

Refused with 404, never queued: a `spec` naming nothing loaded (boot or
uploaded) — see "Runtime spec registration" below; this is the client's cue
to upload the spec and retry, not a malformed request. Refused with 400,
never queued: a field the spec lacks or that is not a `choice`, an option
outside the enum, fewer than two options, no question, a field asked
twice, a non-null `context`, control tokens in the state. Refused with
500, each with its own message: a description that ends before the slot
(the grammar makes every field required, so this is the context running
out or the turn ending where the grammar forbids it); the model writing
past the slot in one token (the grammar admits any token whose bytes
begin the value, so `"a` is legal there — the model leaving the canonical
path; 0 of 246 rows so far); an option that does not tokenize on its own
pretokens after the slot. The harness counts these as `http_error` rows,
never as reads.

### Runtime spec registration

Boot loads every `--opinion-spec` (zero is valid); a running daemon can
also load a spec at runtime, so a consumer (kaijutsu, for the shell specs)
owns and iterates on its own specs without a redeploy of this daemon. Ruled 2026-09-23, `docs/system1-split-plan.md` (git f9ca081) "Runtime spec
registration".

```
POST   /v1/opinion/specs        body: the spec's exact bytes
DELETE /v1/opinion/specs/{id}
```

- **Identity is a content hash.** A spec's `id` is the lowercase hex
  sha256 of the exact uploaded bytes — kaibo's CAS digest
  (`kaibo/src/cas.rs`, `Digest::of_bytes`) and this crate's own
  `hash::sha256_hex_bytes`. Nothing is canonicalized first: parsing into
  `serde_json::Value` and re-serializing would sort every object's keys,
  and a spec's field order IS its emission order (the grammar walks
  `required` in the order stated — `btreemap-sorting-reaches-the-prompt`
  bit this repo three times before). So two specs differing only in field
  order get different ids, on purpose; re-formatting a spec's whitespace
  also changes its id, which does no harm. Boot-time specs get their id
  the same way, from their file's bytes, and also answer to their
  file-stem name (as they always have) — `id` and name both work
  wherever `spec` is accepted.
- **`POST /v1/opinion/specs`** — the body IS the spec's bytes (any
  content type; the daemon parses it as JSON regardless), capped at 1 MiB
  (`MAX_SPEC_BYTES`) by a `DefaultBodyLimit` layer on this route — a spec's
  system prompt, tools and schema are text, never a checkpoint, so this is
  generous headroom, not a tuned limit. `413` for anything over that,
  uniformly (never a `400` for a body just over the cap but under axum's
  own larger built-in default). `201` when it was genuinely loaded now,
  `200` when that id was already loaded (boot or a previous upload) — no
  load runs the second time, which is what makes "upload any time you're
  not sure" free. `400` when the body doesn't parse as a spec. `422` with
  the load-time refusal text when it parses but cannot be served — the
  same checks a boot spec passes before it ever answers a request: the
  output schema doesn't compile to a grammar, an option or a field name
  doesn't tokenize the way the slot detector needs, the opinion block's
  options don't split cleanly. The response body is the registered spec's
  menu entry (`id` and
  `snapshot_id` included).
- **Memory only, bounded.** Uploaded specs live in memory, under
  `--opinion-spec-capacity` (env `LFM2D_OPINION_SPEC_CAPACITY`, default
  8) — each holds a resident prefix state and a described cache, same as
  a boot spec, which is why there's a limit. Past capacity, the least
  recently used upload is evicted to make room for a new one; serving a
  request against an upload OR re-registering it both count as use, so a
  spec in active rotation never gets evicted out from under a caller that
  keeps asking about it. Boot-time specs are never evicted and don't
  count against the capacity.
- **`DELETE /v1/opinion/specs/{id}`** — unloads an upload. `204` on
  deletion, `404` if `id` names nothing loaded (already deleted, evicted,
  or never registered), `403` if `id` names a boot-time spec — those
  cannot be deleted at runtime; restart the daemon with a different
  `--opinion-spec` list instead.
- **An id cannot change what it means**, so there is no `409` pin to
  worry about the way a mutable name would need one. `snapshot_id` still
  tells a consumer when the model *under* that spec changed (a weights,
  tokenizer, template, repetition-penalty, device-target or candle-build
  change) and invariant 8's
  rule to refit calibration on a new `snapshot_id` still applies —
  registering the same content twice never changes it.
- **`POST /v1/opinion` and `POST /v1/adjudicate`'s `spec` field accepts
  an id or a boot-time name, and is required on both.** Naming nothing
  loaded is `404` — never a fallback to a different spec. Omitting it on
  `/v1/adjudicate` is a `400` naming the menu (since 2026-09-24; before
  that it served the `--adjudicator-prompt` spec, a flag that no longer
  exists, so no spec is a default). An escalation from an opinion resumes from the
  SAME spec's described cache only when the generative call names that
  spec too — resolution shares the loaded spec instance between
  `/v1/opinion` and `/v1/adjudicate`, so this falls out of the id/name
  lookup rather than being special-cased.
- **The menu never comes from request text.** Registration is its own
  write, on its own route; `GET /v1/opinion/specs` (still) lists boot
  specs and uploads together, each with its `id`. A registration or
  eviction gets its own span and one log line: the id, the
  `snapshot_id`, how long the call took, and whether it evicted anything
  — `newly_loaded`/`evicted` in the response's telemetry, not the wire
  body.

## Probe and tokenize

Ruled 2026-09-23, `docs/system1-split-plan.md` (git f9ca081) "Tokenize and probe
endpoints". Amy: "if we don't still have a tokenizing endpoint on lfm2d I
think we should still have that... might add a general inference endpoint
too so we can use it to probe the model consistently." Neither `/v1/opinion`
nor `/v1/adjudicate` fit: both take questions only from a loaded spec's
menu, never request text, on purpose (framing words move label tokens by
an order of magnitude — "The opinion API" above). These two routes are
the escape hatch: raw text in, raw numbers out, no menu, no judgement
contract.

### `POST /v1/tokenize`

```json
{"model": "LFM2.5-8B-A1B-Q5_K_M",
 "text": "auto_close", "context": "{\"verdict\": \""}
```

- `model` is any id `GET /v1/models` lists (an encoder head), or the
  adjudicator's own id (`GET /v1/adjudicator`'s `model_id`) — each is a
  separate tokenizer, read from a `TokenizerRegistry`
  (`lfm2d/src/tokenize_api.rs`) built once at startup from every loaded
  model's own tokenizer, cloned out BEFORE that model's owning engine
  moves into its worker thread (`RealEngine::tokenizers`,
  `Adjudicator::tokenizer_clone`, both called from `main.rs` ahead of
  `WorkerHandle::spawn_crash_on_panic`/`adjudicator::Handle::spawn`).
  Unknown `model` is `404`. **Every clone the
  registry stores has truncation and padding explicitly cleared**
  (`TokenizerRegistry::insert`), regardless of what the source checkpoint
  set them to for its OWN inference path — `Lfm2Embedding` loads its
  tokenizer with 512-token truncation (`MAX_SEQ_LEN`), which is correct
  for embedding but would otherwise make `/v1/tokenize` silently report
  only the first 512 tokens of a longer input as if that were the whole
  thing (fixed 2026-09-23, kaibo review; same class of hazard
  `Checkpoint::load` already guards against for the adjudicator's own
  tokenizer).
- **Answered entirely on the request handler, over the cloned tokenizer —
  it shares no queue with either worker.** `lfm2d/tests/tokenize_api.rs`
  proves this directly: a `Generator` double that never returns still lets
  `/v1/tokenize` answer in milliseconds.
- Response: `ids` (vocabulary ids), `tokens` (`{id, token, start, end}` per
  token — `start`/`end` are UTF-8 BYTE offsets into `text`, same
  convention as `/v1/spans`; `token` is the tokenizer's own byte-level BPE
  piece, e.g. a leading space reads as `"Ġ..."`, NOT a decoded string),
  and `tokenizer_hash` (sha256 of that model's `tokenizer.json`, same
  convention as `weight_hash`).
- With `context`: a `context` block — the tokens `text` occupies once
  `context` precedes it (found by stripping the longest common prefix
  `context`'s own encoding shares with `context + text`'s), and
  `suffix_stable`: whether `text`, tokenized ALONE, is exactly the tail of
  `context + text`'s tokenization. This is the SAME readability check
  `LoadedSpec::load` runs on every opinion option before trusting it can
  be taught-forced at a slot, and `describe_then_read` re-runs per request
  (`suffix_is_stable`, `lfm2d/src/adjudicator.rs`) — factored into one
  function all three call, so `/v1/tokenize` reports exactly the fact
  those paths enforce, never a second implementation of it that could
  drift. `false` means a BPE merge crossed the `context`/`text` boundary
  (the classic case: `context` ends in a space, so the option's first
  token picks up a leading-space variant it would not have alone) — the
  same hazard that makes teacher-forcing arbitrary text at an arbitrary
  slot unsafe without this check.

### `POST /v1/probe`

Raw inference over exact text, on the adjudicator's own stack. **An
instrument, not a judgement API: no calibration contract, and
`docs/integration.md` invariants 8–11 do not apply to it.** It takes
request text by design — unlike `/v1/opinion`, whose questions come only
from a spec's menu — which is exactly why it is a separate route with a
separate, weaker contract.

On by default whenever the adjudicator is configured; `--no-probe` (or
`LFM2D_PROBE=0`/`false`) removes the route entirely (`404`, never
present-but-`403` — an unauthenticated prober should not learn "this
feature exists but is off" from "this daemon never had it"). Runs as a
worker `Job` like `/v1/opinion`, sharing the adjudicator's bounded queue
and deadline handling.

```json
{"text": "cargo clean", "top_k": 5, "continuations": ["allow\"}", "ask\"}"],
 "generate": 0, "use_cache": true, "timeout_ms": 30000}
```

or, in place of `text`:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "cargo clean"}],
 "assistant_prefill": "{\"verdict\": \""}
```

or, the exact-ids form:

```json
{"ids": [124894, 124899, ...], "decode_from_token": 238, "continuations": ["allow\"}", "ask\"}"]}
```

- **Input**: exactly one of `text`, `messages`, or `ids`.
  - `text`: exact bytes, fed as-is — no template, and none of
    `validate_text`'s literal-control-token refusal that every other
    prompt path in this crate enforces, because the whole point is
    inspecting what the model does with bytes those paths would refuse to
    render.
  - `messages`: rendered `<|startoftext|>` then one
    `<|im_start|>{role}\n{content}<|im_end|>\n` per message, then an
    opened, unclosed assistant turn, optionally continued by
    `assistant_prefill` — the SAME renderer
    `PromptSpec::render_prefix`/`render_user_turn` use for a spec's own
    turns, reused rather than reimplemented. `role` must be
    `system`/`user`/`assistant`; message `content` and `assistant_prefill`
    DO go through `validate_text` (they sit inside the templated
    structure, so a literal control token in them would corrupt it, same
    reasoning as every spec-driven path).
  - `ids`: exact token ids, teacher-forced VERBATIM — no tokenization at
    all. **`text`/`messages` are re-tokenized fresh by this endpoint,
    which is exact for bytes a caller wrote itself but NOT for bytes
    containing a model's own generated output**: BPE is not injective in
    the decode-then-re-encode direction, so a caller replaying generated
    text can silently land on a different token path than the model
    actually sampled (the same "leaving the canonical path" hazard
    `describe_then_read` already guards against on its own decode). Use
    `ids` whenever the bytes being replayed came out of a generation —
    `/v1/opinion`'s `rendered_token_ids` (below) is built for exactly
    this. `ids` needs its own schedule split, `decode_from_token` (a
    TOKEN INDEX, not a byte offset — trivially exact, bounds-checked at
    request-validation time, `0..=ids.len()`), in place of `decode_from`.
  - The response echoes `rendered` when `messages` OR `ids` was used (a
    `text` request already has its own bytes in the request body; an
    `ids` request gets a DECODED string back for a human to read, used
    for nothing else) and always carries `rendered_sha256`.
- `top_k` (0–20, default 5, `crate::types::MAX_DISTRIBUTION_TOP_K` — the
  same cap `/v1/adjudicate`'s `distributions.top_k` uses): full-vocabulary
  top-k logprobs at the last input position, RAW (no repetition penalty,
  no renormalization) — same convention as every other distribution this
  daemon reports.
- `continuations` (0–32 strings, each non-empty): each is teacher-forced
  after the input and scored via
  `crate::opinion::score_continuations_verbose` — the SAME forward-pass
  loop `/v1/opinion` and `/v1/adjudicate`'s `opinion: true` share, now
  also returning the PER-TOKEN logprobs those endpoints throw away.
  Per option: `tokens` (the continuation's own canonical encoding),
  `token_logprobs`, `sequence_logprob` (their sum), `first_logprob`
  (`token_logprobs[0]`, the F9 slot-score number), and `canonical` —
  whether this continuation's tokens, encoded alone, are exactly the tail
  of the rendered input with this text appended
  (`suffix_is_stable` again). `canonical: false` is reported, never
  refused: this is an inspection tool, and a non-canonical continuation is
  still real data. The block also carries `sequence_mass`
  (`logsumexp` of the options' sequence logprobs) and the ONE renormalized
  number this endpoint allows, `prob` over the continuations — read it
  beside `sequence_mass`, same "low mass means unasked, not wrong"
  convention as everywhere else in this crate.
- `generate` (0–256): greedy steps under the EXACT SAME deterministic
  policy production uses (sign-aware repetition penalty over `full_ids`,
  no grammar — `crate::constrain::Decoder::new(None, ...)`), stopping
  early at end-of-text. Each step carries its own `top_k` via
  `crate::types::step_distribution_under`, the same machinery
  `/v1/adjudicate`'s `distributions` uses. Step 0's distribution IS
  "`top_k` at the last input position" — computed once, not twice.
- `use_cache` (default true): the daemon picks the LONGEST loaded spec
  (boot or uploaded) whose resident prefix is a STRICT token-id prefix of
  the input up to `decode_from`/`decode_from_token` (or the whole input,
  when neither is given) — `best_prefix_match`/`SpecStore::
  resolve_best_prefix_mut`, deterministic regardless of iteration or LRU
  order (a first-match search used to make the resumed spec depend on
  which spec happened to be checked first — fixed 2026-09-23, kaibo
  review). "STRICT" matters: a prefix EQUAL to the bulk-phase ids is not a
  candidate (nothing would be left to bulk-forward), so that case resumes
  a SHORTER match or runs the bulk phase cold, never an error — a
  reported choice, not a refusal (also fixed 2026-09-23; it used to be a
  `400 nothing to forward`). A hit on an UPLOADED spec touches it to the
  back of the LRU, same as naming it by id would ("served-or-registered =
  use," `docs/system1-split-plan.md` (git f9ca081) "Runtime spec registration" — also
  fixed 2026-09-23; a probe resuming an upload used to leave its LRU
  position untouched). Once found, the daemon clones that spec's prefix
  state and bulk-forwards only the suffix up to `decode_from`
  (`forward_chunks`, `CHUNK`-sized pieces — the SAME function
  `PromptCache::prepare`'s cache-miss branch uses, so the two schedules
  cannot silently diverge into two implementations of "forward a suffix
  onto a resident prefix"). `false` always runs the bulk phase cold from
  zero. The response's `cache` block always states which happened:
  `used_cache`, `resumed_spec` (the matched spec's id, if any),
  `cached_tokens`, and the schedule split (`prefill_tokens`,
  `stepwise_tokens` — see `decode_from`). **Never mutates any spec's own
  request-serving cache** (`PromptCache::prepare`'s `ready` slot) — a
  probe against arbitrary text must not silently evict a spec's warm
  production cache as a side effect; measured directly in
  `lfm2d/tests/probe_real.rs` (a `/v1/opinion` read taken after a probe
  that warm-resumed the same spec is bit-identical to one taken before
  it).
- `decode_from` (the `text`/`messages` form's schedule split — optional, a
  BYTE offset into the rendered input, the same bytes `rendered_sha256`
  hashes) or `decode_from_token` (the `ids` form's — a TOKEN INDEX,
  trivially exact and bounds-checked at request-validation time, no
  tokenizer needed): splits the schedule. Everything BEFORE it is
  bulk-forwarded exactly as `use_cache` describes above; everything FROM
  it onward is forwarded ONE TOKEN AT A TIME via
  `forward_chunks(..., chunk_size: 1, ...)` — the SAME `model.forward(&[token],
  &mut state)` call `describe_then_read`'s decode loop makes per generated
  token, reused at that chunk size rather than reimplemented (see
  `forward_chunks`'s doc comment). `decode_from` (the byte-offset form)
  must land exactly on a token boundary of THIS endpoint's own fresh
  tokenization of the input — refused with `400` naming the nearest
  boundaries either side otherwise, never rounded silently; note this
  inherits the `text`/`messages` re-tokenization caveat above, so a
  boundary that looks right can still, in principle, land on the wrong
  token if the bytes came out of a generation — prefer `ids`/
  `decode_from_token` when they're available. This is what lets a caller
  replay `/v1/opinion`'s OWN schedule, not just the bulk-only one below:
  set it to the byte length (or, with `ids`, the token count) of the
  prompt text BEFORE any of the model's own output (prefix + user turn,
  ending right where the forced object/key opening — `{"verdict`, in the
  shipped verdict specs — begins), and the response reproduces
  `/v1/opinion`'s slot bit-for-bit (measured below). Omitted, the whole
  input stays in the bulk phase — today's only schedule before this field
  existed, and still the default.
- **Bounds**: input tokens must fit the adjudicator context (`400`
  otherwise); `continuations`/`generate` are capped as above (also `400`,
  never clamped); `timeout_ms` 1–120000.
- **Identity block**: `model_id`, `weight_hash`, `tokenizer_hash`,
  `backend`, `device`, `candle_rev`, `dtype`, `sampling` (mirrors
  `PrefixInfo`'s fields, minus the spec-specific ones — a probe is not bound
  to a spec), plus `rendered_sha256`.
- Telemetry: one span per probe, fields are lengths/counts only
  (`input_tokens`, `cached_tokens`, `continuations`, `generated`,
  `prefill_ms`, `score_ms`) — never the text, matching this crate's
  blanket rule (`lib.rs`'s module docs, "no span, log, or metric anywhere
  in this crate ever records input text").

**What each schedule can reproduce bit-identically — measured, not
assumed.** Without `decode_from`, `/v1/probe`'s warm resume is ONE bulk
forward of everything after a spec's resident prefix.
`POST /v1/adjudicate {"opinion": true}` (the F8 read, "Opinion reads"
above) is the SAME shape: its `rendered` text (prefix + user turn + the
spec's fixed `prefill` string) is bulk-prefilled in one call via
`PromptCache::prepare`, no decode loop at all — so a bulk-only probe fed
that exact text plus the spec's options as continuations reproduces it
EXACTLY (`lfm2d/tests/probe_real.rs`,
`probe_reproduces_the_f8_opinion_reads_slot_bit_identically_when_warm`,
`#[ignore]`d: token identity, `first_logprob`, and `sequence_logprob` all
bit-identical, `cached_tokens` equal to the spec's resident prefix
length).

`POST /v1/opinion`'s `describe_then_read` is NOT the same shape, even for
a spec with nothing to describe: it still writes the JSON object's own
opening (`{"verdict": "` — several tokens) through a GREEDY DECODE LOOP,
one token at a time, because that text is GENERATED under the grammar,
not given outright. A bulk forward of N tokens and N single-token forwards
are the same computation in exact arithmetic, but not on this backend —
quantized ROCm kernels differ by batch size (`lfm25-chunk-kernels.md`'s
"block size picks the kernel"; `cold-and-cached-schedules-disagree`). So a
**bulk-only** probe fed `/v1/opinion`'s own `rendered` text does NOT
reproduce it bit-for-bit — measured on ROCm, `cargo clean`, `ask`'s
`first_logprob` disagreed by 0.05 nats with ZERO fields preceding the
slot (`command-verdict-opinion-v1`) and by up to 3.4 nats with three
(`command-verdict-enum-v1`, `effect`/`scope`/`undo` before `verdict`) —
small in the zero-field case, large enough to flip a threshold in the
three-field one.

**`decode_from` closes that gap.** Set to the byte length of the prompt
text alone (before any of the model's own output), a probe fed
`/v1/opinion`'s `rendered` text reproduces its slot EXACTLY — same token
identity, same `first_logprob`, same `sequence_logprob` — for BOTH the
zero-described-fields spec and the three-fields one that measured the 3.4
nat gap above (`lfm2d/tests/probe_real.rs`,
`probe_reproduces_the_opinion_reads_slot_bit_identically_with_decode_from`,
`#[ignore]`d, 2026-09-23). That test uses the `text`+`decode_from` form,
which still carries the re-tokenization caveat in principle even though it
passed here; a caller that wants the exactness guaranteed rather than
observed uses `ids` (from `/v1/opinion`'s `rendered_token_ids`) +
`decode_from_token` instead — see "The opinion API" above. **So:
`/v1/probe` reproduces `POST /v1/adjudicate {"opinion": true}` with the
bulk-only (default) schedule, and reproduces `POST /v1/opinion` when
given its OWN decode schedule via `decode_from`/`decode_from_token`** —
the caller supplies the split, because only the caller (having asked
`/v1/opinion` or read `describe_then_read`'s own code) knows where a
given render's generation actually began; `/v1/probe` has no spec/field
awareness to infer it. A probe's warm resume never mutates the spec it
resumes — measured directly: a `/v1/opinion` read taken right after a
probe that warm-resumed the same spec is bit-identical to one taken
before it, same test. Cold (`use_cache: false`) is
reported the same way regardless of schedule: measured, never asserted
equal to warm, exactly because production is always warm and a cold probe
answers a genuinely different question.

## Context budget

`--adjudicator-context` accepts 128–32768 tokens (the GGUF declares 128000).
Chat keeps preserved reasoning in its history, which fills 4096 tokens within
a few turns. Attention is eager: each 128-token prefill chunk materializes
`32 heads x 128 x kv_len` f32 scores, 480 MiB at 30k, and decode reads the
whole K/V for every token. What a deep context costs on gfx1151, cold
prefill from an empty state straight on the model, host otherwise idle
(2026-09-26, candle with allocator size classes):

| depth | prefill | decode | peak GPU (model 6.4 GiB) | one state's K/V |
|---:|---:|---:|---:|---:|
| 128 | 0.2 s | 94–105 tok/s | 6.6 GiB | 3 MiB |
| 8192 | 13.4 s (613 tok/s) | 68 tok/s | 9.3 GiB | 192 MiB |
| 16384 | 31 s (529 tok/s) | 50 tok/s | 12.4 GiB | 384 MiB |
| 30000 | 71 s (423 tok/s) | 34 tok/s | 17.4 GiB | 768 MiB |

n=2 at 128–16384 and n=1 at 30000; 128 decoded tokens per depth. A state's
K/V is 24 KiB per token (6 attention layers x K and V x 8 heads x 64 x f32)
at the power-of-two capacity the K/V allocator rounds to, so any state past
16384 tokens holds 768 MiB.

Before the candle allocator used size classes, the ROCm caching allocator
kept one set of score buffers per prefill chunk: +3.4 GiB after a cold 4096
prefill, +12.9 GiB after 8192, and a 30k prefill exhausted the 128 GB host.
That parked memory is GTT, host RAM that the OOM killer does not count
against the process, so it killed other pods first. A context above 8192
needs that allocator fix (candle `6223100c`, "rocm: geometric size classes
for allocator buckets above 64 KiB"); with it, what a prefill parks grows
linearly: 2.7 GiB after 8192, 5.9 GiB after 16384, 10.8 GiB after 30000.

Two things scale with the budget rather than with traffic. The caches evict
by entry count, not bytes: 16 described states per spec plus one ready
prompt, each up to 768 MiB for a long read. And `/v1/probe` `ids` are capped
by the context only, not by bytes, so one request can prefill the whole
budget. A cold opinion read with a 64 KiB `facts` block is about 18k tokens
and 36 s, over the default 30 s `timeout_ms`.

## Snapshot semantics

Candle's `quantized_lfm2_moe::{Model, State}` separates immutable weights
from attention K/V, convolution history, and position. `State::clone()` shares
immutable prefixes of append-only KV buffers and immutable convolution state.
KV appends reserve unused tail space, growing or forking the buffer when
necessary. Convolution updates allocate replacement tensors. Forward commits
the new state only after success; failed writes cannot overwrite saved data.
Foreign-model state, bad tokens, and context overflow are errors.

The [KV allocator](lfm25-kv-cache.md) amortizes prefix copying over a linear
continuation. Branches copy their prefix when their desired tail is occupied.
Attention still materializes repeated KV heads; paged attention is not present.
The daemon owns one fixed prefix and one complete-input checkpoint with
saved logits, and one worker thread with two bounded queues of eight: opinion
reads and probes in one, generation and spec registration/deletion in the
other. A generation pauses before every prefill chunk and decoded token, and
the worker serves waiting reads and probes there, so a read waits one chunk
or one token, not a whole generation; the generation's output does not move
(`lfm2d/tests/interleave_real.rs`), and its `prefill_ms`/`decode_ms` and
deadline include the reads it paused for. Generations and spec admin never
overtake each other. Exact input repeats reuse the complete-input
checkpoint; `cached_tokens` then equals `prompt_tokens`.
`input_cache_capacity: 1` advertises this behavior. Cold requests bypass
cache reads and writes. Deadlines, disconnected callers, and shutdown discard the current
branch. The encoder worker is separate. Shutdown waits for both models to
drop before flushing telemetry.

The JSON rubric is 331 tokens: about 7.76 MiB of F32 attention KV plus
0.42 MiB of convolution state. The model uses the fork's existing Q5K/Q6K
indexed ROCm expert operations. No new HIP kernel was necessary for this
first path. CPU operation explicitly dequantizes expert weights and is a
memory-heavy correctness reference; full-checkpoint CPU serving has not
been performance-qualified. ROCm/F32 is the tested serving configuration.

Disk persistence is deferred. For this prefix it would avoid about a
second of startup prefill while weight loading and hashing would remain.
It becomes more useful with long examples, several prefixes, or frequent
restarts. A future format must serialize **both KV and convolution state**,
position, prefix token IDs, model/tokenizer/template hashes, dtype and RoPE
configuration, with a version and checksums. Loading must verify identity
and transfer host tensors back to the GPU. A bare KV dump is insufficient.

## Validation and evidence

The Candle fixture has independently generated NumPy logits for a tiny real
GGUF with dense and MoE FFNs, convolution, attention, and grouped queries.
Tests compare full, incremental, and every prefix split; verify A/B/A branch
isolation; reject invalid/foreign state; and inject a failure after the layers
to check transactional commit. A separate explicitly requested ROCm test
checks packed Q5K/Q6K expert operations against dequantized reference math.

```bash
# in a checkout of the fork's lfm25-trace branch
cargo test -p candle-transformers --lib 'models::quantized_lfm2_moe'
cargo test -p candle-transformers --test lfm2_moe
cargo test -p candle-transformers --release --features rocm --lib \
  'rocm_quantized_experts_match_dequantized_reference' -- --ignored

# in this repo
cargo test -p lfm2d
cargo test -p lfm2d --release --features rocm --test opinion_real -- --ignored
```

The daemon's real-PII tests also need the PII-Detector checkpoint
([development.md](development.md)); set `LFM2_TOKEN_CLF_DIR` when using a worktree
without `.models/`.
The end-to-end evaluator behind the 2026-09-13 numbers below
(`benchmarks/lfm25/evaluate.py`: four shell-severity cases, cold/cached/
repeated, errors/deadlines, reuse after cancellation, in-flight SIGTERM)
left with the shell material and is in git at `f9ca081`. Four cases were
an integration probe, not an accuracy certification.

On the real Q5_K_M checkpoint, a 40-token greedy smoke continuation matched
llama.cpp exactly. Cold versus cached fixed-input logits for the adjudicator
also matched exactly (max absolute difference and RMS both zero). This is
bounded evidence; different devices, quantizations, shapes, and optimizations
need their own comparisons. The early native-tool prompt produced divergent
cold/cached continuations; that path remains an unqualified comparison mode.

**2026-09-19, the verdict prompt: cold and cached are NOT the same, and the
difference reaches the answer.** Same daemon, same input bytes, greedy, ROCm,
`command-verdict-enum-v1`, 40 val_F rows, the only difference `use_cache`:

- identical generated text on **13 of 40** rows,
- identical verdict on **39 of 40** — one verdict moved on a cache hit alone,
- per-word |delta| at the verdict slot p50 **0.145** nats, p95 1.71, max 3.15.

The resident prefix is 327 tokens and `327 % 128 = 71`, so the cached path's
chunk boundaries sit 71 tokens off the cold path's and even the prefix region is
built with a different last-chunk shape. This does not contradict the
fixed-input logit parity above — that was a bounded measurement on a different
prompt — but it does mean cold/cached equality must be re-measured per prompt and
per optimization, never inherited.

**What this does and does not say about production.** Both production paths are
warm: a new input prefills from the resident prefix (`cached_tokens` = 327), and an
exact repeat of the previous input replays the stored state and logits unchanged.
`use_cache: false` is only ever set by a measurement harness. So production
verdicts are not nondeterministic on this account — every input meets the same
schedule. What the measurement does say is that **a cold reader does not reproduce
what production answered**, which makes `use_cache: false` a poor baseline for
anything meant to describe the daemon, and puts a floor under every cold-path
probe. `lfm25-examine` is a cold reader. `benchmarks/lfm25/prompts/verdict_eval.py
--no-cache` was the arm (git `f9ca081`), and `docs/lfm25-grouped-prefill.md` predicted exactly
this: "Cold and cached chunk schedules now produce different long greedy
generations."

The [2026-09-13 results](../benchmarks/lfm25/results/2026-09-13.json)
record 12/12 schema-and-severity passes (four cases, cold/cached/repeated),
with all corresponding generated text identical. Median synchronized prefill
was 148 ms cached, 1158 ms cold, and 158 ms on repeated branches. Decoding
these 839–1500-token responses took 11.3–22.1 seconds; a later reuse check
slowed to 26.7 seconds on the shared host. Do not infer a throughput win over
Ollama/llama.cpp from this run. In-flight SIGTERM returned 408 and the daemon
exited 0 after model teardown. The full daemon suite passed 148 tests with
one existing ignored test; the final adjudicator suite passed 12 tests.
Candle's three new unit tests, two GGUF integration tests, and explicit ROCm
expert test passed. Targeted Candle clippy and daemon all-targets clippy
(`--no-deps`) passed with warnings denied.

Kaibo's default cast (`crusoe/GLM-5.3`) reviewed whole source files. It found no
snapshot alias/transaction flaw and highlighted model-reference coverage,
template fidelity, worker shutdown, and service tests. Those areas received
additional checks. Review is supporting evidence, not a substitute for tests.

## Follow-up work

- Broader adjudication cases and explanation grading. The reset case can
  name the correct severity while incorrectly claiming discarded uncommitted
  work is recoverable. Schema validation cannot catch that.
- ~~Shorter reasoning and grammar-constrained final JSON~~ — done. The JSON
  is grammar-constrained, and `reasoning: "closed"` removes the reasoning pass
  outright for schema-bearing prompts. Latency is now the report's own tokens.
  Still open for `"open"`, which is not built alongside a schema.
- An append-efficient KV allocator and less GQA materialization; grouped
  quantized MoE prefill is implemented.
- Wider fixed-token reference comparisons, including the native-tool prompt.
- The old dense `lfm2`/`quantized_lfm2` cached multi-token convolution paths
  ignore existing history. This new model fixes its own path; repair and
  regression-test the older modules separately. Not to be reached for as the
  explanation of a chunk-shaped effect *here* — it was, once, and the answer
  was the quantized matmul's `b_size <= 8` kernel switch
  (`docs/lfm25-chunk-kernels.md`).
