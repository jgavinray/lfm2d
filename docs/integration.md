# Consumer integration contract

What lfm2d promises the programs that call it. Consumers cite this file
by invariant number, so **the numbering below is part of the contract —
add new invariants at the end, never renumber**. An invariant that no
longer applies is marked RETIRED and keeps its number.

The first consumer was kaijutsu's `S50-lfm2d.kai` advisory hook, which
scores shell commands through `/v1/cascade`. On 2026-09-24 the shell
severity head and the cascade left lfm2d (Amy: lfm2d becomes "a general
system1 model built around the LFM suite"). That hook keeps calling the
frozen `lfm2d-1` pod, whose image still serves them under the invariants
as they read at commit `f9ca081`; this file describes the tree it sits in.

Wire shapes (exact field names and JSON structure) are pinned by
`lfm2d/tests/contract_serde.rs`. This file holds the semantic invariants
that shapes alone can't express. The endpoint reference and the
non-normative client-design notes are in `lfm2d/README.md`.

## Invariants

**1. `GET /v1/models` is the source of truth for label vocabulary.**
Read it at fire time. Never hard-code a label name, count, or position
into a consumer — a baked-in label goes stale silently the next time the
daemon's checkpoint changes.

**2. The label vocabulary changes between checkpoints.** Not
hypothetically: the retired shell severity head went from
`destructive` / `informative` / `mutating` (v6) to `informative` /
`situation-normal` / `data-critical` (v8 on). A consumer that matches on
label strings breaks on rollback or upgrade; invariant 1 is how it
survives.

**3. RETIRED (2026-09-24).** Promised ascending-severity label order for
the shell severity classifiers. lfm2d no longer ships a severity head,
and since the same day serves no sequence classifier at all
(`Lfm2SequenceClassifier` remains in the library).

**4. RETIRED (2026-09-24).** Described `/v1/cascade`'s `models` array.
The cascade left lfm2d.

**5. Scores are advisory.** An encoder head's scores are wrong in both
directions on real traffic (measured repeatedly on the retired shell head;
its history is in git). A consumer may let a score RAISE a prompt that
would not otherwise fire; it must never let one lower a prompt,
auto-decide, or silently allow — kaijutsu's ledger design (2026-08-17
ruling) is the reference posture. A score is also a function of (weights,
input, device, scoring stack): a torch scorer flipped ~3% of verdicts
between CPU and GPU on identical weights and input, so byte-identical
reproduction across hosts and stacks is not promised.

**6. Fail open, loudly.** When lfm2d is unreachable, slow, or answers
with an unexpected shape, a consumer must proceed with its baseline
controls unchanged and record that it skipped — never block on the
daemon's availability. (The daemon itself has been down for 21-hour
stretches; a guard that denied on outage would have frozen every seat.)

**7. RETIRED (2026-09-24) as a cascade rule; the lesson stands.** It
told consumers to rank clauses within a statement through `/v1/cascade`
instead of thresholding the severity head, because that head's benign and
data-critical bands touched (0.3415 vs 0.3440) and its axis was corpus
familiarity, not danger. The cascade and the head are gone. What stays:
lfm2d promises no global threshold for any head's scores; a consumer that
needs one fits it on its own data and owns it (invariant 8 says the same
for opinions).

**8. An opinion is a distribution, not a decision.** `POST /v1/opinion`
(`docs/lfm25-adjudicator.md`, "The opinion API") never names a choice. A
consumer that acts on one picks the option itself, from thresholds fitted
on its own data, per spec, and refits when `snapshot_id` changes —
calibration does not transfer between specs, and the failure is silent.
`snapshot_id` covers the weights, tokenizer, template, rendered prefix,
repetition penalty and spec, and since 2026-09-24 the device target
(`GET /v1/adjudicator`'s `device`, e.g. `rocm:gfx1151:hip7.2`,
`cuda:sm_121:nvcc13.0:drv580.173.02`, or `sycl:2026.1:Intel(R) Arc(TM) Pro B70
Graphics`) and the candle build (`candle_rev`), because the same weights give
different numbers on another GPU target, kernel build or, on CUDA, driver. The
SYCL identity names the oneAPI compiler that built the kernels and the device
the selected ordinal resolved to. On Metal `device` is still the bare
backend name.
Invariant 5 still holds for scores from the encoder heads; an opinion is a
different instrument with this contract of its own.

**9. Read the raw mass beside the renormalised probability.** `prob` and
`margin` are renormalised over the options asked; `sequence_mass` and
`first_token_mass` are the raw log mass the model put on that answer set.
A low mass means the model was never asked this question here, and a
renormalised number over it is noise that looks like an answer. A consumer
that thresholds `prob` or `margin` alone is thresholding that noise on
low-mass rows; gate on mass first, or record it beside every decision.

**10. An opinion may raise, never lower.** A static denial upstream (a
consumer's own policy, kaijutsu's ledger) is never lowered by an opinion.
An opinion may escalate to the generative adjudicator; it is not the
adjudicator, and the description it echoes is the model's own text, not
evidence a consumer may act on.

**11. Specs, fields and options are read from the menu at runtime.**
`GET /v1/opinion/specs` lists every loaded spec with its fields in emission
order and each choice field's options. Never hard-code a spec name, a
field name, an option, or an option's position — invariants 1–3 again,
for the opinion vocabulary. `described` is an ordered list of
`{field, value}`, not an object; the order is the spec's. The menu now
also carries uploads (invariant 12) beside the boot-time specs.

**12. A spec's `id` is a content hash; an unknown `spec` means upload it.**
Every menu entry (boot or uploaded) has an `id`: the lowercase hex sha256
of the exact spec bytes, computed over the raw upload — never a
re-serialized/canonicalized form, so two specs differing only in field
order get different ids (a spec's field order is its emission order, which
is meaning, not formatting). `POST /v1/opinion/specs` registers a spec at
runtime; the body IS the spec's bytes. Registering bytes already loaded
(boot or upload) is free — `200`, no work done — which is what makes
"upload any time you're unsure" a correct client strategy, not just a
convenient one. `POST /v1/opinion` and `POST /v1/adjudicate`'s `spec`
field accepts an id or a boot-time spec's name; naming nothing loaded is
now `404`, not `400` — the client's cue to upload (or re-upload) and
retry, never a reason to fall back to a different spec or invent one.
Uploads are held in memory only, under a bounded least-recently-used
cache (`--opinion-spec-capacity`); an evicted upload's next request is
also a clean `404`. Boot-time specs (`--opinion-spec`) are never
evicted and cannot be deleted at runtime (`DELETE /v1/opinion/specs/{id}`
on one is `403`). There is no default spec (2026-09-24): `spec` is
required on both routes, and `/v1/adjudicate` without it is a `400`. See
`docs/lfm25-adjudicator.md` "Runtime spec registration" for the wire, and
`docs/system1-split-plan.md` (git f9ca081) "Runtime spec registration" for the ruling
this implements (each consumer owns its specs and uploads them at
startup and on change; lfm2d ships only demo props).

**13. `/v1/tokenize` and `/v1/probe` are instruments, not judgement
APIs — invariants 5 and 8–11 do not apply to either.** Neither returns a
decision, a calibrated score, or an opinion; `/v1/probe` takes free-form
request text by design (unlike `/v1/opinion`, whose questions come only
from a loaded spec's menu — invariant 11 exists precisely because that
one doesn't take request text). A consumer must not treat a `/v1/probe`
number as comparable to a `/v1/opinion`/`/v1/adjudicate` verdict without
checking first WHICH schedule it ran: with `decode_from`/
`decode_from_token` omitted, `/v1/probe`'s bulk-only resume reproduces
`POST /v1/adjudicate {"opinion": true}`'s read bit-identically, but does
**NOT** reproduce `POST /v1/opinion`'s own read that way — measured
disagreement up to several nats at the same slot on the same input,
because the two paths take different kernels for the same tokens on this
backend. `/v1/probe` CAN reproduce `POST /v1/opinion`'s own read
bit-identically, but only when the caller supplies the split point where
that read's generation actually began, since `/v1/probe` has no
spec/field awareness to infer it itself; without it, or with the wrong
one, the numbers are a different, related computation, not
`/v1/opinion`'s answer. Two ways to supply it, not equally safe:
`text`/`messages` + `decode_from` (a byte offset, re-tokenized fresh —
exact for bytes the caller wrote, NOT guaranteed exact for bytes that
came out of a generation, since BPE is not injective in the
decode-then-re-encode direction) or `ids` + `decode_from_token` (exact
token ids, teacher-forced verbatim, immune to that hazard by
construction — feed it `/v1/opinion`'s own `rendered_token_ids`, present
under the same `rendered: true` condition as `rendered` itself). Prefer
the `ids` form whenever replaying a generation
(`docs/lfm25-adjudicator.md` "Probe and tokenize"). `/v1/probe` may be
disabled entirely (`--no-probe`/`LFM2D_PROBE=0`) without affecting
`/v1/opinion` or `/v1/adjudicate`; a consumer must not assume its
presence. `/v1/tokenize` carries no invariant beyond the general ones
(1–3, for whichever model's vocabulary is asked about) — it exposes
tokenization, not a scored or judged quantity.

**14. A spec names its own input; send `state.input` and read the label
from the menu.** Every spec carries a required top-level `input_label`
(2026-09-24; one line, no colon, no control tokens, at most 64 bytes), and `/v1/opinion` renders the user turn as
`{facts}{input_label}:\n{input}` from `state: {"input", "facts"?}`. The
menu entry repeats `input_label`; never hard-code it (invariant 11). The
label is part of the spec's bytes, so it changes the spec's `id`, and it
is part of `snapshot_id`: adding the field changed every spec's
`snapshot_id` once, including a spec labelled `Command` whose rendered
bytes did not change, so a consumer refits once (invariant 8). To escalate
with a warm resume, send those same state bytes,
`{facts}{input_label}:\n{input}` with the label read from the menu, as
`/v1/adjudicate`'s `input` (it wraps `input` in the same user turn); any
other bytes are a cold start, not an error. The old
`state.command` is a `400`, with no alias.

**15. A read never waits out a generation (2026-09-26).** Opinion reads
(`/v1/opinion`, `/v1/adjudicate` with `opinion: true`) and `/v1/probe`
queue apart from generation (`/v1/adjudicate`, `/v1/chat`) and spec
registration, and go first: one arriving while a generation runs is served
at that generation's next prefill chunk or decoded token. The generation's
output does not change, bit for bit; its `prefill_ms`/`decode_ms` and its
deadline include the reads it paused for. Each queue holds eight, so a
backlog of generations does not refuse a read, and a steady stream of
reads can postpone a waiting generation until its deadline. Generations
and registrations keep their arrival order.

**16. A chat is its token ids; continue it only from a checkpoint
(2026-09-26).** `POST /v1/chat` starts a chat or continues one `from` a
checkpoint id it returned, appending user and tool turns; an assistant
turn exists only as the daemon generated it, and a request carrying one is
a `400`. A turn returns `checkpoint_user` (after the appended turns,
announced before the assistant's turn is generated) and `checkpoint`
(after the assistant's turn; `null` when the turn hit `max_tokens`). An id
is the lowercase hex sha256 of the checkpoint's token ids as little-endian
u32s, and the same ids always name the same state on one daemon and
backend. Checkpoints live in memory under a byte budget
(`--chat-checkpoint-budget-mib`), least recently used first; an evicted or
unknown id is a `404`, never a rebuild from text, so a consumer that gets
one starts the chat again. A daemon restart forgets every checkpoint.
`thinking`/`content` split `text` at the reasoning region; tool calls stay
raw in `content`.

**17. A tail read is its own instrument (2026-09-26).** `/v1/opinion`
with `context: {"checkpoint": id}` asks the spec's questions after the
chat instead of after the spec's system turn: the spec's instructions and
schema move into the read's own user turn. Invariants 8–11 and 14 hold for
it, with two differences a consumer must respect. Its numbers answer a
different prompt from a read without `context`, so thresholds fitted on
one do not carry to the other (invariant 8); and `use_cache: false` reads
the same bytes cold, which is a different computation, not a reference
(resumed and cold disagree by |Δ logprob| p50 0.21 nats, max 1.8, top
option on 2 of 24 questions; numbers below). A repeated tail read from the
same checkpoint is identical, and so is one that starts from a tail
prefix the daemon filled in the background. The response echoes
`context`, and
`rendered_sha256` covers the whole chat. A spec with `tools` cannot read a
tail (`400`), nor can a state that begins with a newline.

## Operational numbers (measured, dated — re-measure before designing on them)

The encoder-head numbers that stood here measured the retired severity
head and the cascade (history in git). The opinion engine's numbers are
per spec, because prompt length and field count set the cost:

- **`/v1/opinion`** (2026-09-22, lfm2d-system1 on ROCm, the shell spec
  `command-verdict-enum-v1` over 236 rows): p50 **888 ms** on a
  described-cache miss, **46 ms** on a hit (bit-identical to the miss);
  escalation to `/v1/adjudicate` from the described state 240–550 ms.
  Measure hits by an immediate repeat: a sweep evicts the LRU.
- **`/embed`** (2026-09-24, lfm2d-system1, LFM2.5-Embedding-350M on ROCm):
  p50 **15.6 ms** for one input, ~120 ms for a batch of five; the embedder
  has its own worker queue beside the opinion engine's thread.
- **Reads during generation** (2026-09-26, ROCm, `email-triage-v1`, a
  ~1.4 s generation, n=40 per binary): a read fired mid-generation waited
  p50 9 ms, max 175 ms in the queue, where before it waited out the
  generation (p50 573–804 ms). `benchmarks/lfm25/results/2026-09-26-interleave.json`.
- **Tail reads** (2026-09-26, ROCm, `email-triage-v2`, chats of 860–1150
  tokens, n=12 reads): a resumed tail read p50 **1.10 s** on a described
  miss, a cold one 2.09 s; two reads forked from a streaming turn's
  `checkpoint_user` answered 6/6 before the turn finished.
  `benchmarks/lfm25/results/2026-09-26-chat-tail.json`.
- **Tail prefixes** (2026-09-26, same stack, six chats, n=12 per class):
  a described-cache miss that starts after a background-filled tail
  prefix took p50 653 ms, against 1274 ms forwarding the spec's head
  itself. `benchmarks/lfm25/results/2026-09-26-tail-prefill.json`.
- The encoder heads serve through ONE serial inference worker, so
  concurrent callers queue rather than parallelise; a caller that fans out
  backlogs everyone behind it.
