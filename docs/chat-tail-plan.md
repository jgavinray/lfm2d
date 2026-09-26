# Plan: opinion reads on a live chat's tail

Status: in progress, 2026-09-26. Delete this doc when the work lands; its
lasting parts move into `docs/lfm25-adjudicator.md` and `docs/integration.md`.

Landed: piece 1 (interleaving worker, `docs/integration.md` invariant 15),
piece 2 (the renderer, `lfm2d/src/chat.rs`)
and pieces 3–5 (`POST /v1/chat`, checkpoints, tail reads; invariants 16–17,
`docs/lfm25-adjudicator.md` "Chat sessions and tail reads").

## What we are building

A side-by-side demo. On one side the user chats with LFM2.5-8B-A1B: a plain
scenario system prompt, open reasoning, and thinking **preserved** in
history. On the other side, opinion (System 1) reads fork the chat's
current tail and answer a spec's questions fast while the generative side
is still reasoning.

Amy, 2026-09-26: "I believe the priming works, so we don't need to
relitigate it." On performance: "prefill should be async if we can and it
should be fine for it to take a while. ideally we're using hot caches
wherever possible, use the memory, we have plenty to work with and find
what works. we separate the system1 specifically so that it can be quick
while the system2 ponders."

## Pieces

Each piece has a name so commits, tests and conversation share vocabulary.

1. **Interleaving worker.** Today `Handle::spawn` runs one thread that
   takes each job to completion. Instead, long jobs yield at the points
   where they already call `check()` (prefill chunks, decode tokens), and
   opinion reads jump ahead of generation. What must not change: every
   job's output is bit-identical with and without interleaving, and
   caches are never published from a half-finished job.
2. **Chat renderer.** Renders the multi-turn chat structure the model's template
   defines: system, user, assistant (with a `thinking` part and
   `tool_calls`), and tool turns, with `preserve_thinking` on. The
   daemon renders all of it; content still refuses control tokens. It is
   verified byte-for-byte against the GGUF's own Jinja using fixtures a
   Python script generates (the `tests/reference/dump_*.py` pattern).
3. **Sessions and checkpoints.** A chat session keeps the token ids it
   generated (never a re-tokenization of its own text) and a model state
   at every turn boundary. A **checkpoint** is content-addressed: sha256
   of its token ids. The key one is *end of the last user turn*: it is a
   prefix of everything the assistant generates next, so reads can fork
   it while the assistant is still reasoning.
4. **`POST /v1/chat`.** Appends a user turn to a checkpoint (or starts a
   session from a scenario system prompt), generates the assistant turn
   with open reasoning, streams tokens, and returns the new checkpoint
   ids.
5. **Tail reads.** `OpinionRequest.context` (reserved, `null`-only today)
   names a checkpoint. The read renders a user turn carrying the spec's
   instructions and schema, then the state, then the assistant opening
   with the closed `<think>\n\n</think>\n` region, and describes and scores
   the same way as today. `rendered_sha256` and the read's identity cover
   the checkpoint.
6. **Background tail prefill and memory-sized caches.** When a checkpoint
   appears, the worker prefills *checkpoint + each active spec's
   instruction block* at low priority. Caches are evicted by bytes, not
   by entry count; zorak has the memory for it.
7. **Probe.** Read `{"`'s rank at the report slot on real chat tails: do
   the closed-region bytes that were measured on a single user turn
   still sit on the model's manifold after a reasoning-heavy history?
8. **Demo page.** Chat on the left, System 1 reads of the tail on the
   right.

## In parallel: batched decode in candle

In our fork (`~/src/wt/candle-batch`, branched from the pinned
`lfm25-trace` @ `dda984e0`): decode B independent sequences, each with its
own `State`, in one forward. It helps lfm2d whatever happens to the demo
idea (several reasoning branches from one fork, several specs or candidate
actions scored at once). Batch size changes which ROCm kernel runs (MMVQ
at `b_size <= 8`), so batched numbers are a different instrument from
batch-1 numbers, and each has to be measured on its own terms.

## Order

1 and 2 in parallel, then 3–5, then 6, 7 and 8. Candle batching runs
alongside the whole time.
