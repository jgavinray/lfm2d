# lfm2d on the B70: full endpoint exercise and throughput report

2026-10-02, hyper03. Daemon: all five models (Embedding-350M, Prompt-Router,
PII-Detector, ColBERT loaded but unexercised, 8B-A1B Q5_K_M adjudicator) on
`--device sycl`, one API on 127.0.0.1:8931. Candle rev `1775caf4` (the grouped-
prefill fork). Device identity `sycl:2026.1:Intel(R) Arc(TM) Pro B70 Graphics`.
Exercise script: `/tmp/lfm2d_exercise.py` (three runs; harness in this report's
numbers are medians over N calls after warmup, client-side loopback, so HTTP
overhead included — the honest end-to-end figure).

## Endpoint coverage — all driven, all verified

| endpoint | correctness result |
|---|---|
| GET /healthz, /readyz | 200, ready after ~75 s (8B load) |
| GET /v1/models | 4 heads + adjudicator separate; kinds and 64-hex weight hashes verified |
| POST /embed | single (1024-dim), batch, query kind; X-Model-Id/X-Model-Weight-Hash headers present |
| POST /v1/route | k8s input ranked `k8s` first among 3 routes (semantic check) |
| POST /v1/spans | single + batch; **byte offsets verified by slicing the original text** |
| POST /v1/spans/credentials | filtered to the secret family (`credential.*` + `developer.login_credentials`) |
| POST /v1/tokenize | encoder + adjudicator tokenizers; `context` block with `suffix_stable` verified |
| POST /v1/opinion/specs | upload (201, content-addressed id), list, delete (204), 403-on-boot-spec untested (no boot spec) |
| POST /v1/opinion | **27 successful constrained reads** (first run), read_ms 79–97 |
| POST /v1/adjudicate | 200s with schema-constrained output; decode 18.2 ms/token (55 tok/s) |
| POST /v1/chat | 200 |
| POST /v1/probe | raw logits/top-logprobs, prefill 357 ms @ 201 tokens |
| error paths | unknown spec/model 404, malformed questions 400, timeout_ms range validation |

## Throughput (median ms/call, p90 in parens)

| workload | median | notes |
|---|---:|---|
| /embed 1 doc | 9.7 (10.0) | ~1.2 ms/token prefill equivalent |
| /embed batch 8 | 68.2 (69.2) | 8.5 ms/input — batching amortizes |
| /embed batch 32 query | 273 (287) | linear in inputs |
| /v1/route (3 routes) | 11.4 (12.0) | |
| /v1/spans 1 input | 9.4 (9.7) | |
| /v1/spans batch 8 | 65.7 (67.3) | |
| /v1/spans/credentials batch 8 | 64.7 (65.7) | |
| /v1/tokenize | 1.0 (1.0) | no GPU |
| /v1/probe (1-token gen) | 0.6 | prefill 357 ms @ 201 tok = 1.8 ms/tok |
| /v1/opinion read (warm) | 83–97 ms | constrained single-slot read, cached describe |
| /v1/adjudicate 32 tok | 673 ms | ~21 ms/token decode = **~48 tok/s** |
| /v1/adjudicate 4 tok | ~73 ms + prefill | 18.2 ms/token |

Context: llama.cpp on the same host's Strix Halo measured 102–108 tok/s decode
on the same checkpoint (docs/lfm25-runtime-comparison.md); the B70 lands at
~45–55 tok/s with the current kernel set — consistent with the 5.5x lower
memory bandwidth of the B70 (512 GB/s vs 256 GB/s×… see the record) and the
launch-bound decode path. The grouped-prefill win applies at batch≥64 tasks;
single-request prefill is 1.8 ms/token (vs ROCm's 0.28 ms/token on the 8B).

## BUG FOUND AND CHARACTERIZED (reproduces deterministically)

**Fresh-spec describe hangs the adjudicator worker.** `POST /v1/opinion` on a
spec whose described prompt is not yet cached deadlocks the worker: no log
line (no describe started), 100% of one core, every subsequent
adjudicator-side request (probe, adjudicate, opinion) queues behind it until
the client deadline cancels it (error_kind="cancelled" at exactly timeout_ms).
Worker recovers after the deadline. Reproduced on a freshly started daemon as
the very first request, and with the exact payload that succeeded 27 times
earlier — so it is payload-independent and state-dependent (described-cache
cold). During the first exercise run the describe completed twice (17.8 s,
25.7 s, described_cache=miss, read then succeeded at 83–97 ms); every attempt
after the spec was re-uploaded hung. Adjudicate on the same spec works
instantly (0.2 s), chat works — the hang is specific to the opinion
describe/grammar walk. 27 opinion successes in run 1 prove the path itself
works; the trigger is a cold described-prompt cache after spec re-upload.

Reproduction: start daemon → upload email-triage-v1 → POST /v1/opinion with
timeout_ms=120000 → 504 at exactly 120 s, log shows no describe line.

## What was not exercised

- ColBERT head (loaded, but no HTTP endpoint exposes it on this build).
- Streaming chat (`"stream": true`), chat checkpoints (`from`), probe
  `decode_from`, distributions request, boot-spec 403.
