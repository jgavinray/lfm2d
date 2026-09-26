//! `lfm2d` — the LFM2.5 encoder and causal adjudicator daemon.
//!
//! # Why this exists
//!
//! candle's `from_mmaped_safetensors` copies every tensor into private
//! anonymous RSS on load (see the crate's `candle-mmap-loads-are-copies`
//! memory note) — there is no cross-process sharing, so N processes each
//! loading a 350M checkpoint cost N × ~1.4 GiB, not one shared mapping. A
//! single forward pass also saturates ~13.6 of this box's cores already
//! (measured), so per-request concurrency buys nothing — serializing
//! inference behind one worker loses no real throughput. And `kaibo`
//! cannot link `candle-core` at all (it hard-wires `tokenizers`+`onig`,
//! breaking kaibo's musl static-build invariant). One niced daemon process,
//! loading each configured head exactly once, serving every consumer over
//! HTTP, is the fix for all three.
//!
//! # Architecture
//!
//! One inference worker thread ([`worker::WorkerHandle`]) owns the encoder
//! models. The optional [`adjudicator`] has a separate bounded queue and worker,
//! with immutable hot-prefix state and isolated per-evaluation branches.
//! axum handlers send commands to their worker and await a `oneshot` reply.
//! Encoder requests use [`worker::WorkerCommand`] on an unbounded channel;
//! adjudication uses its own bounded channel. Each worker processes requests
//! serially.
//!
//! The [`worker::InferenceEngine`] trait decouples the channel/handler
//! plumbing from real candle models: [`engine_real::RealEngine`] is what
//! `main.rs` spawns in production, but router tests build the same
//! [`server::build_router`] over [`engine_stub::StubEngine`] instead — no
//! weights loaded, no candle in the test binary's hot path, sub-millisecond
//! tests for every HTTP-level concern (status codes, JSON shape, error
//! mapping).
//!
//! # Two response conventions on purpose
//!
//! Every model load computes a weight hash (sha256 over its
//! `model.safetensors`, hex) and every inference response is required to
//! carry `{model_id, weight_hash}` — an audit requirement from the kaish
//! approval-chain rulings. But `/embed` is specified as TEI-compatible-ish: a
//! bare `[[f32,...]]` array, matching what existing TEI clients already
//! parse. Putting `model_id`/
//! `weight_hash` IN that body would break TEI wire compatibility for no
//! reason; leaving them out entirely would violate the audit requirement.
//! This crate's resolution: `/embed` and `/v1/spans` carry `X-Model-Id` and
//! `X-Model-Weight-Hash` response headers (inspectable, satisfies "every
//! response carries," never touches the JSON body), while `/v1/route` —
//! "our full contract," not TEI-compat —
//! carry the same pair directly in the JSON body, per the API spec's own
//! text. See `server::attach_audit_headers` and each handler's doc comment.
//!
//! # `/v1/spans` never returns the matched text
//!
//! `POST /v1/spans`/`POST /v1/spans/credentials` (backed by
//! `--token-classifier-dir`, repeatable — the PII detector is the first
//! consumer, but nothing here is special-cased to it) answer with byte
//! OFFSETS and an entity TYPE only, never the substring that matched. The
//! caller already has the full text it just sent; echoing a credential
//! back would just create a second copy of it in every downstream log this
//! response passes through. See `types::SpanResult`'s doc comment. Byte
//! offsets (not codepoint/char offsets) — see the same doc comment for what
//! that means for non-Rust callers.
//!
//! Telemetry for this endpoint is held to a stricter bar than the rest of
//! this API for the same reason: request text carries live credentials by
//! definition, so no span, log, or metric anywhere in this crate ever
//! records input text, span offsets, or a matched substring. The one
//! opt-in exception, `--log-input-hash` (default off), attaches a HASH of
//! the input text to that call's trace/log span only — never a metric
//! label (unbounded cardinality). See `worker.rs`'s module docs.

pub mod adjudicator;
pub mod chat;
pub mod chat_session;
pub mod chunk_sweep;
pub mod config;
pub mod constrain;
pub mod device;
pub mod examine;
pub mod expert_map;
pub mod engine_real;
pub mod engine_stub;
pub mod hash;
pub mod opinion;
pub mod opinion_api;
pub mod probe;
pub mod probe_api;
pub mod server;
pub mod shutdown;
pub mod telemetry;
pub mod tokenize_api;
pub mod types;
pub mod worker;
