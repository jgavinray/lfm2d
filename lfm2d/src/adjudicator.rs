//! Causal adjudication with one resident prompt and isolated evaluation branches.
//!
//! The worker owns the model. Prefix state is immutable and tensor-shared; each
//! request appends into replacement storage. Outputs are model reports, never
//! executed actions. The protocol deliberately exposes truncation and errors.
use crate::{
    config::Cli,
    hash::{sha256_hex_bytes, sha256_hex_file},
    types::{ModelInfo, ModelKind},
    worker::WorkerExit,
};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use candle_core::{Tensor, quantized::gguf_file};
use candle_nn::sampling::GreedySampler;
use candle_transformers::models::quantized_lfm2_moe::{Model, State as ModelState};
use serde::{Deserialize, Serialize};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::oneshot;

pub(crate) const CHUNK: usize = 128;
/// The candle fork revision this binary was built against, read from
/// `Cargo.lock` by `build.rs`. Part of every `snapshot_id`: the kernels are
/// the fork's, so a new revision can move every number.
pub const CANDLE_REV: &str = env!("LFM2D_CANDLE_REV");
/// `<|im_end|>`, which ends a turn. `Checkpoint::load` refuses a tokenizer that
/// puts it anywhere else.
const EOS: u32 = 124900;
const MAX_NEW: usize = 2048;
/// `/v1/adjudicate`'s input cap: `/v1/opinion`'s 64 KiB state plus
/// headroom for the rendered `{input_label}:\n`, so an escalation of a
/// full-size opinion is never refused (pinned by
/// `opinion_api::tests::a_full_size_state_still_fits_adjudicate_once_rendered`).
pub(crate) const MAX_INPUT_BYTES: usize = 65536 + 1024;
/// The longest `input_label` a spec may carry. Bounded so a full-size
/// opinion state, rendered under its label, still fits `MAX_INPUT_BYTES`
/// when a consumer escalates it to `/v1/adjudicate`.
pub const MAX_INPUT_LABEL_BYTES: usize = 64;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromptSpec {
    /// What the user turn calls its input: an opinion state renders as
    /// `{facts}{input_label}:\n{input}` ([`crate::opinion_api::OpinionState`]).
    /// Required, and part of the spec's bytes, so its id and its
    /// `snapshot_id` cover it. The spec owns the label because the spec's
    /// system prompt is what tells the model where the input is; a label the
    /// system prompt never names would be one the model was never told
    /// about. One line, no colon, no model control tokens — refused at load
    /// ([`PromptSpec::render_prefix`]).
    pub input_label: String,
    pub system: String,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    #[serde(default)]
    pub output_schema: Option<serde_json::Value>,
    /// Whether the assistant's turn opens with a finished reasoning region.
    /// Policy, so it lives with the prompt rather than in the renderer.
    #[serde(default)]
    pub reasoning: Reasoning,
    /// The System-1 question asked at the verdict slot by `opinion` requests.
    /// Not part of the prefix: it is rendered into each request's suffix.
    #[serde(default)]
    pub opinion: Option<crate::opinion::OpinionSpec>,
}

/// Whether the assistant's turn opens with a *completed* reasoning region.
///
/// The checkpoint's chat template never emits one: `<think>` is a token the
/// model writes, and after `<|im_start|>assistant\n` it writes it at p=1.00.
/// Under an `output_schema` the grammar's first legal byte is the object's, so
/// the model was masked off its own manifold at step 0 — the forced `{"` scored
/// logprob -17.8 to -21.8, and every later token was conditioned on a prefix
/// the model considers impossible.
///
/// `Closed` prefills the region already finished, so generation begins where the
/// report begins. The bytes are `<think>\n\n</think>\n` and they were measured,
/// not derived: the template's own dialect for a completed region
/// (`"<think>" + thinking + "</think>"`, no surrounding newlines) leaves the
/// model wanting a newline at p=0.97, and the grammar forces `{"` from rank 5.
/// Five candidates over three inputs on ROCm, `{"`'s standing at the slot:
///
/// | after `<|im_start|>assistant\n` | `{"` |
/// |---|---|
/// | nothing (v1)          | logprob -17.8 to -21.8, rank 6 or worse |
/// | `<think></think>`     | rank 5 |
/// | `<think></think>\n`   | rank 2-3 |
/// | `<think>\n</think>\n` | rank 2 |
/// | `<think>\n\n</think>\n` | **rank 1, p 0.42-0.71** |
///
/// A second round varied one newline at a time (one more inside, one more
/// after, none after) and every neighbour was worse, so this is a local
/// optimum rather than the best of an arbitrary five.
/// Run record: `lfm25-think-prefill-2026-09-18` (author's private notes).
///
/// `Open` leaves the turn as the template opens it and the model reasons. That
/// is today's free-generation behaviour and is honest *without* a schema.
/// Combined with one it is refused: the grammar would have to admit a free-text
/// region it cannot bound and then start the object after `</think>`, which is
/// not built. See `crate::constrain`'s first ruling, and the measurement that
/// LFM2.5's reasoning argues severity *down*.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Reasoning {
    #[default]
    Closed,
    Open,
}
impl Reasoning {
    /// What the renderer writes after `<|im_start|>assistant\n`.
    fn opening(self) -> &'static str {
        match self {
            Reasoning::Closed => "<think>\n\n</think>\n",
            Reasoning::Open => "",
        }
    }
}
/// Prompt content carries no model control tokens; the renderer supplies them.
pub fn validate_text(s: &str) -> Result<(), String> {
    if s.trim().is_empty() {
        return Err("input must not be empty".into());
    }
    if s.contains("<|") || s.contains("<think>") || s.contains("</think>") {
        return Err("literal model control tokens are not allowed in prompt content".into());
    }
    Ok(())
}
impl PromptSpec {
    /// The checkpoint's single-system/single-user chat-template subset. Tool
    /// schemas are part of the frozen system message, as in the GGUF template.
    pub fn render_prefix(&self) -> Result<String, String> {
        self.validate_input_label()?;
        validate_text(&self.system)?;
        if self.reasoning == Reasoning::Open && self.output_schema.is_some() {
            return Err(
                "reasoning \"open\" under an output_schema is not built: the grammar would have \
                 to admit an unbounded reasoning region and start the object after </think>"
                    .into(),
            );
        }
        let mut system = self.system.clone();
        if let Some(schema) = &self.output_schema {
            if !self.tools.is_empty() {
                return Err("choose output_schema or tools, not both".into());
            }
            validate_schema(schema)?;
            let schema = render_schema(schema)?;
            validate_text(&schema)?;
            system.push_str("\nReturn exactly one JSON object matching this schema: ");
            system.push_str(&schema);
        }
        if !self.tools.is_empty() {
            let mut rendered = Vec::new();
            for tool in &self.tools {
                if tool["type"] != "function" || !tool["function"]["name"].is_string() {
                    return Err("each tool must be a named function schema".into());
                }
                let json = serde_json::to_string(tool).map_err(|e| e.to_string())?;
                validate_text(&json)?;
                rendered.push(json);
            }
            system.push_str("\nList of tools: [");
            system.push_str(&rendered.join(", "));
            system.push(']');
        }
        Ok(format!(
            "<|startoftext|><|im_start|>system\n{system}<|im_end|>\n"
        ))
    }

    /// The label must render as exactly one `label:` line: a newline or a
    /// colon inside it would make the rendered state say something other
    /// than "the input follows", and a control token would forge a turn.
    fn validate_input_label(&self) -> Result<(), String> {
        let label = &self.input_label;
        if label.trim().is_empty() {
            return Err("input_label must not be empty".into());
        }
        if label.contains('\n') || label.contains('\r') {
            return Err("input_label must be one line".into());
        }
        if label.contains(':') {
            return Err("input_label must not contain a colon; the renderer writes the one after it".into());
        }
        if label.contains("<|") || label.contains("<think>") || label.contains("</think>") {
            return Err("input_label must not carry model control tokens".into());
        }
        if label.len() > MAX_INPUT_LABEL_BYTES {
            return Err(format!("input_label is {} bytes; at most {MAX_INPUT_LABEL_BYTES}", label.len()));
        }
        Ok(())
    }

    /// The single user turn and the opening of the assistant's, appended to
    /// [`PromptSpec::render_prefix`]. One definition, so anything examining a
    /// prompt renders exactly what the daemon runs.
    pub fn render_user_turn(&self, input: &str) -> String {
        let opening = self.reasoning.opening();
        format!("<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{opening}")
    }

    /// The rendered template is different bytes per [`Reasoning`] mode, so the
    /// mode is part of the version a consumer reads beside every
    /// `/v1/adjudicate` response, and part of the prefix snapshot's identity. v1 was `closed`'s bytes
    /// without the reasoning region.
    pub fn template_version(&self) -> &'static str {
        match self.reasoning {
            Reasoning::Closed => "lfm25-single-user-v2-closed",
            Reasoning::Open => "lfm25-single-user-v2-open",
        }
    }

    /// The user turn continued by text the assistant has already written, for
    /// the examination tools that stand at an answer slot. A prefill carrying
    /// reasoning delimiters would render a second region beside the one a
    /// `closed` spec has already written — `<think>\n\n</think>\n<think>...`
    /// — which is not a prompt anyone meant to examine, so it is refused
    /// rather than quietly rendered. Both delimiters are refused: a bare
    /// `</think>` closes a region that is already closed.
    ///
    /// The remedy is exact text, not `reasoning: "open"`, which is refused
    /// alongside an `output_schema` and so is unavailable for exactly the
    /// schema-bearing prompts these tools mostly examine.
    pub fn render_user_turn_with_prefill(
        &self,
        input: &str,
        prefill: &str,
    ) -> Result<String, String> {
        let reasoning_delimiter = prefill.contains("<think>") || prefill.contains("</think>");
        if self.reasoning == Reasoning::Closed && reasoning_delimiter {
            return Err(
                "this prompt spec already closes a reasoning region, so a prefill cannot carry \
                 reasoning delimiters: supply the whole prompt as exact text instead"
                    .into(),
            );
        }
        Ok(format!("{}{prefill}", self.render_user_turn(input)))
    }
}



#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdjudicateRequest {
    pub input: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Explicit cold baseline for parity/latency measurements, not a fallback.
    #[serde(default = "yes")]
    pub use_cache: bool,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    /// Per-generated-token distributions (top-k logprobs, raw named-set
    /// mass). Omitted entirely (the default) => `AdjudicateResponse` is
    /// today's shape, byte-identical; see `docs/field-requests.md` (git f9ca081)
    /// decision 5 and [`crate::types::DistributionRequest`].
    #[serde(default)]
    pub distributions: Option<crate::types::DistributionRequest>,
    /// Read the spec's `opinion` question instead of generating: one prefill,
    /// every option scored, nothing decoded. See [`crate::opinion`].
    #[serde(default)]
    pub opinion: bool,
    /// Which loaded spec to use: an id (`POST /v1/opinion/specs`'s content
    /// hash) or a boot-time spec's file-stem name. Required: no spec is a
    /// default, so a request without one is a 400 that says where the menu
    /// is ([`AdjudicateRequest::spec_key`]). An `Option` on the wire only so
    /// that refusal can say that, instead of serde's bare "missing field".
    /// An escalation from `POST /v1/opinion` resumes from that spec's
    /// described cache only when `spec` names the SAME spec the opinion read
    /// used — see `docs/system1-split-plan.md` (git f9ca081) "Runtime spec registration".
    #[serde(default)]
    pub spec: Option<String>,
}
fn default_max_tokens() -> usize {
    2048
}
fn default_timeout() -> u64 {
    30000
}
fn yes() -> bool {
    true
}
impl AdjudicateRequest {
    /// The spec this request names. Missing or empty is refused with where
    /// to look, never answered with some spec the caller did not name.
    pub fn spec_key(&self) -> Result<&str, String> {
        match self.spec.as_deref() {
            Some(key) if !key.is_empty() => Ok(key),
            _ => Err("spec is required: name a loaded spec by its id or boot-time name, \
                      as listed by GET /v1/opinion/specs"
                .into()),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        self.spec_key()?;
        validate_text(&self.input)?;
        if self.input.len() > MAX_INPUT_BYTES {
            return Err(format!("input exceeds {MAX_INPUT_BYTES} bytes"));
        }
        if self.max_tokens == 0 || self.max_tokens > MAX_NEW {
            return Err("max_tokens must be 1..=2048".into());
        }
        if self.timeout_ms == 0 || self.timeout_ms > 120000 {
            return Err("timeout_ms must be 1..=120000".into());
        }
        if let Some(d) = &self.distributions {
            d.validate()?;
        }
        if self.opinion && self.distributions.is_some() {
            return Err("an opinion read decodes nothing, so distributions do not apply".into());
        }
        Ok(())
    }
}
/// `GET /v1/adjudicator`: the loaded checkpoint and how it runs — nothing
/// that depends on a spec. Every spec's own identity (`snapshot_id`) is on
/// its `GET /v1/opinion/specs` menu entry, and every `/v1/adjudicate`
/// response carries the full [`PrefixInfo`] of the spec it named.
#[derive(Clone, Debug, Serialize)]
pub struct AdjudicatorInfo {
    pub model_id: String,
    pub weight_hash: String,
    pub tokenizer_hash: String,
    pub context_limit: usize,
    pub backend: String,
    /// `ExecutionDevice::identity`: the target the kernels were built for.
    pub device: String,
    /// [`CANDLE_REV`].
    pub candle_rev: String,
    pub dtype: String,
    pub sampling: String,
    pub weight_dtypes: Vec<String>,
}
impl From<&PrefixInfo> for AdjudicatorInfo {
    /// The spec-independent half of a spec's identity. Every loaded spec
    /// shares these, since they come from the one checkpoint.
    fn from(p: &PrefixInfo) -> Self {
        Self {
            model_id: p.model_id.clone(),
            weight_hash: p.weight_hash.clone(),
            tokenizer_hash: p.tokenizer_hash.clone(),
            context_limit: p.context_limit,
            backend: p.backend.clone(),
            device: p.device.clone(),
            candle_rev: p.candle_rev.clone(),
            dtype: p.dtype.clone(),
            sampling: p.sampling.clone(),
            weight_dtypes: p.weight_dtypes.clone(),
        }
    }
}
/// One loaded spec's identity: the checkpoint's, plus what the spec's own
/// prefix makes of it. Flattened into every `/v1/adjudicate` response.
#[derive(Clone, Debug, Serialize)]
pub struct PrefixInfo {
    pub model_id: String,
    pub weight_hash: String,
    pub tokenizer_hash: String,
    pub template_version: String,
    pub snapshot_id: String,
    pub prefix_tokens: usize,
    /// Complete-input checkpoints retained alongside the fixed prefix.
    pub input_cache_capacity: usize,
    pub context_limit: usize,
    pub backend: String,
    pub device: String,
    pub candle_rev: String,
    pub dtype: String,
    pub sampling: String,
    pub weight_dtypes: Vec<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct AdjudicateResponse {
    #[serde(flatten)]
    pub prefix: PrefixInfo,
    /// Includes reasoning and native tool-call delimiters. No tool is executed.
    pub output: String,
    pub report: Option<serde_json::Value>,
    pub report_error: Option<String>,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub queue_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    /// One entry per generated token (including a trailing eos, if any),
    /// same count as `completion_tokens`. Present only when the request set
    /// `distributions` — `#[serde(skip_serializing_if)]` keeps the field
    /// entirely absent from the JSON body otherwise, so a request that
    /// never asked for it gets byte-identical output to before this field
    /// existed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distributions: Option<Vec<crate::types::StepDistribution>>,
    /// Present only on an `opinion` request. `output` is then empty,
    /// `finish_reason` is `opinion`, `prompt_tokens` counts the prefilled
    /// shared prefix (the scored continuations are in `opinion.scored_tokens`),
    /// `completion_tokens` is 0 and `decode_ms` is the time spent scoring the
    /// options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opinion: Option<crate::opinion::OpinionRead>,
    /// Present only when the generation resumed from a described state that
    /// `/v1/opinion` left for these exact prompt bytes: how many of
    /// `completion_tokens` were taken from it rather than decoded now. The
    /// report is the one a fresh generation writes (greedy under the grammar
    /// is a pure function of the prompt); only the time differs. Cold
    /// requests and requests asking for `distributions` never resume.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resumed_tokens: Option<usize>,
}

#[derive(Debug)]
pub enum Failure {
    BadRequest(String),
    /// A `spec` names nothing loaded — boot or uploaded, never (this
    /// checkpoint or an evicted upload). The client's cue: upload it (or
    /// re-upload) via `POST /v1/opinion/specs` and retry. Distinct from
    /// `BadRequest` so a caller can tell "malformed request" from "right
    /// shape, wrong/missing spec" without parsing the message.
    NotFound(String),
    /// A spec's bytes parsed but could not be served: the load-time
    /// refusals `LoadedSpec::load` runs (grammar compile, tokenization,
    /// opinion block split) — the same checks a boot spec passes before it
    /// ever reaches a request, now surfaced to the uploader instead of
    /// exiting the process.
    Unprocessable(String),
    /// A boot-time spec cannot be deleted — `DELETE /v1/opinion/specs/{id}`
    /// against an `--opinion-spec`.
    Forbidden(String),
    Internal(String),
    Cancelled,
    Deadline,
}
impl From<candle_core::Error> for Failure {
    fn from(e: candle_core::Error) -> Self {
        Self::Internal(e.to_string())
    }
}
impl Failure {
    /// The status and the `{"error": {"type", "message"}}` body every
    /// refusal answers with; a streaming chat sends the same body in its
    /// `error` event.
    fn parts(self) -> (StatusCode, serde_json::Value) {
        let (status, kind, msg) = match self {
            Self::BadRequest(s) => (StatusCode::BAD_REQUEST, "bad_request", s),
            Self::NotFound(s) => (StatusCode::NOT_FOUND, "not_found", s),
            Self::Unprocessable(s) => (StatusCode::UNPROCESSABLE_ENTITY, "unprocessable", s),
            Self::Forbidden(s) => (StatusCode::FORBIDDEN, "forbidden", s),
            Self::Internal(s) => (StatusCode::INTERNAL_SERVER_ERROR, "internal", s),
            Self::Cancelled => (
                StatusCode::REQUEST_TIMEOUT,
                "cancelled",
                "evaluation cancelled".into(),
            ),
            Self::Deadline => (
                StatusCode::GATEWAY_TIMEOUT,
                "deadline",
                "evaluation deadline exceeded".into(),
            ),
        };
        (status, serde_json::json!({"error":{"type":kind,"message":msg}}))
    }
}
impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        let (status, body) = self.parts();
        (status, Json(body)).into_response()
    }
}

/// A long job's view of the worker at the points where it may pause.
///
/// The worker runs one job at a time on one thread, but it does not have to
/// run a long one to completion before anything else: a generative
/// adjudication (seconds of decode) calls [`YieldPoint::pause`] before every
/// prefill chunk and every decoded token, and the worker serves the pending
/// jobs of a HIGHER priority class there, to completion, on the same
/// generator, before the paused job goes on (see `Priority` and `Pause`).
/// So an opinion read waits at most one chunk or one token, not a whole
/// generation.
///
/// Why nested at the pause and not resumable step machines: a paused job's
/// progress is already plain owned values (a `ModelState` it cloned out of
/// the caches, its logits, its sampler), so the only thing in the way of
/// running another job at its pause is the `&mut` generator borrow — and
/// `pause` takes that borrow back as an argument. The price is a rule the
/// generator must keep: hold NO borrow into its own caches or spec store
/// across a `pause`, and publish to a cache only after the last pause, from
/// a complete result, re-resolving the entry it publishes into
/// (`Adjudicator::generate` shows the pattern). Reads served at a pause never
/// pause themselves, so they may hold their borrows as before.
///
/// Every `Fn() -> Result<(), Failure>` is a yield point that serves nothing:
/// `pause` is then exactly `check`. Direct callers (the real-model tests,
/// the examiner) keep passing a plain check closure.
pub trait YieldPoint<G: ?Sized> {
    /// The paused job's own cancellation and deadline, unchanged: `Cancelled`
    /// once its caller is gone or the daemon is stopping, `Deadline` once
    /// its deadline (from enqueue) passes.
    fn check(&self) -> Result<(), Failure>;
    /// Serve every pending job of a higher class on `generator`, with
    /// `check` before, between and after them: time spent serving counts
    /// against the paused job's deadline, and a paused job that is
    /// cancelled or out of time stops at the next boundary between them.
    fn pause(&self, generator: &mut G) -> Result<(), Failure>;
}
impl<G: ?Sized, F: Fn() -> Result<(), Failure>> YieldPoint<G> for F {
    fn check(&self) -> Result<(), Failure> {
        self()
    }
    fn pause(&self, _: &mut G) -> Result<(), Failure> {
        self()
    }
}

pub trait Generator: Send + 'static {
    /// Generative adjudication, the one job that pauses: see [`YieldPoint`]
    /// for what an implementation owes the jobs served at its pauses.
    fn generate(
        &mut self,
        request: &AdjudicateRequest,
        at: &dyn YieldPoint<Self>,
    ) -> Result<AdjudicateResponse, Failure>;
    /// Describe-then-read for `/v1/opinion`: `questions` were resolved
    /// against the spec's menu by the handler, in emission order. See
    /// [`crate::opinion_api`].
    fn opine(
        &mut self,
        request: &crate::opinion_api::OpinionRequest,
        questions: &[crate::opinion_api::ResolvedQuestion],
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<crate::opinion_api::OpinionResponse, Failure>;
    /// Register a spec by its exact uploaded bytes: `id` is already the
    /// content hash of those bytes (computed once, centrally, by the
    /// handler — see `Handle::register`), `prompt` is them parsed. Already
    /// loaded (boot or uploaded) is a no-op that still counts as use; a
    /// genuine miss loads it, possibly evicting the least recently used
    /// upload. See [`RegisterOutcome`].
    fn register(
        &mut self,
        id: String,
        prompt: PromptSpec,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<RegisterOutcome, Failure>;
    /// Unload an uploaded spec by id. Never fails: unknown and boot-spec
    /// are both outcomes, not errors — see [`UnregisterOutcome`].
    fn unregister(&mut self, id: &str) -> UnregisterOutcome;
    /// `POST /v1/probe`: raw inference over exact text, an instrument with
    /// no calibration contract — see [`crate::probe_api`].
    fn probe(
        &mut self,
        request: &crate::probe_api::ProbeRequest,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<crate::probe_api::ProbeResponse, Failure>;
    /// `POST /v1/chat`: append turns and generate the assistant's, pausing
    /// like [`Generator::generate`] (the same [`YieldPoint`] rule holds).
    /// `events` carries a streaming caller's progress. Only the real engine
    /// chats; a test double that never serves `/v1/chat` need not.
    fn chat(
        &mut self,
        request: &crate::chat_session::ChatRequest,
        events: &dyn Fn(crate::chat_session::ChatEvent),
        at: &dyn YieldPoint<Self>,
    ) -> Result<crate::chat_session::ChatResponse, Failure> {
        let _ = (request, events, at);
        Err(Failure::Internal("this generator does not chat".into()))
    }
}

/// `POST /v1/opinion/specs`'s result: the spec's own menu entry, whether
/// this call actually loaded it (`false` for an idempotent re-registration
/// of the same id — `200`, not `201`), the id of an upload evicted to make
/// room (if any), how long THIS call took, and the fresh full menu so the
/// handler's cached view (`Handle`'s) never drifts from the worker's.
pub struct RegisterOutcome {
    pub entry: crate::opinion_api::SpecMenuEntry,
    pub newly_loaded: bool,
    pub evicted: Option<String>,
    pub load_ms: f64,
    pub menu: Vec<crate::opinion_api::SpecMenuEntry>,
}

/// `DELETE /v1/opinion/specs/{id}`'s result.
pub enum UnregisterOutcome {
    /// The upload was removed. Carries its former menu entry (for the log
    /// line) and the fresh full menu.
    Deleted {
        entry: crate::opinion_api::SpecMenuEntry,
        menu: Vec<crate::opinion_api::SpecMenuEntry>,
    },
    /// `id` names a boot-time spec (`--opinion-spec`), which cannot be
    /// deleted at runtime — `403`.
    BootSpec,
    /// `id` names nothing loaded — `404`.
    NotFound,
}

// One exact-input checkpoint in addition to the fixed system prefix. This is
// reusable model computation, never a cached adjudication or mutable sampler.
struct PreparedPrompt {
    token_ids: Vec<u32>,
    state: ModelState,
    logits: Tensor,
}
struct PreparedEvaluation {
    state: ModelState,
    logits: Tensor,
    cached_tokens: usize,
}

/// Forward `ids` through `state` in `chunk_size`-sized pieces, returning
/// the last chunk's logits. Every path that must forward a suffix onto
/// SOME resident state goes through this one function, parameterized on
/// `chunk_size`, rather than its own copy of the loop:
/// [`PromptCache::prepare`]'s cache-miss branch and `/v1/probe`'s
/// warm-prefix resume (`crate::probe_api`) both call it with
/// [`CHUNK`] — they need to produce bit-identical numerics for the same
/// suffix on the same prefix (the chunk boundaries this loop picks are
/// themselves part of what makes cold and warm schedules disagree by
/// ~0.15 nats — `docs/lfm25-chunk-kernels.md`), so a second hand-written
/// copy is exactly the kind of drift that check exists to catch. `/v1/probe`
/// ALSO calls it with `chunk_size: 1` for its `decode_from` replay: a
/// single-element chunk is `model.forward(&[token], state)`, the EXACT
/// call `describe_then_read`'s decode loop makes per generated token —
/// reusing this function at that chunk size is what makes the replay the
/// same forward call, not a new one shaped to look similar.
fn forward_chunks(
    model: &Model,
    state: &mut ModelState,
    ids: &[u32],
    chunk_size: usize,
    check: &mut dyn FnMut() -> Result<(), Failure>,
) -> Result<Tensor, Failure> {
    let mut logits = None;
    for chunk in ids.chunks(chunk_size) {
        check()?;
        logits = Some(model.forward(chunk, state)?);
    }
    logits.ok_or_else(|| Failure::Internal("no suffix tokens".into()))
}

/// The canonical prefill schedule for rendered segments: each segment in
/// [`CHUNK`]-sized chunks from its OWN first token, never a chunk spanning
/// two segments. What makes a chat checkpoint's state a function of its ids
/// (`crate::chat_session`): starting a chat with turns A and B, and
/// continuing after A with B, forward exactly the same chunks. Empty
/// segments are refused (they would have no logits).
fn forward_segments(
    model: &Model,
    state: &mut ModelState,
    segments: &[&[u32]],
    between: &mut dyn FnMut() -> Result<(), Failure>,
) -> Result<Tensor, Failure> {
    let mut logits = None;
    for segment in segments {
        if segment.is_empty() {
            return Err(Failure::Internal("an empty segment has nothing to forward".into()));
        }
        logits = Some(forward_chunks(model, state, segment, CHUNK, between)?);
    }
    logits.ok_or_else(|| Failure::Internal("no segments to forward".into()))
}

/// What a chat checkpoint costs beyond its model state: its ids and an
/// allowance for its text (16 bytes a token covers the longest byte-level
/// token with room to spare).
fn checkpoint_extra_bytes(tokens: usize) -> usize {
    tokens * (std::mem::size_of::<u32>() + 16)
}

/// Resolve `/v1/probe`'s `decode_from` (a byte offset into the rendered
/// input) against `offsets` (that input's own per-token byte spans, in
/// order) into a token count: how many of the input's tokens fall in the
/// bulk-forward phase, the rest replaying the decode loop one at a time.
/// `None` in means "no split" (`Ok(None)`, today's only schedule). `Some(0)`
/// is always valid. Any other value must land exactly on some token's END
/// offset, or this refuses naming the nearest boundaries either side —
/// never rounds to the nearest one silently, since a caller replaying a
/// specific production schedule needs the exact split it asked for or a
/// loud refusal, not an approximation it might not notice.
///
/// Pure and offset-only (no tokenizer, no model), so it is unit-tested
/// directly against hand-built offset lists as well as the real
/// tokenizer's own offsets (`probe_decode_from_tests`, below).
fn resolve_decode_from(offsets: &[(usize, usize)], decode_from: Option<usize>) -> Result<Option<usize>, String> {
    let Some(offset) = decode_from else {
        return Ok(None);
    };
    if offset == 0 {
        return Ok(Some(0));
    }
    match offsets.iter().position(|&(_, end)| end == offset) {
        Some(i) => Ok(Some(i + 1)),
        None => {
            let lower = offsets.iter().map(|&(_, end)| end).filter(|&end| end <= offset).max().unwrap_or(0);
            let upper = offsets.iter().map(|&(_, end)| end).filter(|&end| end > offset).min();
            Err(format!(
                "decode_from={offset} is not a token boundary; nearest boundaries are {lower}{}",
                upper.map(|u| format!(" and {u}")).unwrap_or_default()
            ))
        }
    }
}

#[cfg(test)]
mod probe_decode_from_tests {
    use super::*;

    /// Three tokens covering "abc" "def" "ghi" — boundaries at 0, 3, 6, 9.
    fn offsets() -> Vec<(usize, usize)> {
        vec![(0, 3), (3, 6), (6, 9)]
    }

    #[test]
    fn none_keeps_everything_bulk() {
        assert_eq!(resolve_decode_from(&offsets(), None), Ok(None));
    }

    #[test]
    fn zero_is_always_a_valid_boundary_even_with_no_tokens() {
        assert_eq!(resolve_decode_from(&[], Some(0)), Ok(Some(0)));
        assert_eq!(resolve_decode_from(&offsets(), Some(0)), Ok(Some(0)));
    }

    #[test]
    fn a_token_end_offset_resolves_to_the_token_count_up_to_it() {
        assert_eq!(resolve_decode_from(&offsets(), Some(3)), Ok(Some(1)));
        assert_eq!(resolve_decode_from(&offsets(), Some(6)), Ok(Some(2)));
        assert_eq!(resolve_decode_from(&offsets(), Some(9)), Ok(Some(3)), "the last boundary: nothing left over");
    }

    #[test]
    fn a_mid_token_offset_is_refused_naming_both_neighbors() {
        let err = resolve_decode_from(&offsets(), Some(4)).unwrap_err();
        assert!(err.contains("3") && err.contains("6"), "{err}");
    }

    #[test]
    fn an_offset_past_the_last_token_is_refused_naming_only_the_lower_boundary() {
        let err = resolve_decode_from(&offsets(), Some(50)).unwrap_err();
        assert!(err.contains("nearest boundaries are 9"), "{err}");
        assert!(!err.contains(" and "), "no upper boundary exists past the end: {err}");
    }
}

/// The best warm-prefix resume candidate for `ids` among `prefixes`: the
/// LONGEST one that is a STRICT token-id prefix of `ids` (`ids.len() >
/// prefix.len()`, not `>=` — a prefix EQUAL to `ids` leaves nothing to
/// bulk-forward, so it is not a usable resume candidate here; the caller
/// falls back to a shorter match or cold rather than erroring "nothing to
/// forward" — F4, kaibo review 2026-09-23). Returns the WINNING index, not
/// a reference, so this stays pure and model-free: no `ModelState` to
/// clone, no tokenizer, just index arithmetic over token ids — unit-tested
/// directly below, and exercised for real by
/// [`SpecStore::resolve_best_prefix_mut`].
///
/// F2, same review: this used to be "whichever loaded spec is checked
/// first," which made the resumed spec (and therefore every number
/// downstream of it) depend on iteration order — boot-then-uploaded, and
/// WITHIN uploaded, LRU order, which live traffic changes continuously.
/// Two specs can genuinely both prefix the same input (one spec's prompt
/// extending another's, or simply sharing a common opening); the longest
/// one is the most resident computation reused and the only choice that
/// does not change answer-for-answer as an unrelated spec gets
/// registered, served, or evicted elsewhere.
fn best_prefix_match(ids: &[u32], prefixes: &[&[u32]]) -> Option<usize> {
    prefixes
        .iter()
        .enumerate()
        .filter(|(_, p)| ids.len() > p.len() && ids.starts_with(p))
        .max_by_key(|(_, p)| p.len())
        .map(|(i, _)| i)
}

#[cfg(test)]
mod best_prefix_match_tests {
    use super::*;

    #[test]
    fn the_longest_matching_prefix_wins_regardless_of_order() {
        let short: Vec<u32> = vec![1, 2];
        let long: Vec<u32> = vec![1, 2, 3, 4];
        let ids = vec![1, 2, 3, 4, 5];
        // Short listed first: a first-match search would pick it wrongly.
        assert_eq!(best_prefix_match(&ids, &[&short, &long]), Some(1));
        // Long listed first: must still win, not just win by accident of order.
        assert_eq!(best_prefix_match(&ids, &[&long, &short]), Some(0));
    }

    #[test]
    fn a_prefix_equal_to_ids_is_not_a_candidate() {
        let exact: Vec<u32> = vec![1, 2, 3];
        let ids = vec![1, 2, 3];
        assert_eq!(best_prefix_match(&ids, &[&exact]), None, "nothing left to bulk-forward");
        // A SHORTER genuine prefix beside the exact-length one still wins.
        let shorter: Vec<u32> = vec![1, 2];
        assert_eq!(best_prefix_match(&ids, &[&exact, &shorter]), Some(1));
    }

    #[test]
    fn a_non_prefix_never_matches_even_if_shorter() {
        let unrelated: Vec<u32> = vec![9, 9];
        let ids = vec![1, 2, 3, 4];
        assert_eq!(best_prefix_match(&ids, &[&unrelated]), None);
    }

    #[test]
    fn no_candidates_or_no_match_is_none() {
        let ids = vec![1, 2, 3];
        assert_eq!(best_prefix_match(&ids, &[]), None);
        let longer: Vec<u32> = vec![1, 2, 3, 4];
        assert_eq!(best_prefix_match(&ids, &[&longer]), None, "a prefix longer than ids cannot match");
    }
}

struct PromptCache {
    prefix: ModelState,
    prefix_ids: Vec<u32>,
    ready: Option<PreparedPrompt>,
}
/// What [`PromptCache::lookup`] found for an input: the ready entry (a
/// complete evaluation, cloned out), or where a prefill has to start.
enum Lookup {
    Ready(PreparedEvaluation),
    Cold { state: ModelState, start: usize },
}
impl PromptCache {
    /// The read half of [`PromptCache::prepare`]. Everything it returns is
    /// cloned out (`State::clone` shares the immutable prefix buffers), so
    /// the caller holds no borrow of the cache while it prefills: a
    /// generative adjudication pauses there, and an opinion read served at
    /// the pause may prepare on this same cache
    /// ([`YieldPoint::pause`]).
    fn lookup(&self, model: &Model, full: &[u32], use_cache: bool) -> Result<Lookup, Failure> {
        if !model.owns_state(&self.prefix) {
            return Err(Failure::Internal(
                "input checkpoint belongs to another model".into(),
            ));
        }
        if !full.starts_with(&self.prefix_ids) || full.len() <= self.prefix_ids.len() {
            return Err(Failure::BadRequest(
                "rendered input changes cached token prefix or has no suffix".into(),
            ));
        }
        if use_cache
            && let Some(ready) = &self.ready
            && ready.token_ids == full
        {
            return Ok(Lookup::Ready(PreparedEvaluation {
                state: ready.state.clone(),
                logits: ready.logits.clone(),
                cached_tokens: full.len(),
            }));
        }
        Ok(if use_cache {
            Lookup::Cold {
                state: self.prefix.clone(),
                start: self.prefix_ids.len(),
            }
        } else {
            Lookup::Cold {
                state: model.new_state(),
                start: 0,
            }
        })
    }
    /// The write half: replace the one ready entry with a COMPLETE prefill
    /// of `full`. Callers publish only after the last chunk, a device
    /// synchronize and a final check, so a failed or cancelled preparation
    /// leaves the previous entry; later decode never mutates the saved
    /// state/logits (they are clones).
    fn publish(&mut self, full: &[u32], state: &ModelState, logits: &Tensor) {
        self.ready = Some(PreparedPrompt {
            token_ids: full.to_vec(),
            state: state.clone(),
            logits: logits.clone(),
        });
    }
    fn prepare(
        &mut self,
        model: &Model,
        full: &[u32],
        use_cache: bool,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<PreparedEvaluation, Failure> {
        check()?;
        let (mut state, start) = match self.lookup(model, full, use_cache)? {
            Lookup::Ready(ready) => return Ok(ready),
            Lookup::Cold { state, start } => (state, start),
        };
        let logits = forward_chunks(model, &mut state, &full[start..], CHUNK, &mut || check())?;
        logits.device().synchronize()?;
        check()?;
        if use_cache {
            self.publish(full, &state, &logits);
        }
        Ok(PreparedEvaluation {
            state,
            logits,
            cached_tokens: start,
        })
    }
}

/// A validated LFM2.5 GGUF with its tokenizer: the chat template is the one
/// this renderer was checked against, the tokenizer agrees with the GGUF
/// vocabulary row for row, and both files are hashed. The daemon and the
/// examiner load through here so neither can run an unvalidated pairing.
pub struct Checkpoint {
    pub model: Model,
    pub tokenizer: tokenizers::Tokenizer,
    pub model_id: String,
    pub weight_hash: String,
    pub tokenizer_hash: String,
    pub weight_dtypes: Vec<String>,
    pub execution: crate::device::ExecutionDevice,
    pub state_size: StateSize,
}
/// What one model state costs, read off the GGUF: the byte accounting of the
/// chat checkpoint store (`crate::chat_session`). An upper bound, never an
/// estimate that could come in low.
#[derive(Clone, Copy, Debug)]
pub struct StateSize {
    /// K and V (f32, the dtype the model computes in) over every attention
    /// layer's KV heads, per position.
    pub kv_bytes_per_token: usize,
    /// Every convolution layer's state, at most `hidden` × `l_cache` f32s.
    pub conv_bytes: usize,
    /// The model's own context: the KV allocator never rounds past it.
    pub context: usize,
}
impl StateSize {
    fn from_gguf(ct: &gguf_file::Content) -> Result<Self, String> {
        let number = |key: &str| -> Result<usize, String> {
            ct.metadata
                .get(&format!("lfm2moe.{key}"))
                .ok_or_else(|| format!("GGUF has no lfm2moe.{key}"))?
                .to_u32()
                .map(|v| v as usize)
                .map_err(|e| e.to_string())
        };
        let hidden = number("embedding_length")?;
        let heads = number("attention.head_count")?;
        let l_cache = number("shortconv.l_cache")?;
        let context = number("context_length")?;
        let kv: Vec<usize> = match ct.metadata.get("lfm2moe.attention.head_count_kv") {
            Some(gguf_file::Value::Array(v)) => v
                .iter()
                .map(|n| match n {
                    gguf_file::Value::U32(v) => Ok(*v as usize),
                    gguf_file::Value::I32(v) if *v >= 0 => Ok(*v as usize),
                    _ => Err("KV head count must be a nonnegative integer".to_string()),
                })
                .collect::<Result<_, _>>()?,
            _ => return Err("GGUF has no per-layer lfm2moe.attention.head_count_kv".into()),
        };
        if heads == 0 || hidden % heads != 0 {
            return Err("invalid LFM2 MoE head configuration".into());
        }
        let f32_bytes = std::mem::size_of::<f32>();
        Ok(Self {
            kv_bytes_per_token: kv.iter().sum::<usize>() * (hidden / heads) * 2 * f32_bytes,
            conv_bytes: kv.iter().filter(|&&k| k == 0).count() * hidden * l_cache * f32_bytes,
            context,
        })
    }
    /// Bytes charged for a state of `len` positions: KV
    /// capacity rounded up as the allocator rounds it (a power of two, at
    /// least 128, at most the model's context).
    pub fn bytes(&self, len: usize) -> usize {
        let capacity = len
            .max(128)
            .checked_next_power_of_two()
            .unwrap_or(self.context)
            .min(self.context);
        self.kv_bytes_per_token * capacity + self.conv_bytes
    }
}
impl Checkpoint {
    pub fn load(
        path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        device: crate::device::DeviceArg,
        device_index: usize,
    ) -> Result<Self, String> {
        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("tokenizer {}: {e}", tokenizer_path.display()))?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
        let mut file = std::fs::File::open(path).map_err(|e| format!("GGUF {}: {e}", path.display()))?;
        let ct = gguf_file::Content::read(&mut file).map_err(|e| format!("GGUF {}: {e}", path.display()))?;
        let embedded_template = ct
            .metadata
            .get("tokenizer.chat_template")
            .ok_or("GGUF has no chat template")?
            .to_string()
            .map_err(|e| e.to_string())?;
        if sha256_hex_bytes(embedded_template.as_bytes())
            != "6d65c8804847ad74eea912dd7eca3dc1cf7a457b53a77f47d841a14121910963"
        {
            return Err("unsupported LFM2.5 chat template; validate the renderer before using this checkpoint".into());
        }
        let weight_dtypes: Vec<String> = ct
            .tensor_infos
            .values()
            .map(|t| format!("{:?}", t.ggml_dtype))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        // Match every usable token, including control tokens, to these weights.
        // GGUF pads vocabulary rows beyond the tokenizer's usable vocabulary.
        let vocab = ct
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or("GGUF has no tokenizer vocabulary")?
            .to_vec()
            .map_err(|e| e.to_string())?;
        for (text, id) in tokenizer.get_vocab(true) {
            if vocab
                .get(id as usize)
                .and_then(|v| v.to_string().ok())
                .map(String::as_str)
                != Some(text.as_str())
            {
                return Err(format!("tokenizer disagrees with GGUF at token {id}"));
            }
        }
        for (text, id) in [
            ("<|startoftext|>", 124894),
            ("<|im_start|>", 124899),
            ("<|im_end|>", EOS),
        ] {
            if tokenizer.token_to_id(text) != Some(id) {
                return Err(format!("unsupported LFM2.5 control token {text}"));
            }
        }
        let weight_hash = sha256_hex_file(path).map_err(|e| e.to_string())?;
        let tokenizer_hash = sha256_hex_file(tokenizer_path).map_err(|e| e.to_string())?;
        let state_size = StateSize::from_gguf(&ct)?;
        let execution = crate::device::ExecutionDevice::select(device, device_index)?;
        for reason in &execution.selection_reasons {
            eprintln!("lfm2d adjudicator device: {reason}");
        }
        let model =
            Model::from_gguf(ct, &mut file, &execution.device).map_err(|e| e.to_string())?;
        let model_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("invalid GGUF filename")?
            .to_string();
        Ok(Self {
            model,
            tokenizer,
            model_id,
            weight_hash,
            tokenizer_hash,
            weight_dtypes,
            execution,
            state_size,
        })
    }
}

/// How many described states each spec keeps ([`DescribedCache`]).
pub const DESCRIBED_CACHE_CAPACITY: usize = 16;

/// The prompt plus its generated description, standing at a question's slot:
/// the fourth cache layer. Greedy decoding under the grammar is a pure
/// function of the rendered prompt on a fixed backend, so the description
/// and the model state after it are reusable computation — never a cached
/// answer; the options are always scored fresh. Keyed by the exact prompt
/// token ids and the question field (a different field stops at a different
/// slot). Least recently used goes first.
struct DescribedEntry {
    prompt_ids: Vec<u32>,
    field: String,
    generated: Vec<u32>,
    text: String,
    state: ModelState,
    logits: Tensor,
}
struct DescribedCache {
    capacity: usize,
    entries: std::collections::VecDeque<DescribedEntry>,
}
impl DescribedCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: std::collections::VecDeque::with_capacity(capacity),
        }
    }
    /// A hit is moved to the back (most recent) and its parts cloned:
    /// `State::clone` shares the immutable prefix buffers.
    fn get(&mut self, prompt_ids: &[u32], field: &str) -> Option<(Vec<u32>, String, ModelState, Tensor)> {
        let at = self
            .entries
            .iter()
            .position(|e| e.field == field && e.prompt_ids == prompt_ids)?;
        let entry = self.entries.remove(at).expect("position came from this deque");
        let out = (
            entry.generated.clone(),
            entry.text.clone(),
            entry.state.clone(),
            entry.logits.clone(),
        );
        self.entries.push_back(entry);
        Some(out)
    }
    /// The deepest description held for these prompt bytes, whatever field
    /// it stopped at: the most of the report that a generation can resume
    /// from. Moved to the back like a hit.
    fn get_deepest(&mut self, prompt_ids: &[u32]) -> Option<(Vec<u32>, String, ModelState, Tensor)> {
        let field = self
            .entries
            .iter()
            .filter(|e| e.prompt_ids == prompt_ids)
            .max_by_key(|e| e.generated.len())
            .map(|e| e.field.clone())?;
        self.get(prompt_ids, &field)
    }
    fn insert(&mut self, entry: DescribedEntry) {
        if self.capacity == 0 {
            return;
        }
        if let Some(at) = self
            .entries
            .iter()
            .position(|e| e.field == entry.field && e.prompt_ids == entry.prompt_ids)
        {
            self.entries.remove(at);
        }
        while self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }
    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// A borrowed view of a [`Checkpoint`]'s pieces, so [`LoadedSpec::load`] can
/// run either against an owned `Checkpoint` (boot load) or against an
/// `Adjudicator`'s own fields (runtime registration, where the `Checkpoint`
/// was destructured into the adjudicator long ago) without duplicating any
/// of them.
#[derive(Clone, Copy)]
struct CheckpointView<'a> {
    model: &'a Model,
    tokenizer: &'a tokenizers::Tokenizer,
    model_id: &'a str,
    weight_hash: &'a str,
    tokenizer_hash: &'a str,
    weight_dtypes: &'a [String],
    execution: &'a crate::device::ExecutionDevice,
}
impl<'a> From<&'a Checkpoint> for CheckpointView<'a> {
    fn from(c: &'a Checkpoint) -> Self {
        Self {
            model: &c.model,
            tokenizer: &c.tokenizer,
            model_id: &c.model_id,
            weight_hash: &c.weight_hash,
            tokenizer_hash: &c.tokenizer_hash,
            weight_dtypes: &c.weight_dtypes,
            execution: &c.execution,
        }
    }
}

/// One loaded prompt spec: its rendered prefix resident as model state, its
/// grammar compiled once, its identity, and the caches that hang off it.
/// Every loaded spec (boot or uploaded) is on the `/v1/opinion` menu, and
/// none is privileged: `/v1/opinion` and `/v1/adjudicate` both name one.
struct LoadedSpec {
    /// Lowercase hex sha256 of the exact bytes this spec was loaded from.
    /// Content-addressed identity — see [`crate::hash::sha256_hex_bytes`]
    /// and `docs/system1-split-plan.md` (git f9ca081) "Runtime spec registration".
    id: String,
    /// A boot spec's file stem; an uploaded spec has no file, so this
    /// equals `id`. `/v1/opinion` and `/v1/adjudicate` accept either.
    name: String,
    /// The spec as loaded: it renders every user turn, so the reasoning mode
    /// and the schema cannot drift apart from what the prefix was built from.
    prompt: PromptSpec,
    prefix_text: String,
    /// `prompt.output_schema` compiled against the tokenizer, once, at load.
    grammar: Option<std::sync::Arc<crate::constrain::Grammar>>,
    cache: PromptCache,
    described: DescribedCache,
    info: PrefixInfo,
    menu: crate::opinion_api::SpecMenuEntry,
}
impl LoadedSpec {
    fn load(
        id: String,
        name: String,
        prompt: PromptSpec,
        view: &CheckpointView,
        context_limit: usize,
        repeat_penalty: f32,
    ) -> Result<Self, String> {
        let CheckpointView {
            model,
            tokenizer,
            model_id,
            weight_hash,
            tokenizer_hash,
            weight_dtypes,
            execution,
        } = *view;
        let prefix_text = prompt.render_prefix()?;
        if let Some(opinion) = &prompt.opinion {
            opinion.validate()?;
            prompt.render_user_turn_with_prefill("probe", &opinion.prefill)?;
        }
        // Compile the output grammar before the prefix prefill: a schema this
        // tokenizer cannot honour refuses to start, and never reaches a request.
        let grammar = prompt
            .output_schema
            .as_ref()
            .map(|schema| {
                crate::constrain::Grammar::compile(schema, tokenizer, model.vocab_size(), EOS)
                    .map_err(|e| format!("output_schema cannot be constrained: {e}"))
            })
            .transpose()?;
        let prefix_ids = tokenizer
            .encode(prefix_text.as_str(), false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        let boundary_probe = tokenizer
            .encode(format!("{prefix_text}<|im_start|>user\n"), false)
            .map_err(|e| e.to_string())?;
        if !boundary_probe.get_ids().starts_with(&prefix_ids) {
            return Err("tokenizer changes the system prefix at the user-message boundary".into());
        }
        if prefix_ids.is_empty() || prefix_ids.len() + 1 >= context_limit {
            return Err("adjudicator prefix leaves no context for input/output".into());
        }
        // The option split depends on the tokenizer and the spec, not on the
        // input, so a close that cannot separate the options is refused here
        // and never reaches a request.
        if let Some(opinion) = &prompt.opinion {
            let probe = prompt.render_user_turn_with_prefill("probe", &opinion.prefill)?;
            crate::opinion::split_options(tokenizer, &format!("{prefix_text}{probe}"), opinion)
                .map_err(|e| format!("opinion block cannot be read with this tokenizer: {e}"))?;
        }
        let mut prefix = model.new_state();
        for chunk in prefix_ids.chunks(CHUNK) {
            let _ = model
                .forward(chunk, &mut prefix)
                .map_err(|e| e.to_string())?;
        }
        let template_version = prompt.template_version();
        let snapshot_id = snapshot_id(
            &prompt,
            weight_hash,
            tokenizer_hash,
            &prefix_ids,
            &execution.identity,
            CANDLE_REV,
            repeat_penalty,
        )?;
        let info = PrefixInfo {
            model_id: model_id.to_string(),
            weight_hash: weight_hash.to_string(),
            tokenizer_hash: tokenizer_hash.to_string(),
            template_version: template_version.into(),
            snapshot_id,
            prefix_tokens: prefix_ids.len(),
            input_cache_capacity: 1,
            context_limit,
            backend: execution.backend.as_str().into(),
            device: execution.identity.clone(),
            candle_rev: CANDLE_REV.into(),
            dtype: "f32".into(),
            sampling: sampling(repeat_penalty),
            weight_dtypes: weight_dtypes.to_vec(),
        };
        let menu = crate::opinion_api::SpecMenuEntry::from_prompt(
            &id,
            &name,
            &prompt,
            &info.snapshot_id,
            DESCRIBED_CACHE_CAPACITY,
        )
        .map_err(|e| format!("prompt spec {name:?}: {e}"))?;
        // Every choice field's options must tokenize on their own pretokens
        // after the slot, or the read would score off the model's path. The
        // tokenizer's pre-tokenizer decides this, not the input, so it is
        // checked here on a probe and is an invariant at request time.
        for field in &menu.fields {
            if field.kind != crate::opinion_api::FieldKind::Choice {
                continue;
            }
            for (close, probe) in [("\",", "{\"x\": \"a\", \""), ("\"}", "{\"x\": \"a\", \"")] {
                let slot = format!(
                    "{probe}{}: \"",
                    serde_json::to_string(&field.field).map_err(|e| e.to_string())?
                );
                for option in &field.options {
                    let (_, stable) =
                        suffix_is_stable(tokenizer, &format!("{prefix_text}{slot}"), &format!("{option}{close}"))?;
                    if !stable {
                        return Err(format!(
                            "prompt spec {name:?}: option {option:?} of {:?} does not tokenize on \
                             its own after the slot; it cannot be read",
                            field.field
                        ));
                    }
                }
            }
        }
        Ok(Self {
            id,
            name,
            prompt,
            prefix_text,
            grammar,
            cache: PromptCache {
                prefix,
                prefix_ids,
                ready: None,
            },
            described: DescribedCache::new(DESCRIBED_CACHE_CAPACITY),
            info,
            menu,
        })
    }
}

/// A spec's `snapshot_id`: everything that decides what the daemon answers
/// under it. Only the spec's parsed content counts, never its id or file
/// name, so the same content loaded twice lands on the same snapshot.
fn snapshot_id(
    prompt: &PromptSpec,
    weight_hash: &str,
    tokenizer_hash: &str,
    prefix_ids: &[u32],
    device: &str,
    candle_rev: &str,
    repeat_penalty: f32,
) -> Result<String, String> {
    // The repetition penalty shapes every description and report, so a
    // consumer that refits on `snapshot_id` must see it change.
    let mut identity = serde_json::json!([
        prompt.template_version(),
        weight_hash,
        tokenizer_hash,
        prefix_ids,
        device,
        candle_rev,
        "f32",
        repeat_penalty
    ]);
    let parts = identity.as_array_mut().expect("identity is an array");
    // The opinion question is part of what this daemon answers, so it is
    // part of its identity. Appended only when present: specs without one
    // keep the snapshot ids they had.
    if let Some(opinion) = &prompt.opinion {
        parts.push(serde_json::to_value(opinion).map_err(|e| e.to_string())?);
    }
    // The label is rendered into every user turn, not the prefix, so
    // `prefix_ids` never sees it: without this, two specs that differ only
    // in what they call their input would share a snapshot. Always present,
    // so adding it changed every spec's snapshot_id once (2026-09-24).
    parts.push(serde_json::json!({"input_label": prompt.input_label}));
    Ok(sha256_hex_bytes(&serde_json::to_vec(&identity).map_err(|e| e.to_string())?))
}

/// The sampling description every identity carries: one definition, so the
/// checkpoint's (`GET /v1/adjudicator`) and each spec's cannot disagree.
fn sampling(repeat_penalty: f32) -> String {
    format!("greedy; repetition_penalty={repeat_penalty}; history=full")
}

fn encode_ids(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec())
}

/// The readability check every teacher-forced continuation must pass before
/// it can be trusted: `suffix`, tokenized ALONE, must be exactly the tail of
/// `prefix` and `suffix` tokenized TOGETHER. A BPE merge across the
/// prefix/suffix boundary can otherwise put `suffix`'s canonical tokens on a
/// path the model would never write from `prefix` — scoring `alone`'s tokens
/// there would then be scoring gibberish, not the continuation.
///
/// Three call sites share this: [`LoadedSpec::load`]'s pretokenization probe
/// over every opinion option, [`Adjudicator::describe_then_read`]'s
/// per-question option check, and `POST /v1/tokenize`'s `context`
/// suffix-stability report (`crate::tokenize_api`) — the same fact asked for
/// three different reasons (refuse at load, refuse at request time, report
/// as data), so it is one function, not three copies that could drift.
///
/// Returns `suffix`'s own encoding (a caller that goes on to teacher-force
/// it, or to report it, doesn't need to re-tokenize) alongside whether it is
/// stable. `suffix` encoding to nothing is never stable (there would be
/// nothing left to force or to report as "the tokens after context").
pub(crate) fn suffix_is_stable(
    tokenizer: &tokenizers::Tokenizer,
    prefix: &str,
    suffix: &str,
) -> Result<(Vec<u32>, bool), String> {
    let alone = encode_ids(tokenizer, suffix)?;
    let whole = encode_ids(tokenizer, &format!("{prefix}{suffix}"))?;
    let stable = !alone.is_empty() && whole.ends_with(&alone);
    Ok((alone, stable))
}

/// `snapshot_id` must cover the spec's `input_label`: the label is rendered
/// into every user turn, never the prefix, so nothing else in the identity
/// sees it. Model-free: the identity is a pure function of its inputs.
/// The opinion engine's error `type`s are the vocabulary `lfm2d/README.md`
/// documents, and a 500 is spelled like the encoder routes' (`"internal"`,
/// `types::ApiError`), so a consumer switching on `type` meets one word for
/// one failure across the daemon.
#[cfg(test)]
mod failure_wire_tests {
    use super::*;

    async fn kind(f: Failure) -> (u16, String) {
        let response = f.into_response();
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        (status, v["error"]["type"].as_str().unwrap().to_string())
    }

    #[tokio::test]
    async fn every_failure_speaks_the_documented_vocabulary() {
        let cases = [
            (Failure::BadRequest("x".into()), 400, "bad_request"),
            (Failure::NotFound("x".into()), 404, "not_found"),
            (Failure::Forbidden("x".into()), 403, "forbidden"),
            (Failure::Unprocessable("x".into()), 422, "unprocessable"),
            (Failure::Cancelled, 408, "cancelled"),
            (Failure::Deadline, 504, "deadline"),
            (Failure::Internal("x".into()), 500, "internal"),
        ];
        for (failure, status, expected) in cases {
            assert_eq!(kind(failure).await, (status, expected.to_string()));
        }
    }
}

#[cfg(test)]
mod snapshot_id_tests {
    use super::*;

    fn spec(label: &str) -> PromptSpec {
        serde_json::from_value(serde_json::json!({"input_label": label, "system": "Judge."})).unwrap()
    }
    fn id(p: &PromptSpec) -> String {
        snapshot_id(p, "w", "t", &[1, 2, 3], "rocm:gfx1151:hip7.2", "rev-a", 1.05).unwrap()
    }

    #[test]
    fn a_spec_that_changes_only_its_input_label_gets_a_new_snapshot_id() {
        assert_eq!(id(&spec("Email")), id(&spec("Email")), "same content, same snapshot");
        assert_ne!(id(&spec("Email")), id(&spec("Ticket")), "the label is part of what is answered");
    }

    #[test]
    fn the_prefix_and_the_penalty_still_move_it() {
        let p = spec("Email");
        assert_ne!(id(&p), snapshot_id(&p, "w", "t", &[1, 2, 4], "rocm:gfx1151:hip7.2", "rev-a", 1.05).unwrap());
        assert_ne!(id(&p), snapshot_id(&p, "w", "t", &[1, 2, 3], "rocm:gfx1151:hip7.2", "rev-a", 1.0).unwrap());
    }

    /// Kernels branch on the GPU target and change with the candle build, so
    /// the same weights and spec give different numbers on another card or
    /// another fork revision; a consumer that refits on `snapshot_id` must
    /// see both move it, not just the backend's name.
    #[test]
    fn the_device_and_the_candle_build_move_it() {
        let p = spec("Email");
        assert_ne!(id(&p), snapshot_id(&p, "w", "t", &[1, 2, 3], "rocm:gfx1100:hip7.2", "rev-a", 1.05).unwrap());
        assert_ne!(id(&p), snapshot_id(&p, "w", "t", &[1, 2, 3], "rocm:gfx1151:hip7.1", "rev-a", 1.05).unwrap());
        assert_ne!(id(&p), snapshot_id(&p, "w", "t", &[1, 2, 3], "rocm:gfx1151:hip7.2", "rev-b", 1.05).unwrap());
    }
}

#[cfg(test)]
mod checkpoint_load_error_tests {
    use super::*;
    use std::path::{Path, PathBuf};

    // A missing file must name itself: a bare "No such file or directory"
    // leaves an operator (or a fresh clone's test run) guessing which of two
    // paths it was.
    #[test]
    fn a_missing_tokenizer_names_its_path() {
        let tok = PathBuf::from("/nonexistent/lfm2d-test/tokenizer.json");
        let err = Checkpoint::load(Path::new("/nonexistent/model.gguf"), &tok, crate::device::DeviceArg::Cpu, 0)
            .err()
            .expect("a missing tokenizer must fail");
        assert!(err.contains("/nonexistent/lfm2d-test/tokenizer.json"), "{err}");
    }

    #[test]
    fn a_missing_gguf_names_its_path() {
        let models = std::env::var_os("LFM2_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join(".models"));
        let tok = models.join("LFM2.5-8B-A1B/tokenizer.json");
        assert!(tok.is_file(), "missing tokenizer at {} (see docs/development.md)", tok.display());
        let gguf = PathBuf::from("/nonexistent/lfm2d-test/model.gguf");
        let err = Checkpoint::load(&gguf, &tok, crate::device::DeviceArg::Cpu, 0)
            .err()
            .expect("a missing GGUF must fail");
        assert!(err.contains("/nonexistent/lfm2d-test/model.gguf"), "{err}");
    }
}

/// Real-tokenizer facts pinned against the actual LFM2.5-8B-A1B
/// `tokenizer.json` — no GGUF, no candle model, so this stays a plain
/// `#[test]` rather than an `#[ignore]`d real-model one (same "Tier 1b"
/// convention `tests/constrained_decoding.rs` uses for its
/// `real_vocabulary_*` tests). Verify against the fixture, not a
/// description of it: a prior version of this comment claimed `"allow"`
/// takes 1 token from memory alone, which is exactly the kind of claim
/// this module exists to pin instead of assert from recollection.
#[cfg(test)]
mod suffix_is_stable_tests {
    use super::*;
    use std::path::PathBuf;

    fn models_dir() -> PathBuf {
        std::env::var_os("LFM2_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join(".models")
            })
    }

    fn real_tokenizer() -> tokenizers::Tokenizer {
        let path = std::env::var_os("LFM2D_ADJUDICATOR_TOKENIZER")
            .map(PathBuf::from)
            .unwrap_or_else(|| models_dir().join("LFM2.5-8B-A1B/tokenizer.json"));
        assert!(
            path.is_file(),
            "missing tokenizer at {}\n\n  (hf download LiquidAI/LFM2.5-8B-A1B tokenizer.json \
             --local-dir .models/LFM2.5-8B-A1B, or point LFM2D_ADJUDICATOR_TOKENIZER at it; \
             LFM2_MODELS_DIR overrides the .models root)\n",
            path.display()
        );
        tokenizers::Tokenizer::from_file(&path).unwrap()
    }

    /// The verdict slot's own prefix: no trailing space before the option,
    /// the shape every shipped opinion spec uses. `allow` is one token
    /// here; `deny` is two (`memory verdict-words-tokenize-in-string-context`).
    /// Both are stable: the pretokenizer never merges across the closing
    /// quote.
    #[test]
    fn allow_is_one_token_and_deny_is_two_at_the_verdict_slot_and_both_are_stable() {
        let tok = real_tokenizer();
        let prefix = "{\"verdict\": \"";
        let (allow_ids, allow_stable) = suffix_is_stable(&tok, prefix, "allow").unwrap();
        assert_eq!(allow_ids, vec![13537], "allow's token id moved; re-pin or re-check the fixture");
        assert!(allow_stable);
        let (deny_ids, deny_stable) = suffix_is_stable(&tok, prefix, "deny").unwrap();
        assert_eq!(deny_ids.len(), 2, "deny's token count moved: {deny_ids:?}");
        assert!(deny_stable);
    }

    /// The hazard the check exists to catch: a prefix that ends in a SPACE
    /// (unlike every real opinion prefill, which ends in `"`) merges the
    /// space into the option's first token, so the option's own alone
    /// encoding is no longer a tail of the combined encoding at all —
    /// `suffix_is_stable` must say so rather than silently reporting it
    /// readable.
    #[test]
    fn a_prefix_ending_in_a_space_makes_the_option_unstable() {
        let tok = real_tokenizer();
        let (_, stable) = suffix_is_stable(&tok, "the verdict is ", "allow").unwrap();
        assert!(!stable, "a leading-space BPE merge must be caught, not missed");
    }

    #[test]
    fn an_empty_suffix_is_never_stable() {
        let tok = real_tokenizer();
        let (ids, stable) = suffix_is_stable(&tok, "hello ", "").unwrap();
        assert!(ids.is_empty());
        assert!(!stable, "nothing was left to force or to report");
    }
}

/// What [`SpecStore`] needs to identify an entry: its content-hash id and
/// its lookup name (boot: file stem; uploaded: the id again).
trait SpecIdentity {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
}
impl SpecIdentity for LoadedSpec {
    fn id(&self) -> &str {
        &self.id
    }
    fn name(&self) -> &str {
        &self.name
    }
}

/// Content-addressed spec bookkeeping: the fixed boot specs (never evicted,
/// possibly none) plus a bounded,
/// least-recently-used cache of runtime-uploaded specs
/// (`POST /v1/opinion/specs`). Pure data structure — no model access, no I/O
/// — so the dedup/eviction rules "Runtime spec registration" rules on
/// (`docs/system1-split-plan.md` (git f9ca081)) are unit-tested below without a
/// checkpoint. [`Adjudicator`] is the only production user; `T` is
/// [`LoadedSpec`] there.
struct SpecStore<T> {
    boot: Vec<T>,
    /// Front = least recently used, back = most recently used. A hit from
    /// [`SpecStore::resolve_mut`] or [`SpecStore::register_or_load`] both
    /// touch an entry to the back — "use" is registered-or-served, per the
    /// ruling.
    uploaded: std::collections::VecDeque<T>,
    capacity: usize,
}
enum RemoveOutcome<T> {
    Removed(T),
    Boot,
    NotFound,
}
impl<T: SpecIdentity> SpecStore<T> {
    fn new(boot: Vec<T>, capacity: usize) -> Self {
        Self { boot, uploaded: std::collections::VecDeque::new(), capacity }
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    /// Whether `id` already names a loaded spec, boot or uploaded, without
    /// touching LRU order — used to let a re-registration of an existing id
    /// through even when `capacity` is 0 (which refuses every GENUINE load;
    /// see [`SpecStore::register_or_load`]'s doc on why that precondition
    /// is the caller's job, not this method's).
    fn contains(&self, id: &str) -> bool {
        self.boot.iter().any(|s| s.id() == id) || self.uploaded.iter().any(|s| s.id() == id)
    }
    /// Every loaded spec, boot first, in the order `/v1/opinion/specs`
    /// lists them.
    fn iter(&self) -> impl Iterator<Item = &T> {
        self.boot.iter().chain(self.uploaded.iter())
    }
    /// The spec `key` names: a boot spec's id or name, else an uploaded
    /// spec's id, touched to the back on a hit. `None` on an unknown key —
    /// there is no default spec to fall back to, so a stale caller gets a
    /// clean miss rather than a silently wrong spec.
    fn resolve_mut(&mut self, key: &str) -> Option<&mut T> {
        if let Some(pos) = self.boot.iter().position(|s| s.id() == key || s.name() == key) {
            return Some(&mut self.boot[pos]);
        }
        if let Some(pos) = self.uploaded.iter().position(|s| s.id() == key) {
            let spec = self.uploaded.remove(pos).expect("position just found");
            self.uploaded.push_back(spec);
            return self.uploaded.back_mut();
        }
        None
    }
    /// The spec whose id is exactly `id`, boot or uploaded, WITHOUT touching
    /// LRU order: a job returning to a spec it already resolved (and
    /// touched) once, to publish into its caches, is not a second use.
    fn get_mut(&mut self, id: &str) -> Option<&mut T> {
        self.boot
            .iter_mut()
            .chain(self.uploaded.iter_mut())
            .find(|s| s.id() == id)
    }
    /// `/v1/probe`'s warm-prefix resume: the spec (boot or uploaded) whose
    /// resident prefix is the LONGEST STRICT match for `ids`
    /// ([`best_prefix_match`] — deterministic regardless of iteration or
    /// LRU order, never "whichever is checked first"). A hit on an
    /// UPLOADED spec touches it to the back of the LRU, same as
    /// [`SpecStore::resolve_mut`]'s hit does: "served-or-registered = use"
    /// (`docs/system1-split-plan.md` (git f9ca081) "Runtime spec registration") applies
    /// here exactly as it does to a request that names the spec by id —
    /// resuming its resident state IS serving a request against it.
    /// `prefix_ids` extracts each spec's resident-prefix token ids; a
    /// closure rather than a trait requirement so this stays usable from
    /// `spec_store_tests`' model-free `Fixture` without giving every
    /// `SpecStore<T>` a real `ModelState`.
    fn resolve_best_prefix_mut(&mut self, ids: &[u32], prefix_ids: impl Fn(&T) -> &[u32]) -> Option<&mut T> {
        let prefixes: Vec<&[u32]> = self.boot.iter().chain(self.uploaded.iter()).map(&prefix_ids).collect();
        let winner = best_prefix_match(ids, &prefixes)?;
        let boot_len = self.boot.len();
        if winner < boot_len {
            Some(&mut self.boot[winner])
        } else {
            let pos = winner - boot_len;
            let spec = self.uploaded.remove(pos).expect("position just found");
            self.uploaded.push_back(spec);
            self.uploaded.back_mut()
        }
    }
    /// The registration decision, in one place: already a boot spec (no
    /// work), already an uploaded spec (touched to the back, no work), or a
    /// genuine miss — `load` runs, at most once, only then, evicting the
    /// least recently used upload first if this would exceed capacity. This
    /// is what makes registering the same bytes twice free the second time:
    /// `(spec, newly_loaded, evicted_id)`.
    ///
    /// Requires `capacity >= 1` for a genuine miss to make sense at all —
    /// the caller's job ([`SpecStore::contains`] before this, or refuse the
    /// upload outright), not this method's: it evicts-then-inserts, which
    /// keeps the uploaded count at exactly `capacity` for `capacity >= 1`
    /// but cannot honour `capacity == 0` (there is no entry left to hand
    /// back a reference to after evicting the one just inserted).
    fn register_or_load<E>(
        &mut self,
        id: &str,
        load: impl FnOnce() -> Result<T, E>,
    ) -> Result<(&T, bool, Option<String>), E> {
        if let Some(pos) = self.boot.iter().position(|s| s.id() == id) {
            return Ok((&self.boot[pos], false, None));
        }
        if let Some(pos) = self.uploaded.iter().position(|s| s.id() == id) {
            let spec = self.uploaded.remove(pos).expect("position just found");
            self.uploaded.push_back(spec);
            return Ok((self.uploaded.back().expect("just touched"), false, None));
        }
        let spec = load()?;
        let evicted = if self.uploaded.len() >= self.capacity {
            self.uploaded.pop_front().map(|s| s.id().to_string())
        } else {
            None
        };
        self.uploaded.push_back(spec);
        Ok((self.uploaded.back().expect("just inserted"), true, evicted))
    }
    /// Unload an uploaded spec by id. A boot spec's id OR name is never
    /// removable — [`RemoveOutcome::Boot`], matching the same two keys
    /// [`SpecStore::resolve_mut`] answers a boot spec to (an uploaded
    /// spec's id still answers only to itself here, same as resolution).
    fn remove(&mut self, id: &str) -> RemoveOutcome<T> {
        if self.boot.iter().any(|s| s.id() == id || s.name() == id) {
            return RemoveOutcome::Boot;
        }
        match self.uploaded.iter().position(|s| s.id() == id) {
            Some(pos) => RemoveOutcome::Removed(self.uploaded.remove(pos).expect("position just found")),
            None => RemoveOutcome::NotFound,
        }
    }
}

/// A `spec` naming nothing loaded: 404, the client's cue to upload it (or
/// re-upload, if it was evicted) and retry. Never a fallback to another
/// spec — a stale menu view must fail loudly here, not silently serve the
/// wrong prefix.
fn unknown_spec(key: &str) -> Failure {
    Failure::NotFound(format!(
        "no loaded spec {key:?}; POST /v1/opinion/specs to upload it, or GET /v1/opinion/specs \
         to list what's loaded"
    ))
}

pub struct Adjudicator {
    /// Shared so a generative adjudication can forward through its own
    /// handle while `pause` holds `&mut self` (see [`YieldPoint`]).
    model: Arc<Model>,
    tokenizer: tokenizers::Tokenizer,
    eos: u32,
    repeat_penalty: f32,
    model_id: String,
    weight_hash: String,
    tokenizer_hash: String,
    weight_dtypes: Vec<String>,
    execution: crate::device::ExecutionDevice,
    context_limit: usize,
    specs: SpecStore<LoadedSpec>,
    /// Chat checkpoints (`POST /v1/chat`), which tail reads fork. See
    /// `crate::chat_session`.
    chats: crate::chat_session::CheckpointStore<crate::chat_session::ChatCheckpoint>,
    state_size: StateSize,
}
impl Adjudicator {
    pub fn load(cli: &Cli) -> Result<Self, String> {
        let path = cli
            .adjudicator_model
            .as_ref()
            .ok_or("missing adjudicator model")?;
        let tokenizer_path = cli
            .adjudicator_tokenizer
            .as_ref()
            .ok_or("missing adjudicator tokenizer")?;
        let checkpoint = Checkpoint::load(path, tokenizer_path, cli.device, cli.device_index)?;
        if cli.adjudicator_context > checkpoint.model.context_length() {
            return Err("adjudicator context exceeds model context".into());
        }
        let mut boot_specs = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        {
            let view = CheckpointView::from(&checkpoint);
            for spec_path in &cli.opinion_specs {
                let name = spec_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| format!("prompt spec path {} has no name", spec_path.display()))?
                    .to_string();
                if !names.insert(name.clone()) {
                    return Err(format!("prompt spec name {name:?} repeats; names come from file stems"));
                }
                let bytes =
                    std::fs::read(spec_path).map_err(|e| format!("{}: {e}", spec_path.display()))?;
                let id = sha256_hex_bytes(&bytes);
                let prompt: PromptSpec = serde_json::from_slice(&bytes)
                    .map_err(|e| format!("{}: {e}", spec_path.display()))?;
                boot_specs.push(LoadedSpec::load(
                    id,
                    name,
                    prompt,
                    &view,
                    cli.adjudicator_context,
                    cli.adjudicator_repeat_penalty,
                )?);
            }
        }
        // Compile/warm device selection kernels before announcing readiness:
        // over each boot spec's own prefix (every one of them was just
        // prefilled above), or over a one-token history when the menu starts
        // empty, so the first upload does not pay for the kernels either.
        let warm_eos = [EOS];
        let warm_histories: Vec<&[u32]> = if boot_specs.is_empty() {
            vec![&warm_eos]
        } else {
            boot_specs.iter().map(|s| s.cache.prefix_ids.as_slice()).collect()
        };
        for history in warm_histories {
            let _ = GreedySampler::new(
                &checkpoint.execution.device,
                checkpoint.model.vocab_size(),
                history,
                cli.adjudicator_repeat_penalty,
            )
            .map_err(|e| e.to_string())?;
        }
        checkpoint
            .execution
            .device
            .synchronize()
            .map_err(|e| e.to_string())?;
        let Checkpoint {
            model,
            tokenizer,
            model_id,
            weight_hash,
            tokenizer_hash,
            weight_dtypes,
            execution,
            state_size,
        } = checkpoint;
        let budget = cli
            .chat_checkpoint_budget_mib
            .checked_mul(1 << 20)
            .ok_or("--chat-checkpoint-budget-mib overflows")?;
        let full = state_size.bytes(cli.adjudicator_context) + checkpoint_extra_bytes(cli.adjudicator_context);
        if budget < full {
            return Err(format!(
                "--chat-checkpoint-budget-mib {} cannot hold one checkpoint at the full {}-token context \
                 ({} MiB)",
                cli.chat_checkpoint_budget_mib,
                cli.adjudicator_context,
                full.div_ceil(1 << 20)
            ));
        }
        Ok(Self {
            model: Arc::new(model),
            tokenizer,
            eos: EOS,
            repeat_penalty: cli.adjudicator_repeat_penalty,
            model_id,
            weight_hash,
            tokenizer_hash,
            weight_dtypes,
            execution,
            context_limit: cli.adjudicator_context,
            specs: SpecStore::new(boot_specs, cli.opinion_spec_capacity),
            chats: crate::chat_session::CheckpointStore::new(budget),
            state_size,
        })
    }
    /// The checkpoint's identity, for `GET /v1/adjudicator`. Built from the
    /// adjudicator itself, never from a spec: the menu may be empty.
    pub fn info(&self) -> AdjudicatorInfo {
        AdjudicatorInfo {
            model_id: self.model_id.clone(),
            weight_hash: self.weight_hash.clone(),
            tokenizer_hash: self.tokenizer_hash.clone(),
            context_limit: self.context_limit,
            backend: self.execution.backend.as_str().into(),
            device: self.execution.identity.clone(),
            candle_rev: CANDLE_REV.into(),
            dtype: "f32".into(),
            sampling: sampling(self.repeat_penalty),
            weight_dtypes: self.weight_dtypes.clone(),
        }
    }
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: self.model_id.clone(),
            kind: ModelKind::Adjudicator,
            weight_hash: self.weight_hash.clone(),
            labels: None,
            hidden_size: self.model.hidden_size(),
        }
    }
    /// Every loaded spec, boot then uploaded, as `/v1/opinion/specs` lists
    /// it.
    pub fn menu(&self) -> Vec<crate::opinion_api::SpecMenuEntry> {
        self.specs.iter().map(|s| s.menu.clone()).collect()
    }
    /// A clone of this adjudicator's own tokenizer, for `POST /v1/tokenize`
    /// (`crate::tokenize_api`). `main.rs` calls this BEFORE handing `self`
    /// to [`Handle::spawn`], which moves it into the worker thread — the
    /// whole point of cloning here is that tokenizing never waits behind
    /// the worker's model queue (`docs/system1-split-plan.md` (git f9ca081) "Tokenize
    /// and probe endpoints"). `tokenizers::Tokenizer` clones cheaply (its
    /// heavy pieces — the vocabulary, the merge table — are reference
    /// counted internally), so this is not a second copy of the vocabulary.
    pub fn tokenizer_clone(&self) -> tokenizers::Tokenizer {
        self.tokenizer.clone()
    }
}
impl Generator for Adjudicator {
    /// Pauses before every prefill chunk and every decoded token. Every
    /// borrow of `self.specs` is scoped to end before the first pause: what
    /// the loop needs from the spec is cloned out up front, and the prefill
    /// is published afterwards through a fresh lookup by id (see
    /// [`YieldPoint`]).
    fn generate(
        &mut self,
        request: &AdjudicateRequest,
        at: &dyn YieldPoint<Self>,
    ) -> Result<AdjudicateResponse, Failure> {
        request.validate().map_err(Failure::BadRequest)?;
        if request.opinion {
            return self.opinion(request, &|| at.check());
        }
        let key = request.spec_key().map_err(Failure::BadRequest)?;
        let spec = self
            .specs
            .resolve_mut(key)
            .ok_or_else(|| unknown_spec(key))?;
        if let Some(d) = &request.distributions {
            d.validate_vocab(self.model.vocab_size())
                .map_err(Failure::BadRequest)?;
            if d.constrained && spec.prompt.output_schema.is_none() {
                return Err(Failure::BadRequest(
                    "distributions.constrained needs an output schema; this adjudicator has none"
                        .into(),
                ));
            }
        }
        at.check()?;
        let suffix = spec.prompt.render_user_turn(&request.input);
        let full = self
            .tokenizer
            .encode(format!("{}{suffix}", spec.prefix_text), false)
            .map_err(|e| Failure::Internal(e.to_string()))?
            .get_ids()
            .to_vec();
        if full
            .len()
            .checked_add(request.max_tokens)
            .is_none_or(|n| n > spec.info.context_limit)
        {
            return Err(Failure::BadRequest(
                "prompt plus max_tokens exceeds adjudicator context".into(),
            ));
        }
        let begin = Instant::now();
        // Escalation from an opinion: if `/v1/opinion` left a described state
        // for these exact prompt bytes, continue from its slot rather than
        // describing again. Greedy under the grammar is a pure function of
        // the prompt, so the report is the one a fresh generation writes. A
        // request that wants every step's distribution, or a cold one, gets
        // the fresh path.
        let resumed = if request.use_cache && request.distributions.is_none() {
            spec.described
                .get_deepest(&full)
                .filter(|(generated, ..)| generated.len() < request.max_tokens)
        } else {
            None
        };
        // Resumed, ready, or a prefill still to run: all three are owned
        // values from here on, so the lookup's borrow of the spec ends.
        let resumed_or_lookup = match resumed {
            Some(resumed) => Ok(resumed),
            None => Err(spec.cache.lookup(&self.model, &full, request.use_cache)?),
        };
        // What the rest of this job reads from the spec, cloned while it is
        // still borrowed: an opinion read served at a pause below resolves
        // the spec store itself.
        let spec_id = spec.id.clone();
        let spec_info = spec.info.clone();
        let grammar = spec.grammar.clone();
        let output_schema = spec.prompt.output_schema.clone();
        let (mut state, mut logits, cached_tokens, mut generated, resumed_tokens) = match resumed_or_lookup {
            Ok((generated, _, state, logits)) => {
                let n = generated.len();
                (state, logits, full.len(), generated, Some(n))
            }
            Err(Lookup::Ready(PreparedEvaluation {
                state,
                logits,
                cached_tokens,
            })) => (state, logits, cached_tokens, Vec::new(), None),
            Err(Lookup::Cold { mut state, start }) => {
                // `PromptCache::prepare`'s cold branch, with pauses between
                // chunks: the same chunk boundaries, so the same numerics.
                let model = self.model.clone();
                let logits = forward_chunks(&model, &mut state, &full[start..], CHUNK, &mut || at.pause(self))?;
                logits.device().synchronize()?;
                at.check()?;
                if request.use_cache {
                    // Complete, synchronized and still wanted: only now does
                    // it reach the cache. Reads served at the pauses may
                    // have replaced `ready` meanwhile; this replaces theirs,
                    // as a later request would. The spec cannot have left
                    // the store (only registration and deletion remove one,
                    // and they never run at a pause), so its absence is a
                    // bug, not a miss.
                    self.specs
                        .get_mut(&spec_id)
                        .ok_or_else(|| {
                            Failure::Internal(format!(
                                "spec {spec_id} left the store while its generation was paused"
                            ))
                        })?
                        .cache
                        .publish(&full, &state, &logits);
                }
                (state, logits, start, Vec::new(), None)
            }
        };
        let prefill_ms = begin.elapsed().as_secs_f64() * 1000.;
        let decode = Instant::now();
        // Constrained decoding. With an `output_schema` this masks the logits
        // against the JSON grammar compiled at load before every greedy selection, so an
        // invalid report is unreachable rather than merely detected afterwards
        // by `validate_report`. Without a schema it is `GreedySampler` itself.
        // See `crate::constrain` for the grammar's scope and rulings. The mask
        // lives inside the decoder and never touches `logits`; the loop below
        // goes through `Decoder::step`, which records from the raw logits.
        // A resumed description is history for the penalty and walked by the
        // grammar, exactly as if this loop had sampled it.
        let history: Vec<u32> = full.iter().chain(&generated).copied().collect();
        let mut sampler = crate::constrain::Decoder::new(
            grammar.as_ref(),
            logits.device(),
            self.model.vocab_size(),
            &history,
            self.repeat_penalty,
        )?;
        sampler.advance(&generated)?;
        // Allocated only when asked for: the distribution computation below
        // costs nothing on the request path that doesn't request it.
        let mut distributions = request
            .distributions
            .as_ref()
            .map(|_| Vec::with_capacity(request.max_tokens));
        let mut finish_reason = "length";
        for i in generated.len()..request.max_tokens {
            at.pause(self)?;
            // One call selects AND records, so the record cannot be moved
            // to the wrong side of the grammar mask: see `Decoder::step`.
            let (token, step) = sampler.step(&logits, request.distributions.as_ref(), |id| {
                self.tokenizer.id_to_token(id)
            })?;
            if self.tokenizer.id_to_token(token).is_none() {
                return Err(Failure::Internal(format!(
                    "model selected unused vocabulary row {token}"
                )));
            }
            if let Some(step) = step {
                distributions
                    .as_mut()
                    .expect("allocated above whenever distributions is requested")
                    .push(step);
            }
            generated.push(token);
            if token == self.eos {
                finish_reason = "stop";
                break;
            }
            if i + 1 < request.max_tokens {
                logits = self.model.forward(&[token], &mut state)?;
            }
        }
        at.check()?;
        // Nothing is stripped but the terminating im_end: tool delimiters, and
        // anything else the model wrote, reach `output` as generated.
        let content = if generated.last() == Some(&self.eos) {
            &generated[..generated.len() - 1]
        } else {
            &generated
        };
        let output = self
            .tokenizer
            .decode(content, false)
            .map_err(|e| Failure::Internal(e.to_string()))?;
        let (report, report_error) = match &output_schema {
            None => (None, None),
            Some(schema) => match validate_report(&output, schema, finish_reason) {
                Ok(report) => (Some(report), None),
                Err(error) => (None, Some(error)),
            },
        };
        Ok(AdjudicateResponse {
            prefix: spec_info,
            output,
            report,
            report_error,
            finish_reason: finish_reason.into(),
            prompt_tokens: full.len(),
            cached_tokens,
            completion_tokens: generated.len(),
            queue_ms: 0.,
            prefill_ms,
            decode_ms: decode.elapsed().as_secs_f64() * 1000.,
            distributions,
            opinion: None,
            resumed_tokens,
        })
    }

    fn opine(
        &mut self,
        request: &crate::opinion_api::OpinionRequest,
        questions: &[crate::opinion_api::ResolvedQuestion],
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<crate::opinion_api::OpinionResponse, Failure> {
        request.validate().map_err(Failure::BadRequest)?;
        self.describe_then_read(request, questions, check)
    }

    fn register(
        &mut self,
        id: String,
        prompt: PromptSpec,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<RegisterOutcome, Failure> {
        check()?;
        let begin = Instant::now();
        if self.specs.capacity() == 0 && !self.specs.contains(&id) {
            // A genuine load into a zero-capacity store has nothing to
            // evict its way into — see `SpecStore::register_or_load`'s doc.
            // `Cli::validate` refuses `--opinion-spec-capacity 0` in
            // production; this only fires for a test double or a future
            // caller that skips that check.
            return Err(Failure::Unprocessable(
                "opinion spec capacity is 0; no uploads can be held".into(),
            ));
        }
        // Built from individual field borrows (not `self.checkpoint_view()`,
        // a method call the borrow checker would treat as borrowing all of
        // `self`) so this coexists with the `&mut self.specs` borrow below —
        // disjoint fields, both live at once.
        let view = CheckpointView {
            model: &self.model,
            tokenizer: &self.tokenizer,
            model_id: &self.model_id,
            weight_hash: &self.weight_hash,
            tokenizer_hash: &self.tokenizer_hash,
            weight_dtypes: &self.weight_dtypes,
            execution: &self.execution,
        };
        let context_limit = self.context_limit;
        let repeat_penalty = self.repeat_penalty;
        let id_for_load = id.clone();
        let (entry, newly_loaded, evicted) = self
            .specs
            .register_or_load(&id, move || {
                LoadedSpec::load(
                    id_for_load.clone(),
                    id_for_load,
                    prompt,
                    &view,
                    context_limit,
                    repeat_penalty,
                )
            })
            .map(|(spec, newly_loaded, evicted)| (spec.menu.clone(), newly_loaded, evicted))
            .map_err(Failure::Unprocessable)?;
        // No `check()` here. `register_or_load` above already committed the
        // mutation (inserted the spec, possibly evicted an LRU one) —
        // `self.specs` is the new truth regardless of what happens next.
        // Returning `Err` past this point would tell the worker's dispatch
        // to skip publishing the menu (see the `Job::Register` arm in
        // `Handle::spawn`) and tell the caller the registration failed,
        // while the store already disagrees with both. A disconnected
        // caller or an expired deadline are still real, but they're the
        // CALLER's problem (a `504`/dropped response) — never a reason to
        // un-happen a mutation that already happened.
        Ok(RegisterOutcome {
            entry,
            newly_loaded,
            evicted,
            load_ms: begin.elapsed().as_secs_f64() * 1000.,
            menu: self.menu(),
        })
    }

    fn unregister(&mut self, id: &str) -> UnregisterOutcome {
        match self.specs.remove(id) {
            RemoveOutcome::Removed(spec) => {
                let entry = spec.menu.clone();
                UnregisterOutcome::Deleted { entry, menu: self.menu() }
            }
            RemoveOutcome::Boot => UnregisterOutcome::BootSpec,
            RemoveOutcome::NotFound => UnregisterOutcome::NotFound,
        }
    }

    fn probe(
        &mut self,
        request: &crate::probe_api::ProbeRequest,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<crate::probe_api::ProbeResponse, Failure> {
        request.validate().map_err(Failure::BadRequest)?;
        self.probe_impl(request, check)
    }

    fn chat(
        &mut self,
        request: &crate::chat_session::ChatRequest,
        events: &dyn Fn(crate::chat_session::ChatEvent),
        at: &dyn YieldPoint<Self>,
    ) -> Result<crate::chat_session::ChatResponse, Failure> {
        request.validate().map_err(Failure::BadRequest)?;
        self.chat_turn(request, events, at)
    }
}

/// A checkpoint id naming nothing held: evicted, or never made here. Never a
/// rebuild from text, which would re-tokenize the generated turns.
fn unknown_checkpoint(id: &str) -> Failure {
    Failure::NotFound(format!(
        "no chat checkpoint {id:?}: it was evicted (least recently used, under \
         --chat-checkpoint-budget-mib) or never made by this daemon; start the chat again"
    ))
}

impl Adjudicator {
    /// The ids a held chat checkpoint stands for, without touching its
    /// recency: for tests and tools that check a chain of turns.
    pub fn chat_checkpoint_ids(&self, id: &str) -> Option<Vec<u32>> {
        self.chats.peek(id).map(|c| c.ids.clone())
    }

    /// Hold a chat checkpoint, charged its upper-bound bytes.
    fn hold_checkpoint(
        &mut self,
        id: String,
        checkpoint: Arc<crate::chat_session::ChatCheckpoint>,
    ) -> Result<(), Failure> {
        let bytes = self.state_size.bytes(checkpoint.ids.len()) + checkpoint_extra_bytes(checkpoint.ids.len());
        let evicted = self.chats.insert(id, bytes, checkpoint).map_err(Failure::Internal)?;
        if !evicted.is_empty() {
            tracing::info!(
                evicted = evicted.len(),
                held = self.chats.len(),
                used_bytes = self.chats.used(),
                budget_bytes = self.chats.budget(),
                "chat checkpoints evicted"
            );
        }
        Ok(())
    }

    /// One chat turn: see `crate::chat_session` for the checkpoints and the
    /// canonical schedule this keeps. Like `generate`, no borrow of `self`
    /// is held across a pause: the base checkpoint is an `Arc` clone, every
    /// state is owned, and checkpoints are held only once complete.
    fn chat_turn(
        &mut self,
        request: &crate::chat_session::ChatRequest,
        events: &dyn Fn(crate::chat_session::ChatEvent),
        at: &dyn YieldPoint<Self>,
    ) -> Result<crate::chat_session::ChatResponse, Failure> {
        use crate::chat_session::{ChatCheckpoint, ChatEvent, ChatResponse, CheckpointKind, checkpoint_id};
        at.check()?;
        let begin = Instant::now();
        let base = match &request.from {
            Some(id) => {
                let held = self.chats.get(id).ok_or_else(|| unknown_checkpoint(id))?;
                if held.kind != CheckpointKind::ChatTurn {
                    return Err(Failure::BadRequest(format!("checkpoint {id} does not end at a chat turn")));
                }
                Some(held)
            }
            None => None,
        };
        // The segments this turn appends, each rendered and encoded alone:
        // every one starts at `<|startoftext|>` or `<|im_start|>`, so it
        // encodes alone to the ids it has in place.
        let mut texts = Vec::with_capacity(request.messages.len() + 1);
        if base.is_none() {
            texts.push(
                crate::chat::render_head(
                    request.system.as_deref().unwrap_or(""),
                    request.tools.as_deref().unwrap_or(&[]),
                )
                .map_err(Failure::BadRequest)?,
            );
        }
        for message in &request.messages {
            texts.push(message.render().map_err(Failure::BadRequest)?);
        }
        let encode = |text: &str| encode_ids(&self.tokenizer, text).map_err(Failure::Internal);
        let segments: Vec<Vec<u32>> = texts.iter().map(|t| encode(t)).collect::<Result<_, _>>()?;
        let opening = encode(crate::chat::GENERATION_PROMPT)?;
        let after_eos = encode(crate::chat::AFTER_EOS)?;
        let mut ids = base.as_ref().map(|b| b.ids.clone()).unwrap_or_default();
        let mut text = base.as_ref().map(|b| b.text.clone()).unwrap_or_default();
        for (segment, segment_text) in segments.iter().zip(&texts) {
            ids.extend_from_slice(segment);
            text.push_str(segment_text);
        }
        let prompt_tokens = ids.len();
        if [opening.len(), request.max_tokens, after_eos.len()]
            .iter()
            .try_fold(prompt_tokens, |n, m| n.checked_add(*m))
            .is_none_or(|n| n > self.context_limit)
        {
            return Err(Failure::BadRequest(format!(
                "the chat ({prompt_tokens} tokens) plus the assistant's opening and max_tokens \
                 exceeds the {}-token context",
                self.context_limit
            )));
        }
        let user_id = checkpoint_id(&ids);
        let model = self.model.clone();
        let (user, cached_tokens) = match self.chats.get(&user_id) {
            // Held already: the canonical schedule would compute this very
            // state again, so there is nothing to compute.
            Some(held) => (held, prompt_tokens),
            None => {
                let (mut state, cached) = match &base {
                    Some(b) => (b.state.clone(), b.ids.len()),
                    None => (self.model.new_state(), 0),
                };
                let refs: Vec<&[u32]> = segments.iter().map(Vec::as_slice).collect();
                let logits = forward_segments(&model, &mut state, &refs, &mut || at.pause(self))?;
                logits.device().synchronize()?;
                at.check()?;
                let user = Arc::new(ChatCheckpoint {
                    kind: CheckpointKind::ChatTurn,
                    ids: ids.clone(),
                    text: text.clone(),
                    state,
                });
                // Complete: readers may fork it from here on, while the
                // assistant's turn is still being generated.
                self.hold_checkpoint(user_id.clone(), user.clone())?;
                (user, cached)
            }
        };
        drop(base);
        events(ChatEvent::Checkpoint {
            checkpoint_user: user_id.clone(),
            prompt_tokens,
            cached_tokens,
        });
        let mut state = user.state.clone();
        let mut logits = forward_segments(&model, &mut state, &[&opening], &mut || at.pause(self))?;
        let prefill_ms = begin.elapsed().as_secs_f64() * 1000.;
        let decode = Instant::now();
        let history: Vec<u32> = ids.iter().chain(&opening).copied().collect();
        let mut sampler = crate::constrain::Decoder::new(
            None,
            logits.device(),
            self.model.vocab_size(),
            &history,
            self.repeat_penalty,
        )?;
        let mut generated = Vec::new();
        let mut streamed = 0;
        let mut finish_reason = "length";
        for i in 0..request.max_tokens {
            at.pause(self)?;
            let token = sampler.sample(&logits)?;
            if self.tokenizer.id_to_token(token).is_none() {
                return Err(Failure::Internal(format!("model selected unused vocabulary row {token}")));
            }
            generated.push(token);
            if request.stream {
                let decoded = self
                    .tokenizer
                    .decode(&generated, false)
                    .map_err(|e| Failure::Internal(e.to_string()))?;
                let delta = crate::chat_session::text_delta(&decoded, &mut streamed).map_err(Failure::Internal)?;
                events(ChatEvent::Token { id: token, text: delta });
            }
            // `<|im_end|>` is forwarded too: the next turn continues after it.
            if token == self.eos {
                self.model.forward(&[token], &mut state)?;
                finish_reason = "stop";
                break;
            }
            if i + 1 < request.max_tokens {
                logits = self.model.forward(&[token], &mut state)?;
            }
        }
        at.check()?;
        let content_ids = generated.strip_suffix(&[self.eos]).unwrap_or(&generated);
        let turn_text = self
            .tokenizer
            .decode(content_ids, false)
            .map_err(|e| Failure::Internal(e.to_string()))?;
        let (thinking, content) = crate::chat_session::split_reasoning(&turn_text);
        let mut checkpoint = None;
        let mut checkpoint_tokens = None;
        if finish_reason == "stop" {
            let logits = forward_segments(&model, &mut state, &[&after_eos], &mut || at.pause(self))?;
            logits.device().synchronize()?;
            at.check()?;
            let mut all = history;
            all.extend_from_slice(&generated);
            all.extend_from_slice(&after_eos);
            let generated_text = self
                .tokenizer
                .decode(&generated, false)
                .map_err(|e| Failure::Internal(e.to_string()))?;
            let id = checkpoint_id(&all);
            checkpoint_tokens = Some(all.len());
            self.hold_checkpoint(
                id.clone(),
                Arc::new(ChatCheckpoint {
                    kind: CheckpointKind::ChatTurn,
                    ids: all,
                    text: format!(
                        "{}{}{generated_text}{}",
                        user.text,
                        crate::chat::GENERATION_PROMPT,
                        crate::chat::AFTER_EOS
                    ),
                    state,
                }),
            )?;
            checkpoint = Some(id);
        }
        Ok(ChatResponse {
            model: self.info(),
            checkpoint_user: user_id,
            checkpoint,
            text: turn_text,
            thinking,
            content,
            finish_reason: finish_reason.into(),
            prompt_tokens,
            cached_tokens,
            completion_tokens: generated.len(),
            checkpoint_tokens,
            queue_ms: 0.,
            prefill_ms,
            decode_ms: decode.elapsed().as_secs_f64() * 1000.,
        })
    }

    /// The F8 read primitive: the spec's fixed `opinion` question at its
    /// prefill, on `/v1/adjudicate` with `opinion: true`.
    fn opinion(
        &mut self,
        request: &AdjudicateRequest,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<AdjudicateResponse, Failure> {
        let key = request.spec_key().map_err(Failure::BadRequest)?;
        let spec_slot = self
            .specs
            .resolve_mut(key)
            .ok_or_else(|| unknown_spec(key))?;
        let spec = spec_slot.prompt.opinion.clone().ok_or_else(|| {
            Failure::BadRequest("this adjudicator's prompt spec asks no opinion question".into())
        })?;
        check()?;
        let rendered = format!(
            "{}{}",
            spec_slot.prefix_text,
            spec_slot
                .prompt
                .render_user_turn_with_prefill(&request.input, &spec.prefill)
                .map_err(Failure::BadRequest)?
        );
        // Load checked this split on a probe input; failing here means the
        // input changed the tokenization across the assistant boundary, which
        // the added tokens are supposed to make impossible.
        let crate::opinion::SplitOptions {
            shared: shared_ids,
            continuations,
        } = crate::opinion::split_options(&self.tokenizer, &rendered, &spec)
            .map_err(Failure::Internal)?;
        let shared = shared_ids.len();
        let longest = shared + continuations.iter().map(Vec::len).max().unwrap_or(0);
        if longest > spec_slot.info.context_limit {
            return Err(Failure::BadRequest(
                "prompt plus options exceeds adjudicator context".into(),
            ));
        }
        let begin = Instant::now();
        let PreparedEvaluation {
            state,
            logits,
            cached_tokens,
        } = spec_slot
            .cache
            .prepare(&self.model, &shared_ids, request.use_cache, check)?;
        let prefill_ms = begin.elapsed().as_secs_f64() * 1000.;
        let score = Instant::now();
        let candle_check = || check().map_err(|_| candle_core::Error::Msg("cancelled".into()));
        let scores = crate::opinion::score_continuations(
            &self.model,
            &state,
            &logits,
            &continuations,
            &candle_check,
        )
        .map_err(|e| {
            // Recover the precise failure: a cancellation surfaced through candle.
            check().err().unwrap_or(Failure::Internal(e.to_string()))
        })?;
        logits.device().synchronize()?;
        let decode_ms = score.elapsed().as_secs_f64() * 1000.;
        let read = read_options(&spec.options, &scores, &continuations, &logits)?;
        Ok(AdjudicateResponse {
            prefix: spec_slot.info.clone(),
            output: String::new(),
            report: None,
            report_error: None,
            finish_reason: "opinion".into(),
            prompt_tokens: shared,
            cached_tokens,
            completion_tokens: 0,
            queue_ms: 0.,
            prefill_ms,
            decode_ms,
            distributions: None,
            opinion: Some(crate::opinion::OpinionRead {
                options: read.options,
                sequence_mass: read.sequence_mass,
                first_token_mass: read.first_token_mass,
                shared_tokens: shared,
                scored_tokens: continuations.iter().map(Vec::len).sum(),
                rendered_sha256: sha256_hex_bytes(rendered.as_bytes()),
            }),
            resumed_tokens: None,
        })
    }

    /// `/v1/opinion`: generate the fields before the questions under the
    /// grammar, stop at each question's value slot as the walk passes it,
    /// score every option there. One description serves every question —
    /// they arrive resolved in emission order ([`SpecMenuEntry::resolve_all`]).
    /// See [`crate::opinion_api`] for why the description comes first and
    /// what the caches are.
    ///
    /// [`SpecMenuEntry::resolve_all`]: crate::opinion_api::SpecMenuEntry::resolve_all
    fn describe_then_read(
        &mut self,
        request: &crate::opinion_api::OpinionRequest,
        questions: &[crate::opinion_api::ResolvedQuestion],
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<crate::opinion_api::OpinionResponse, Failure> {
        use crate::opinion_api::{Answer, CacheOutcome, OpinionResponse};
        let last = questions
            .last()
            .ok_or_else(|| Failure::BadRequest("ask at least one question".into()))?;
        if questions
            .windows(2)
            .any(|w| w[0].describe.len() >= w[1].describe.len())
        {
            return Err(Failure::Internal(
                "questions reached the engine out of emission order".into(),
            ));
        }
        // Authoritative, regardless of what the handler's menu snapshot
        // believed: a spec evicted since that snapshot was taken is a clean
        // 404 here, never a silent fall-through to a different spec.
        let spec = self
            .specs
            .resolve_mut(request.spec.as_str())
            .ok_or_else(|| unknown_spec(request.spec.as_str()))?;
        let grammar = spec.grammar.clone().ok_or_else(|| {
            Failure::BadRequest(format!(
                "spec {:?} has no output schema, so there is nothing to describe or ask",
                request.spec
            ))
        })?;
        check()?;
        let state_text = request.state.render(&spec.prompt.input_label);
        // A tail read forks a held chat checkpoint (`crate::chat_session`);
        // the `Arc` keeps its state even if it is evicted meanwhile.
        let tail = match &request.context {
            None => None,
            Some(context) => {
                if !spec.prompt.tools.is_empty() {
                    return Err(Failure::BadRequest(format!(
                        "spec {:?} lists tools; a tools spec cannot read a chat tail (its read turn \
                         was never measured)",
                        request.spec
                    )));
                }
                let held = self
                    .chats
                    .get(&context.checkpoint)
                    .ok_or_else(|| unknown_checkpoint(&context.checkpoint))?;
                if held.kind != crate::chat_session::CheckpointKind::ChatTurn {
                    return Err(Failure::BadRequest(format!(
                        "checkpoint {} does not end at a chat turn",
                        context.checkpoint
                    )));
                }
                Some(held)
            }
        };
        // The prompt as text (what `rendered_sha256` covers), as ids, the
        // text option stability is checked against (it starts at a control
        // token, so the rest of the prompt cannot change its tokens), and a
        // tail read's segments after the checkpoint.
        let (prompt_text, prompt_ids, read_text, tail_segments) = match &tail {
            None => {
                let text = format!("{}{}", spec.prefix_text, spec.prompt.render_user_turn(&state_text));
                let ids = encode_ids(&self.tokenizer, &text).map_err(Failure::Internal)?;
                (text.clone(), ids, text, None)
            }
            Some(held) => {
                let block = spec
                    .prefix_text
                    .strip_prefix("<|startoftext|><|im_start|>system\n")
                    .and_then(|b| b.strip_suffix("<|im_end|>\n"))
                    .ok_or_else(|| Failure::Internal("a spec prefix is not one system turn".into()))?;
                // The read turn: the spec's instructions and schema, then the
                // state, as ONE user turn, then the closed reasoning region.
                let head = format!("<|im_start|>user\n{block}\n\n");
                let turn = spec.prompt.render_user_turn(&format!("{block}\n\n{state_text}"));
                debug_assert!(turn.starts_with(&head));
                let head_ids = encode_ids(&self.tokenizer, &head).map_err(Failure::Internal)?;
                let turn_ids = encode_ids(&self.tokenizer, &turn).map_err(Failure::Internal)?;
                // Two canonical segments, the head (the spec's, whatever the
                // state) and the rest, so a prefill of checkpoint + head
                // (planned background prefill) is the same computation.
                // The split needs the head's own tokens in place: a state
                // that starts with a newline merges into the blank line.
                if !turn_ids.starts_with(&head_ids) {
                    return Err(Failure::BadRequest(
                        "a tail read's state must not begin with a newline: it would merge with the \
                         blank line after the spec's instructions"
                            .into(),
                    ));
                }
                let mut ids = held.ids.clone();
                ids.extend_from_slice(&turn_ids);
                let at = held.ids.len();
                let split = at + head_ids.len();
                (format!("{}{turn}", held.text), ids, turn, Some((at, split)))
            }
        };
        let context_limit = spec.info.context_limit;
        if prompt_ids.len() + 2 >= context_limit {
            return Err(Failure::BadRequest(
                "prompt leaves no adjudicator context to describe the input".into(),
            ));
        }
        let mut cache = CacheOutcome {
            prefix: if request.use_cache { "hit" } else { "bypass" }.into(),
            state: "miss".into(),
            described: "miss".into(),
        };
        // A hit only when EVERY slot is held: a partial set would mix a
        // cached walk with a fresh one for no saving worth the bookkeeping.
        let described_hit: Option<Vec<_>> = if request.use_cache {
            questions
                .iter()
                .map(|q| spec.described.get(&prompt_ids, &q.field))
                .collect()
        } else {
            cache.described = "bypass".into();
            cache.state = "bypass".into();
            None
        };
        // One (generated, text, state, logits) per question, standing at its slot.
        let (slots, cached_tokens, prefill_ms, describe_ms) = match described_hit {
            Some(slots) => {
                cache.described = "hit".into();
                cache.state = "skipped".into();
                (slots, prompt_ids.len(), 0., 0.)
            }
            None => {
                let begin = Instant::now();
                let PreparedEvaluation {
                    mut state,
                    mut logits,
                    cached_tokens,
                } = match (&tail, tail_segments) {
                    (None, _) => spec
                        .cache
                        .prepare(&self.model, &prompt_ids, request.use_cache, check)?,
                    // Resumed: the checkpoint's state, then the read turn's
                    // two segments. Nothing is published: the read turn is
                    // this request's alone.
                    (Some(held), Some((at, split))) if request.use_cache => {
                        let mut state = held.state.clone();
                        let logits = forward_segments(
                            &self.model,
                            &mut state,
                            &[&prompt_ids[at..split], &prompt_ids[split..]],
                            &mut || check(),
                        )?;
                        logits.device().synchronize()?;
                        check()?;
                        PreparedEvaluation { state, logits, cached_tokens: at }
                    }
                    // Cold, as asked: the same bytes from token 0 in plain
                    // chunks. The instrument for how far a resumed read sits
                    // from a cold one; it still needs the checkpoint for
                    // the ids.
                    (Some(_), _) => {
                        let mut state = self.model.new_state();
                        let logits = forward_chunks(&self.model, &mut state, &prompt_ids, CHUNK, &mut || check())?;
                        logits.device().synchronize()?;
                        check()?;
                        PreparedEvaluation { state, logits, cached_tokens: 0 }
                    }
                };
                if request.use_cache && tail.is_none() && cached_tokens == prompt_ids.len() {
                    cache.state = "hit".into();
                }
                let prefill_ms = begin.elapsed().as_secs_f64() * 1000.;
                let describe = Instant::now();
                let mut sampler = crate::constrain::Decoder::new(
                    Some(&grammar),
                    logits.device(),
                    self.model.vocab_size(),
                    &prompt_ids,
                    self.repeat_penalty,
                )?;
                let mut generated = Vec::new();
                let mut text = String::new();
                let mut slots = Vec::with_capacity(questions.len());
                let mut slot_text = questions[0].slot_text();
                for _ in prompt_ids.len()..context_limit {
                    check()?;
                    let token = sampler.sample(&logits)?;
                    if self.tokenizer.id_to_token(token).is_none() {
                        return Err(Failure::Internal(format!(
                            "model selected unused vocabulary row {token}"
                        )));
                    }
                    if token == self.eos {
                        break;
                    }
                    generated.push(token);
                    let before = text.len();
                    text = self
                        .tokenizer
                        .decode(&generated, false)
                        .map_err(|e| Failure::Internal(e.to_string()))?;
                    logits = self.model.forward(&[token], &mut state)?;
                    if text.ends_with(&slot_text) {
                        // At this question's slot: keep the state, then walk
                        // on from the same logits to the next question's.
                        let question = &questions[slots.len()];
                        if request.use_cache {
                            spec.described.insert(DescribedEntry {
                                prompt_ids: prompt_ids.clone(),
                                field: question.field.clone(),
                                generated: generated.clone(),
                                text: text.clone(),
                                state: state.clone(),
                                logits: logits.clone(),
                            });
                        }
                        slots.push((generated.clone(), text.clone(), state.clone(), logits.clone()));
                        match questions.get(slots.len()) {
                            Some(next) => slot_text = next.slot_text(),
                            None => break,
                        }
                        continue;
                    }
                    // The grammar admits any token whose bytes begin the
                    // value, so one token can write the slot's closing
                    // quote AND the first bytes of an answer (`"a`) — the
                    // model leaving the canonical path at the slot. There
                    // is no slot to stand at then, and a read off this
                    // state would score the options on a path the model
                    // did not take. Refuse with the right diagnosis; the
                    // harness counts these as their own outcome.
                    if let Some(at) = text.rfind(&slot_text)
                        && at + slot_text.len() > before
                        && at + slot_text.len() < text.len()
                    {
                        return Err(Failure::Internal(format!(
                            "model wrote past the {:?} slot in one token ({:?}); no slot to read",
                            questions[slots.len()].field,
                            &text[before..]
                        )));
                    }
                }
                if slots.len() < questions.len() {
                    // The grammar makes every field reachable and required,
                    // so this is the context running out or the model
                    // ending the turn where the grammar forbids it — loud,
                    // never a read off the wrong slot.
                    return Err(Failure::Internal(format!(
                        "description ended before the {:?} slot after {} tokens: {text:?}",
                        questions[slots.len()].field,
                        generated.len()
                    )));
                }
                logits.device().synchronize()?;
                check()?;
                let describe_ms = describe.elapsed().as_secs_f64() * 1000.;
                (slots, cached_tokens, prefill_ms, describe_ms)
            }
        };
        let (last_generated, last_text, _, _) = slots.last().expect("one slot per question");
        let described = parse_described(last_text, &last.slot_text(), &last.describe)?;
        let read = Instant::now();
        let candle_check = || check().map_err(|_| candle_core::Error::Msg("cancelled".into()));
        let mut answers = Vec::with_capacity(questions.len());
        for (question, (generated, text, state, logits)) in questions.iter().zip(&slots) {
            // Each option and its close on their own pretokens after the
            // slot: checked at load on a probe, held to here.
            let mut continuations = Vec::with_capacity(question.options.len());
            for option in &question.options {
                let (alone, stable) = suffix_is_stable(
                    &self.tokenizer,
                    &format!("{read_text}{text}"),
                    &format!("{option}{}", question.close),
                )
                .map_err(Failure::Internal)?;
                if !stable {
                    return Err(Failure::Internal(format!(
                        "option {option:?} does not tokenize on its own after the {:?} slot",
                        question.field
                    )));
                }
                continuations.push(alone);
            }
            let scores = crate::opinion::score_continuations(
                &self.model,
                state,
                logits,
                &continuations,
                &candle_check,
            )
            .map_err(|e| check().err().unwrap_or(Failure::Internal(e.to_string())))?;
            logits.device().synchronize()?;
            let read = read_options(&question.options, &scores, &continuations, logits)?;
            let margin = crate::opinion_api::margin(
                &read.options.iter().map(|o| o.prob).collect::<Vec<_>>(),
            );
            answers.push(Answer {
                field: question.field.clone(),
                read: crate::opinion::OpinionRead {
                    options: read.options,
                    sequence_mass: read.sequence_mass,
                    first_token_mass: read.first_token_mass,
                    shared_tokens: prompt_ids.len() + generated.len(),
                    scored_tokens: continuations.iter().map(Vec::len).sum(),
                    rendered_sha256: sha256_hex_bytes(format!("{prompt_text}{text}").as_bytes()),
                },
                margin,
            });
        }
        let read_ms = read.elapsed().as_secs_f64() * 1000.;
        Ok(OpinionResponse {
            prefix: spec.info.clone(),
            spec: request.spec.clone(),
            context: request.context.clone(),
            described,
            answers,
            rendered: request.rendered.then(|| format!("{prompt_text}{last_text}")),
            rendered_token_ids: request.rendered.then(|| {
                let mut ids = prompt_ids.clone();
                ids.extend_from_slice(last_generated);
                ids
            }),
            cache,
            prompt_tokens: prompt_ids.len(),
            cached_tokens,
            described_tokens: last_generated.len(),
            queue_ms: 0.,
            prefill_ms,
            describe_ms,
            read_ms,
        })
    }

    /// `POST /v1/probe`: raw inference over exact text, not bound to any
    /// spec. See `crate::probe_api`'s module docs for the contract; this is
    /// the engine.
    fn probe_impl(
        &mut self,
        request: &crate::probe_api::ProbeRequest,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<crate::probe_api::ProbeResponse, Failure> {
        use crate::probe_api::{ProbeCache, ProbeContinuationScore, ProbeContinuations, ProbeGeneratedStep, ProbeIdentity, ProbeResponse};
        check()?;
        // Two input forms, per `ProbeRequest`'s text-form-caveat docs (F3,
        // kaibo review 2026-09-23): `ids` is exact — teacher-forced
        // verbatim, no tokenizer round-trip that could silently land on a
        // different token path than the one that produced them — and its
        // schedule split (`decode_from_token`) is already bounds-checked
        // by `ProbeRequest::validate` (a token index, not a byte offset,
        // has no "boundary" to search for). `text`/`messages` are
        // tokenized fresh, exact for bytes the CALLER wrote but not for
        // bytes containing a model's own generated output.
        let (full_ids, rendered, decode_from_k) = if let Some(ids) = &request.ids {
            let rendered = self.tokenizer.decode(ids, false).map_err(|e| Failure::Internal(e.to_string()))?;
            let decode_from_k = request.decode_from_token.unwrap_or(ids.len());
            (ids.clone(), rendered, decode_from_k)
        } else {
            let rendered = request.render();
            let encoding = self
                .tokenizer
                .encode(rendered.as_str(), false)
                .map_err(|e| Failure::Internal(e.to_string()))?;
            let full_ids = encoding.get_ids().to_vec();
            // `decode_from`: how many of `full_ids` are bulk-forwarded (the
            // rest replay the decode loop one token at a time — below).
            // `None` keeps every token in the bulk phase, today's only
            // schedule and still the default.
            let decode_from_k = resolve_decode_from(encoding.get_offsets(), request.decode_from)
                .map_err(Failure::BadRequest)?
                .unwrap_or(full_ids.len());
            (full_ids, rendered, decode_from_k)
        };
        let rendered_sha256 = sha256_hex_bytes(rendered.as_bytes());
        if full_ids.is_empty() {
            return Err(Failure::BadRequest("rendered input tokenizes to nothing".into()));
        }
        if full_ids.len() > self.context_limit {
            return Err(Failure::BadRequest("input tokens exceed the adjudicator context".into()));
        }
        if full_ids.len().checked_add(request.generate).is_none_or(|n| n > self.context_limit) {
            return Err(Failure::BadRequest(
                "input tokens plus generate exceeds the adjudicator context".into(),
            ));
        }

        // Warm-prefix resume applies to the BULK (prefill) portion only:
        // the LONGEST loaded spec (boot or uploaded) whose resident prefix
        // is a STRICT token-id prefix of it — `resolve_best_prefix_mut`
        // (deterministic regardless of iteration/LRU order, F2; a prefix
        // EQUAL to `prefill_ids` is not a candidate, so that case resumes
        // a shorter match or runs the bulk phase cold rather than erroring
        // "nothing to forward," F4 — both kaibo review, 2026-09-23), also
        // touching an uploaded winner to the back of the LRU exactly as a
        // request that named it by id would (F5). Cloned out of the found
        // spec immediately so this borrows only `self.specs`, not all of
        // `self` — `self.model` is a disjoint field borrowed right after
        // (same pattern `register`'s `CheckpointView` construction uses).
        let prefill_ids = &full_ids[..decode_from_k];
        let resumed = if request.use_cache {
            self.specs
                .resolve_best_prefix_mut(prefill_ids, |s| s.cache.prefix_ids.as_slice())
                .map(|s| (s.id.clone(), s.cache.prefix.clone(), s.cache.prefix_ids.len()))
        } else {
            None
        };
        let prefill_begin = Instant::now();
        let (mut state, cached_tokens, resumed_spec) = match resumed {
            Some((id, prefix_state, prefix_len)) => (prefix_state, prefix_len, Some(id)),
            None => (self.model.new_state(), 0, None),
        };
        let bulk_ids = &prefill_ids[cached_tokens..];
        let mut logits = if bulk_ids.is_empty() {
            None
        } else {
            Some(forward_chunks(&self.model, &mut state, bulk_ids, CHUNK, &mut || check())?)
        };
        let prefill_tokens = decode_from_k;

        // The stepwise replay: every token from `decode_from` onward,
        // forwarded ONE AT A TIME via `forward_chunks(..., 1, ...)` — the
        // exact `model.forward(&[token], &mut state)` call
        // `describe_then_read`'s decode loop makes per generated token,
        // reused rather than reimplemented (see `forward_chunks`'s docs).
        let stepwise_ids = &full_ids[decode_from_k..];
        if !stepwise_ids.is_empty() {
            logits = Some(forward_chunks(&self.model, &mut state, stepwise_ids, 1, &mut || check())?);
        }
        let stepwise_tokens = stepwise_ids.len();
        // Invariant, not a caller mistake: `full_ids` was already checked
        // non-empty above, and `resolve_best_prefix_mut` only ever returns
        // a STRICT prefix match (F4), so `bulk_ids` is non-empty whenever
        // `resumed` is `Some`. The only way `bulk_ids` is empty is
        // `decode_from_k == 0`, and then `stepwise_ids` IS `full_ids` —
        // non-empty by the same earlier check. So one of the two forwards
        // always ran; `Internal`, not `BadRequest`, if that ever stops
        // being true — a caller cannot construct a request that reaches
        // this, so it would mean the invariant above broke, not that the
        // request was bad.
        let logits = logits.ok_or_else(|| {
            Failure::Internal(
                "unreachable: neither the bulk nor the stepwise phase forwarded anything".into(),
            )
        })?;
        logits.device().synchronize()?;
        let prefill_ms = prefill_begin.elapsed().as_secs_f64() * 1000.;

        let score_begin = Instant::now();
        let candle_check = || check().map_err(|_| candle_core::Error::Msg("cancelled".into()));

        // Continuations, scored against the state/logits at the END of the
        // rendered input — this never advances `state` (see
        // `opinion::score_continuations_verbose`'s doc and test), so it can
        // run before the generate loop below without disturbing it.
        let continuations = if request.continuations.is_empty() {
            None
        } else {
            let mut ids = Vec::with_capacity(request.continuations.len());
            let mut canonical = Vec::with_capacity(request.continuations.len());
            for text in &request.continuations {
                let (alone, stable) =
                    suffix_is_stable(&self.tokenizer, &rendered, text).map_err(Failure::Internal)?;
                if alone.is_empty() {
                    return Err(Failure::BadRequest(format!(
                        "continuation {text:?} tokenizes to nothing"
                    )));
                }
                ids.push(alone);
                canonical.push(stable);
            }
            let longest = ids.iter().map(Vec::len).max().unwrap_or(0);
            if full_ids.len().checked_add(longest).is_none_or(|n| n > self.context_limit) {
                return Err(Failure::BadRequest(
                    "input plus the longest continuation exceeds the adjudicator context".into(),
                ));
            }
            let scores = crate::opinion::score_continuations_verbose(&self.model, &state, &logits, &ids, &candle_check)
                .map_err(|e| check().err().unwrap_or(Failure::Internal(e.to_string())))?;
            logits.device().synchronize()?;
            let sequence_mass =
                crate::opinion::logsumexp(&scores.iter().map(|s| s.sequence_logprob).collect::<Vec<_>>());
            let prob: Vec<f32> =
                scores.iter().map(|s| (s.sequence_logprob - sequence_mass).exp()).collect();
            let options = request
                .continuations
                .iter()
                .cloned()
                .zip(ids)
                .zip(scores)
                .zip(canonical)
                .map(|(((text, tokens), score), canonical)| ProbeContinuationScore {
                    text,
                    tokens,
                    first_logprob: *score
                        .token_logprobs
                        .first()
                        .expect("a stable, non-empty continuation encodes to at least one token"),
                    token_logprobs: score.token_logprobs,
                    sequence_logprob: score.sequence_logprob,
                    canonical,
                })
                .collect();
            Some(ProbeContinuations { options, sequence_mass, prob })
        };

        // Top-k at the last input position, and `generate` greedy steps
        // under the SAME sampling policy production uses (repetition
        // penalty over `full_ids`, no grammar). Step 0's distribution IS
        // "top-k at the last input position" — no extra forward pass is
        // spent computing it separately, since it is read straight off
        // `logits` the same way `generate`'s first step would be.
        let mut top_logprobs = Vec::new();
        let mut generated = Vec::new();
        if request.top_k > 0 || request.generate > 0 {
            let requested_steps = request.generate.max(1);
            let distreq = crate::types::DistributionRequest {
                top_k: request.top_k,
                token_sets: Default::default(),
                constrained: false,
            };
            let mut decoder = crate::constrain::Decoder::new(
                None,
                &self.execution.device,
                self.model.vocab_size(),
                &full_ids,
                self.repeat_penalty,
            )?;
            let mut cur_logits = logits.clone();
            for i in 0..requested_steps {
                check()?;
                let (token, dist) =
                    decoder.step(&cur_logits, Some(&distreq), |id| self.tokenizer.id_to_token(id))?;
                let dist = dist.expect("Some(&distreq) was passed, so Decoder::step always returns Some");
                if i == 0 {
                    top_logprobs = dist.top_logprobs.clone();
                }
                let is_eos = token == self.eos;
                if i < request.generate {
                    generated.push(ProbeGeneratedStep {
                        token,
                        text: dist.text.clone(),
                        logprob: dist.logprob,
                        top_logprobs: dist.top_logprobs,
                    });
                }
                if is_eos {
                    break;
                }
                if i + 1 < requested_steps {
                    cur_logits = self.model.forward(&[token], &mut state)?;
                }
            }
        }
        let score_ms = score_begin.elapsed().as_secs_f64() * 1000.;

        Ok(ProbeResponse {
            identity: ProbeIdentity {
                model_id: self.model_id.clone(),
                weight_hash: self.weight_hash.clone(),
                tokenizer_hash: self.tokenizer_hash.clone(),
                backend: self.execution.backend.as_str().into(),
                device: self.execution.identity.clone(),
                candle_rev: CANDLE_REV.into(),
                dtype: "f32".into(),
                sampling: format!("greedy; repetition_penalty={}; history=full", self.repeat_penalty),
            },
            rendered: (request.messages.is_some() || request.ids.is_some()).then(|| rendered.clone()),
            rendered_sha256,
            input_tokens: full_ids.len(),
            cache: ProbeCache {
                used_cache: cached_tokens > 0,
                resumed_spec,
                cached_tokens,
                prefill_tokens,
                stepwise_tokens,
            },
            top_logprobs,
            continuations,
            generated,
            queue_ms: 0.,
            prefill_ms,
            score_ms,
        })
    }
}

/// The scored options and the two masses, from raw sequence scores and the
/// slot's next-token logits.
struct ReadOptions {
    options: Vec<crate::opinion::OptionScore>,
    sequence_mass: f32,
    first_token_mass: f32,
}
fn read_options(
    names: &[String],
    scores: &[f32],
    continuations: &[Vec<u32>],
    logits: &Tensor,
) -> Result<ReadOptions, Failure> {
    let mass = crate::opinion::logsumexp(scores);
    let first = candle_nn::ops::log_softmax(logits, candle_core::D::Minus1)?
        .to_vec2::<f32>()?
        .remove(0);
    let first_ids: std::collections::BTreeSet<u32> =
        continuations.iter().map(|c| c[0]).collect();
    let first_token_mass = crate::opinion::logsumexp(
        &first_ids.iter().map(|&t| first[t as usize]).collect::<Vec<_>>(),
    );
    let options = names
        .iter()
        .zip(scores)
        .zip(continuations)
        .map(|((option, &logprob), tokens)| crate::opinion::OptionScore {
            option: option.clone(),
            logprob,
            first_logprob: first[tokens[0] as usize],
            prob: (logprob - mass).exp(),
            tokens: tokens.clone(),
        })
        .collect();
    Ok(ReadOptions {
        options,
        sequence_mass: mass,
        first_token_mass,
    })
}

/// The fields written before the slot, parsed from the generated text.
/// `text` ends with `slot_text` (the caller stopped there); what precedes it
/// is either the object's opening (first field asked) or complete fields and
/// the grammar's separator.
pub fn parse_described(
    text: &str,
    slot_text: &str,
    fields: &[String],
) -> Result<Vec<crate::opinion_api::DescribedField>, Failure> {
    let body = text.strip_suffix(slot_text).ok_or_else(|| {
        Failure::Internal(format!("generated text does not end at the slot: {text:?}"))
    })?;
    if fields.is_empty() {
        if body == "{" {
            return Ok(vec![]);
        }
        return Err(Failure::Internal(format!(
            "no field precedes the slot, but the model wrote {body:?} before it"
        )));
    }
    let body = body.strip_suffix(", ").ok_or_else(|| {
        Failure::Internal(format!("described fields do not end at a separator: {body:?}"))
    })?;
    let object: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&format!("{body}}}"))
            .map_err(|e| Failure::Internal(format!("described fields are not JSON: {e}: {body:?}")))?;
    if object.len() != fields.len() {
        return Err(Failure::Internal(format!(
            "described {} fields where the spec lists {} before the slot",
            object.len(),
            fields.len()
        )));
    }
    fields
        .iter()
        .map(|field| {
            object
                .get(field)
                .cloned()
                .map(|value| crate::opinion_api::DescribedField {
                    field: field.clone(),
                    value,
                })
                .ok_or_else(|| Failure::Internal(format!("described fields lack {field:?}")))
        })
        .collect()
}

/// Supported output schema: a closed object of required string/boolean fields.
/// Refuse unsupported constraints rather than pretending to enforce JSON Schema.
/// The schema's top-level keys, in the order the system prompt states them.
///
/// The model copies the stated schema's FIRST key into the report's first key
/// slot. Measured on ROCm over three inputs, `effect`'s probability there:
/// 0.36-0.58 under `serde_json`'s sorted order, where `additionalProperties`
/// comes first and won one row of three; 0.62-0.76 with `type` first; 0.86-0.95
/// with `properties` last. So the field list is stated last, nearest the slot
/// that copies it, and `properties` is also the only key whose own order
/// matters. Run record: `lfm25-think-prefill-2026-09-18/toplevel-*`
/// (author's private notes).
const SCHEMA_KEYS: [&str; 6] = [
    "type",
    "additionalProperties",
    "title",
    "description",
    "required",
    "properties",
];

/// Serialize the output schema in [`SCHEMA_KEYS`] order, `properties` in
/// `required` order.
///
/// `serde_json::Value` is backed by a `BTreeMap`, so the default rendering
/// sorts every object's keys, and arrays keep theirs — which is why `required`
/// was the only order that ever reached `crate::constrain`. The grammar emits
/// fields in `required` order, so the sorted rendering stated one order in the
/// system prompt and then masked the model into another: measured on LFM2.5 at
/// the second key slot, the model put 0.98 on `reason` and was forced to
/// `scope`.
///
/// Call it after `validate_schema`, which is what guarantees `required` and
/// `properties` name the same set and that no key falls outside
/// [`SCHEMA_KEYS`]. A schema that slipped past it is refused rather than
/// rendered with a key silently dropped.
fn render_schema(schema: &serde_json::Value) -> Result<String, String> {
    let object = schema
        .as_object()
        .ok_or("output_schema must be an object")?;
    let order = object
        .get("required")
        .and_then(|r| r.as_array())
        .ok_or("missing required field list")?
        .iter()
        .map(|name| name.as_str().ok_or("required field names must be strings"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = String::from("{");
    let mut written = 0;
    for key in SCHEMA_KEYS {
        let Some(value) = object.get(key) else {
            continue;
        };
        if written > 0 {
            out.push(',');
        }
        written += 1;
        out.push_str(&serde_json::to_string(key).map_err(|e| e.to_string())?);
        out.push(':');
        if key == "properties" {
            out.push_str(&render_properties(value, &order)?);
        } else {
            out.push_str(&serde_json::to_string(value).map_err(|e| e.to_string())?);
        }
    }
    if written != object.len() {
        return Err("unsupported output_schema keyword".into());
    }
    out.push('}');
    Ok(out)
}
fn render_properties(properties: &serde_json::Value, order: &[&str]) -> Result<String, String> {
    let properties = properties
        .as_object()
        .ok_or("missing schema properties")?;
    if properties.len() != order.len() {
        return Err("every property must be required exactly once".into());
    }
    let mut out = String::from("{");
    for (i, name) in order.iter().enumerate() {
        let spec = properties
            .get(*name)
            .ok_or("required names a property that does not exist")?;
        if i > 0 {
            out.push(',');
        }
        out.push_str(&serde_json::to_string(name).map_err(|e| e.to_string())?);
        out.push(':');
        out.push_str(&serde_json::to_string(spec).map_err(|e| e.to_string())?);
    }
    out.push('}');
    Ok(out)
}
fn validate_schema(schema: &serde_json::Value) -> Result<(), String> {
    let object = schema
        .as_object()
        .ok_or("output_schema must be an object")?;
    if object.keys().any(|k| {
        ![
            "type",
            "properties",
            "required",
            "additionalProperties",
            "description",
            "title",
        ]
        .contains(&k.as_str())
    }) {
        return Err("unsupported output_schema keyword".into());
    }
    if schema["type"] != "object" || schema["additionalProperties"] != false {
        return Err("output_schema must declare object and additionalProperties=false".into());
    }
    let props = schema["properties"]
        .as_object()
        .ok_or("missing schema properties")?;
    let required = schema["required"]
        .as_array()
        .ok_or("missing required field list")?;
    let names: std::collections::BTreeSet<_> = required.iter().filter_map(|v| v.as_str()).collect();
    if props.is_empty()
        || names.len() != required.len()
        || names.len() != props.len()
        || props.keys().any(|k| !names.contains(k.as_str()))
    {
        return Err("every property must be required exactly once".into());
    }
    for spec in props.values() {
        let obj = spec.as_object().ok_or("invalid property schema")?;
        if obj
            .keys()
            .any(|k| !["type", "enum", "description", "title"].contains(&k.as_str()))
        {
            return Err("unsupported property constraint".into());
        }
        let string = spec["type"] == "string";
        if !string && spec["type"] != "boolean" {
            return Err("report properties must be string or boolean".into());
        }
        if let Some(choices) = spec.get("enum") {
            let choices = choices.as_array().ok_or("enum must be an array")?;
            if choices.is_empty()
                || choices.iter().any(|v| {
                    if string {
                        !v.is_string()
                    } else {
                        !v.is_boolean()
                    }
                })
            {
                return Err("invalid enum values".into());
            }
        }
    }
    Ok(())
}

// serde_json::Value normally keeps the last duplicate key. Reports are flat,
// so reject duplicate top-level fields before any schema/semantic validation.
struct ReportObject(serde_json::Map<String, serde_json::Value>);
impl<'de> Deserialize<'de> for ReportObject {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = ReportObject;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("one report object without duplicate fields")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut out = serde_json::Map::new();
                while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
                    if out.insert(key, value).is_some() {
                        return Err(serde::de::Error::custom("duplicate report field"));
                    }
                }
                Ok(ReportObject(out))
            }
        }
        de.deserialize_map(Visitor)
    }
}
/// Validate a completed report. The whole completion is the document: reasoning
/// regions, fences, trailing prose, duplicate fields, truncated output, and
/// coercions are never accepted.
///
/// This used to strip a leading `<think>...</think>`. That branch was
/// unreachable by construction: this function only runs under an
/// `output_schema`, which is the same expression that compiles the grammar, and
/// the grammar's first legal byte is the object's `{`. (`<think>` is also an
/// added token, which `constrain::Vocabulary` masks unconditionally — but that
/// is the weaker argument, since it would still be admissible inside a string
/// value.) The reasoning region
/// the model expects is supplied already closed by the prompt instead
/// ([`Reasoning`]), so a completion that still carried one would be something
/// gone wrong, and is now reported rather than quietly stripped.
pub fn validate_report(
    output: &str,
    schema: &serde_json::Value,
    finish: &str,
) -> Result<serde_json::Value, String> {
    validate_schema(schema)?;
    if finish != "stop" {
        return Err("generation was truncated; no report returned".into());
    }
    let ReportObject(report) =
        serde_json::from_str(output.trim()).map_err(|e| format!("invalid report JSON: {e}"))?;
    let properties = schema["properties"]
        .as_object()
        .ok_or("invalid report schema")?;
    if report.len() != properties.len() || properties.keys().any(|k| !report.contains_key(k)) {
        return Err("report has missing or additional fields".into());
    }
    for (name, spec) in properties {
        let value = &report[name];
        let valid = if spec["type"] == "string" {
            value.as_str().is_some_and(|s| !s.trim().is_empty())
        } else {
            value.is_boolean()
        };
        if !valid {
            return Err(format!(
                "report field {name} has an invalid type or empty value"
            ));
        }
        if let Some(choices) = spec.get("enum").and_then(|v| v.as_array())
            && !choices.contains(value)
        {
            return Err(format!("report field {name} is outside its enum"));
        }
    }
    Ok(serde_json::Value::Object(report))
}

/// One unit of work for the single generation worker: a generation or an
/// opinion read. Both share the queue, the deadline and the cancellation
/// path; only the generator call differs.
/// Neither registration nor deletion carries a client-chosen timeout (the
/// upload body is raw spec bytes, no request envelope) — this bounds both.
/// Generous for registration: a genuine load runs the prefix through the
/// model, like startup does; deletion is a map removal and returns long
/// before this ever matters.
const SPEC_ADMIN_TIMEOUT_MS: u64 = 60_000;
/// A spec's system prompt, tools and schema are text a person writes and a
/// daemon renders once at load, not a checkpoint — this is generous headroom
/// over any shipped spec, not a tuned limit.
const MAX_SPEC_BYTES: usize = 1_048_576;

/// A job's scheduling class. Each class has its own bounded queue
/// ([`QUEUE_DEPTH`] deep). The worker picks the highest class with work
/// waiting, and at a running job's [`YieldPoint::pause`] it serves every
/// pending job of a strictly HIGHER class to completion before the paused
/// job goes on. Within a class, jobs keep their arrival order and never
/// overtake one another.
///
/// A higher class goes first without limit: a steady stream of reads
/// postpones a waiting generation until the stream stops or the
/// generation's own deadline passes. Reads are short, and that is the
/// point of the order; aging is the lever if it ever starves generation.
///
/// Adding a class is adding a variant here, a place in [`Priority::ALL`],
/// and a `Job::priority` arm; the queues, the pick and the pause all follow
/// from `ALL`. Background prefill (a class below `Generative`, pausing like
/// a generation does) is the one planned (`docs/chat-tail-plan.md`, piece
/// 6). It must keep one invariant this order gives today for free: nothing
/// served at a pause removes a spec. Its pauses would serve `Generative`,
/// which holds registration and deletion, so either those move to a class
/// that is never served at a pause, or background prefill re-resolves its
/// spec after its pauses and treats a vanished spec as nothing to publish.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Priority {
    /// Generative adjudication, and spec registration and deletion. The
    /// admin jobs sit here, not higher, because they remove specs: served at
    /// a pause, a deletion or an eviction would pull a spec out from under
    /// the paused generation, which publishes into that spec's cache when
    /// its prefill completes.
    Generative,
    /// Opinion reads (`/v1/opinion`, and `/v1/adjudicate` with `opinion:
    /// true`) and probes: short, and they never pause, so one never waits
    /// behind another's pause.
    Interactive,
}
impl Priority {
    /// Every class, lowest first. A class's position is its queue's index.
    const ALL: [Priority; 2] = [Priority::Generative, Priority::Interactive];
    fn index(self) -> usize {
        Priority::ALL
            .iter()
            .position(|p| *p == self)
            .expect("every class is listed in Priority::ALL")
    }
}
/// How many jobs each class's queue holds before `submit` answers
/// `503 overloaded`.
const QUEUE_DEPTH: usize = 8;

enum Job {
    Adjudicate(AdjudicateRequest),
    Opinion(
        crate::opinion_api::OpinionRequest,
        Vec<crate::opinion_api::ResolvedQuestion>,
    ),
    Register(String, PromptSpec),
    Unregister(String),
    Probe(crate::probe_api::ProbeRequest),
    /// A chat turn, and where a streaming caller hears its progress.
    Chat(
        crate::chat_session::ChatRequest,
        Option<tokio::sync::mpsc::UnboundedSender<crate::chat_session::ChatEvent>>,
    ),
}
impl Job {
    fn timeout_ms(&self) -> u64 {
        match self {
            Job::Adjudicate(r) => r.timeout_ms,
            Job::Opinion(r, _) => r.timeout_ms,
            Job::Register(..) | Job::Unregister(_) => SPEC_ADMIN_TIMEOUT_MS,
            Job::Probe(r) => r.timeout_ms,
            Job::Chat(r, _) => r.timeout_ms,
        }
    }
    /// Which queue this job waits in: see [`Priority`].
    fn priority(&self) -> Priority {
        match self {
            Job::Adjudicate(r) if r.opinion => Priority::Interactive,
            Job::Opinion(..) | Job::Probe(_) => Priority::Interactive,
            Job::Adjudicate(_) | Job::Chat(..) | Job::Register(..) | Job::Unregister(_) => {
                Priority::Generative
            }
        }
    }
    fn operation(&self) -> &'static str {
        match self {
            Job::Adjudicate(_) => "adjudicate",
            Job::Opinion(..) => "opinion",
            Job::Register(..) => "register",
            Job::Unregister(_) => "unregister",
            Job::Probe(_) => "probe",
            Job::Chat(..) => "chat",
        }
    }
}
enum Reply {
    Adjudicate(AdjudicateResponse),
    Opinion(crate::opinion_api::OpinionResponse),
    Register(RegisterOutcome),
    Unregister(UnregisterOutcome),
    Probe(crate::probe_api::ProbeResponse),
    Chat(crate::chat_session::ChatResponse),
}
struct Work {
    job: Job,
    reply: oneshot::Sender<Result<Reply, Failure>>,
    enqueued: Instant,
    span: tracing::Span,
}
/// The worker thread's side of the queues: it owns the generator (passed
/// in, never stored) and runs one job at a time on it.
struct Worker {
    queues: Vec<crossbeam_channel::Receiver<Work>>,
    stopping: Arc<AtomicBool>,
    menu: Arc<std::sync::RwLock<Vec<crate::opinion_api::SpecMenuEntry>>>,
}
impl Worker {
    /// Run jobs, highest class first, until every `Handle` is gone and
    /// every queue is drained — what `while let Ok(work) = rx.recv()` did
    /// over the single queue this replaces: work already queued when the
    /// last `Handle` drops still runs (and, once stopping, is refused by
    /// its own check).
    fn serve<G: Generator>(&self, generator: &mut G) {
        let mut open = vec![true; self.queues.len()];
        loop {
            let mut next = None;
            for class in Priority::ALL.iter().rev() {
                let i = class.index();
                if !open[i] {
                    continue;
                }
                match self.queues[i].try_recv() {
                    Ok(work) => {
                        next = Some(work);
                        break;
                    }
                    // All senders gone AND nothing left: closed for good.
                    Err(crossbeam_channel::TryRecvError::Disconnected) => open[i] = false,
                    Err(crossbeam_channel::TryRecvError::Empty) => {}
                }
            }
            if let Some(work) = next {
                self.run(generator, work);
                continue;
            }
            if !open.contains(&true) {
                return;
            }
            // Nothing waiting: sleep until some open queue has work or
            // closes, then pick again by class.
            let mut select = crossbeam_channel::Select::new();
            for (queue, _) in self.queues.iter().zip(&open).filter(|(_, open)| **open) {
                select.recv(queue);
            }
            select.ready();
        }
    }
    /// The next waiting job of a class strictly above `floor`, highest
    /// first, without blocking.
    fn next_above(&self, floor: Priority) -> Option<Work> {
        Priority::ALL
            .iter()
            .rev()
            .take_while(|class| **class > floor)
            .find_map(|class| self.queues[class.index()].try_recv().ok())
    }
    /// One job, start to finish (pauses included), and its reply.
    fn run<G: Generator>(&self, generator: &mut G, work: Work) {
        let deadline = work.enqueued + Duration::from_millis(work.job.timeout_ms());
        let check = || {
            if work.reply.is_closed() || self.stopping.load(Ordering::SeqCst) {
                Err(Failure::Cancelled)
            } else if Instant::now() >= deadline {
                Err(Failure::Deadline)
            } else {
                Ok(())
            }
        };
        let pause = Pause {
            worker: self,
            class: work.job.priority(),
            check: &check,
            served: std::cell::Cell::new(0),
            paused: std::cell::Cell::new(Duration::ZERO),
        };
        let _span = work.span.enter();
        let queue_ms = work.enqueued.elapsed().as_secs_f64() * 1000.;
        let operation = work.job.operation();
        let start = Instant::now();
            let result = match &work.job {
                Job::Adjudicate(request) => check()
                    .and_then(|()| generator.generate(request, &pause))
                    .map(|mut r| {
                        r.queue_ms = queue_ms;
                        tracing::info!(
                            cached_tokens = r.cached_tokens,
                            prompt_tokens = r.prompt_tokens,
                            completion_tokens = r.completion_tokens,
                            prefill_ms = r.prefill_ms,
                            decode_ms = r.decode_ms,
                            "adjudication complete"
                        );
                        Reply::Adjudicate(r)
                    }),
                Job::Opinion(request, questions) => check()
                    .and_then(|()| generator.opine(request, questions, &check))
                    .map(|mut r| {
                        r.queue_ms = queue_ms;
                        tracing::info!(
                            spec = %r.spec,
                            // One key whatever the count, so a
                            // single-question query still matches.
                            field = %questions
                                .iter()
                                .map(|q| q.field.as_str())
                                .collect::<Vec<_>>()
                                .join(","),
                            cached_tokens = r.cached_tokens,
                            prompt_tokens = r.prompt_tokens,
                            described_tokens = r.described_tokens,
                            described_cache = %r.cache.described,
                            prefill_ms = r.prefill_ms,
                            describe_ms = r.describe_ms,
                            read_ms = r.read_ms,
                            "opinion read complete"
                        );
                        Reply::Opinion(r)
                    }),
                // Register/Unregister publish `worker_menu` HERE, on
                // this thread, unconditionally once `generator`
                // reports the store changed — never deferred to the
                // async task that submitted the job. That task's
                // oneshot receiver can already be gone by the time
                // we get here (the caller disconnected, or hit its
                // own client-side deadline) and `work.reply.send`
                // below is allowed to fail silently for exactly
                // that reason; if publishing waited for a
                // successful reply, a disconnected caller's
                // registration would sit in the store forever
                // without ever reaching `Handle.menu`, and a caller
                // whose reply raced another admin request could
                // publish an older snapshot last. Publishing here,
                // on the single serial worker thread, in the same
                // order jobs are dequeued, closes both: by the time
                // ANY reply is sent (or not), the store and the
                // published menu already agree.
                Job::Register(id, prompt) => check()
                    .and_then(|()| generator.register(id.clone(), prompt.clone(), &check))
                    .map(|r| {
                        *self.menu.write().expect("menu lock poisoned") = r.menu.clone();
                        tracing::info!(
                            id = %r.entry.id,
                            snapshot_id = %r.entry.snapshot_id,
                            newly_loaded = r.newly_loaded,
                            evicted = ?r.evicted,
                            load_ms = r.load_ms,
                            "opinion spec registered"
                        );
                        Reply::Register(r)
                    }),
                Job::Unregister(id) => check().map(|()| {
                    let outcome = generator.unregister(id);
                    match &outcome {
                        UnregisterOutcome::Deleted { entry, menu } => {
                            *self.menu.write().expect("menu lock poisoned") = menu.clone();
                            tracing::info!(
                                id = %entry.id,
                                snapshot_id = %entry.snapshot_id,
                                "opinion spec deleted"
                            );
                        }
                        UnregisterOutcome::BootSpec => {
                            tracing::warn!(
                                id = %id,
                                "refused to delete a boot-time opinion spec"
                            );
                        }
                        UnregisterOutcome::NotFound => {
                            tracing::info!(
                                id = %id,
                                "delete requested for an unknown opinion spec"
                            );
                        }
                    }
                    Reply::Unregister(outcome)
                }),
                Job::Probe(request) => check()
                    .and_then(|()| generator.probe(request, &check))
                    .map(|mut r| {
                        r.queue_ms = queue_ms;
                        tracing::info!(
                            input_tokens = r.input_tokens,
                            used_cache = r.cache.used_cache,
                            cached_tokens = r.cache.cached_tokens,
                            continuations = r.continuations.as_ref().map(|c| c.options.len()).unwrap_or(0),
                            generated = r.generated.len(),
                            prefill_ms = r.prefill_ms,
                            score_ms = r.score_ms,
                            "probe complete"
                        );
                        Reply::Probe(r)
                    }),
                Job::Chat(request, events) => {
                    // A closed stream is the streaming caller gone: its
                    // handler drops the reply then, which `check` sees.
                    let send = |event| {
                        if let Some(events) = events {
                            let _ = events.send(event);
                        }
                    };
                    check()
                        .and_then(|()| generator.chat(request, &send, &pause))
                        .map(|mut r| {
                            r.queue_ms = queue_ms;
                            tracing::info!(
                                interleaved = pause.served.get(),
                                interleaved_ms = pause.paused.get().as_secs_f64() * 1000.,
                                prompt_tokens = r.prompt_tokens,
                                cached_tokens = r.cached_tokens,
                                completion_tokens = r.completion_tokens,
                                finish_reason = %r.finish_reason,
                                prefill_ms = r.prefill_ms,
                                decode_ms = r.decode_ms,
                                "chat turn complete"
                            );
                            Reply::Chat(r)
                        })
                }
            };
            if let Err(e) = &result {
                let kind = match e {
                    Failure::BadRequest(_) => "bad_request",
                    Failure::NotFound(_) => "not_found",
                    Failure::Unprocessable(_) => "unprocessable",
                    Failure::Forbidden(_) => "forbidden",
                    Failure::Internal(_) => "internal",
                    Failure::Cancelled => "cancelled",
                    Failure::Deadline => "deadline",
                };
                tracing::warn!(error_kind = kind, operation, "adjudicator work failed");
            }
            // The jobs served at this job's pauses recorded their own durations.
        crate::telemetry::record_inference_duration(operation, start.elapsed() - pause.paused.get());
            let _ = work.reply.send(result);
    }
}

/// The [`YieldPoint`] the worker hands a running job: its own `check`, and
/// at `pause` the jobs of a higher class, served nested on the same thread
/// and the same generator.
struct Pause<'a> {
    worker: &'a Worker,
    class: Priority,
    check: &'a dyn Fn() -> Result<(), Failure>,
    /// How many jobs ran at this job's pauses, and for how long: logged
    /// with the job, and kept out of its own inference duration.
    served: std::cell::Cell<usize>,
    paused: std::cell::Cell<Duration>,
}
impl<G: Generator> YieldPoint<G> for Pause<'_> {
    fn check(&self) -> Result<(), Failure> {
        (self.check)()
    }
    fn pause(&self, generator: &mut G) -> Result<(), Failure> {
        let begin = Instant::now();
        let mut served = 0;
        // Checked before, between and after the jobs served here: a paused
        // job that is cancelled or out of time stops at the next boundary,
        // not once the higher queues run dry, and what is still waiting
        // goes to the worker's next pick.
        let result = loop {
            if let Err(e) = (self.check)() {
                break Err(e);
            }
            let Some(work) = self.worker.next_above(self.class) else {
                break Ok(());
            };
            self.worker.run(generator, work);
            served += 1;
        };
        if served > 0 {
            self.served.set(self.served.get() + served);
            self.paused.set(self.paused.get() + begin.elapsed());
        }
        result
    }
}

#[derive(Clone)]
pub struct Handle {
    /// One queue per [`Priority`], indexed by [`Priority::index`].
    queues: Vec<crossbeam_channel::Sender<Work>>,
    info: AdjudicatorInfo,
    /// What `/v1/opinion` may be asked: read off the loaded specs, so a
    /// question is refused at the handler and never queued. A `RwLock`
    /// (not a bare `Arc<Vec<..>>`) because registration and eviction update
    /// it live — every `Handle` clone shares this lock, so a redeploy of
    /// this daemon's live menu is never needed to see an upload. The
    /// WORKER stays authoritative regardless: this is a fast pre-check
    /// only, and a request that slips past a stale view still gets a clean
    /// 404 from the worker rather than a wrong spec (see `describe_then_read`
    /// and `Adjudicator::opinion`'s own lookups).
    menu: Arc<std::sync::RwLock<Vec<crate::opinion_api::SpecMenuEntry>>>,
    exit: WorkerExit,
    stopping: Arc<AtomicBool>,
}
impl Handle {
    pub fn spawn<G: Generator>(mut generator: G, info: AdjudicatorInfo) -> Self {
        let (queues, receivers): (Vec<_>, Vec<_>) =
            Priority::ALL.iter().map(|_| crossbeam_channel::bounded::<Work>(QUEUE_DEPTH)).unzip();
        let exit = WorkerExit::default();
        let finished = exit.clone();
        let stopping = Arc::new(AtomicBool::new(false));
        // Created here, before the worker thread starts, and cloned into
        // it: the WORKER publishes every registration/eviction/delete to
        // this directly, synchronously, as part of processing that job —
        // never the async request task that happened to submit it. See the
        // long comment on the `Job::Register`/`Job::Unregister` arms in
        // `Worker::run` for why that distinction is the whole fix.
        let menu = Arc::new(std::sync::RwLock::new(Vec::new()));
        let worker = Worker {
            queues: receivers,
            stopping: stopping.clone(),
            menu: menu.clone(),
        };
        let worker = std::thread::Builder::new()
            .name("lfm2d-adjudicator".into())
            .spawn(move || {
                worker.serve(&mut generator);
                drop(generator);
                finished.mark();
            })
            .expect("spawn adjudicator worker");
        std::thread::spawn(move || {
            if worker.join().is_err() {
                eprintln!("lfm2d: adjudicator worker panicked");
                std::process::exit(1);
            }
        });
        Self { queues, info, menu, exit, stopping }
    }
    /// The specs `/v1/opinion` serves at boot. Without this, every opinion
    /// request is refused as naming an unknown spec. Registration and
    /// deletion keep this current afterward — the WORKER publishes it
    /// directly (see the `Job::Register`/`Job::Unregister` arms in
    /// `spawn`), never a request task, so this is the only caller of
    /// [`Handle::replace_menu`] left.
    pub fn with_menu(self, menu: Vec<crate::opinion_api::SpecMenuEntry>) -> Self {
        self.replace_menu(menu);
        self
    }
    pub fn menu(&self) -> Vec<crate::opinion_api::SpecMenuEntry> {
        self.menu.read().expect("menu lock poisoned").clone()
    }
    /// Overwrite the live menu wholesale. Boot-time setup only
    /// (`with_menu`) — a registration or deletion publishes through the
    /// worker's own clone of this `Arc`, not through this method, so it
    /// happens exactly once, synchronously, on the thread that made the
    /// mutation, regardless of whether the request that triggered it is
    /// still around to hear back.
    fn replace_menu(&self, menu: Vec<crate::opinion_api::SpecMenuEntry>) {
        *self.menu.write().expect("menu lock poisoned") = menu;
    }
    pub fn stop_signal(&self) -> Arc<AtomicBool> {
        self.stopping.clone()
    }
    pub fn exit_signal(&self) -> WorkerExit {
        self.exit.clone()
    }
    // The Err is an axum `Response`, the shape a handler hands back; boxing it
    // to please the size heuristic would cost a heap allocation per refusal.
    #[allow(clippy::result_large_err)]
    async fn submit(&self, job: Job, span: &'static str) -> Result<Reply, Response> {
        let (rx, timeout) = self.enqueue(job, span)?;
        Self::reply(rx, timeout).await.map_err(IntoResponse::into_response)
    }
    /// Queue `job`, or refuse it as `submit` does (stopping, overloaded,
    /// worker gone): the part a streaming response must finish before it
    /// commits to a `200`.
    #[allow(clippy::result_large_err)]
    fn enqueue(
        &self,
        job: Job,
        span: &'static str,
    ) -> Result<(oneshot::Receiver<Result<Reply, Failure>>, Duration), Response> {
        if self.stopping.load(Ordering::SeqCst) {
            return Err(Failure::Cancelled.into_response());
        }
        let (reply, rx) = oneshot::channel();
        let timeout = Duration::from_millis(job.timeout_ms());
        let work = Work {
            job,
            reply,
            enqueued: Instant::now(),
            span: tracing::info_span!("adjudicator", operation = span),
        };
        self.queues[work.job.priority().index()].try_send(work).map_err(|e|match e {
            crossbeam_channel::TrySendError::Full(_)=>(StatusCode::SERVICE_UNAVAILABLE,Json(serde_json::json!({"error":{"type":"overloaded","message":"adjudicator queue is full"}}))).into_response(),
            crossbeam_channel::TrySendError::Disconnected(_)=>Failure::Internal("adjudicator worker unavailable".into()).into_response(),
        })?;
        Ok((rx, timeout))
    }
    /// The worker's answer, or `Deadline` once the job's own timeout passes.
    async fn reply(
        rx: oneshot::Receiver<Result<Reply, Failure>>,
        timeout: Duration,
    ) -> Result<Reply, Failure> {
        match tokio::time::timeout(timeout, rx).await {
            Err(_) => Err(Failure::Deadline),
            Ok(Err(_)) => Err(Failure::Internal("adjudicator worker dropped response".into())),
            Ok(Ok(result)) => result,
        }
    }
    /// `POST /v1/chat` answering with one JSON body.
    #[allow(clippy::result_large_err)]
    pub async fn chat(
        &self,
        request: crate::chat_session::ChatRequest,
    ) -> Result<crate::chat_session::ChatResponse, Response> {
        request
            .validate()
            .map_err(|e| Failure::BadRequest(e).into_response())?;
        match self.submit(Job::Chat(request, None), "chat").await? {
            Reply::Chat(r) => Ok(r),
            _ => Err(Failure::Internal("worker answered a chat with something else".into()).into_response()),
        }
    }
    /// `POST /v1/chat` with `stream: true`: server-sent events. A request
    /// the worker would refuse before starting (malformed, overloaded,
    /// stopping) is refused with its HTTP status here; once the `200` is
    /// sent, a failure is an `error` event carrying the status and the
    /// error body. A client that disconnects cancels the turn: dropping the
    /// reply is what the worker's check sees.
    #[allow(clippy::result_large_err)]
    pub fn chat_stream(&self, request: crate::chat_session::ChatRequest) -> Result<Response, Response> {
        use axum::response::sse::{Event, KeepAlive, Sse};
        request
            .validate()
            .map_err(|e| Failure::BadRequest(e).into_response())?;
        let (events_tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel();
        let (rx, timeout) = self.enqueue(Job::Chat(request, Some(events_tx)), "chat")?;
        let (sse_tx, sse_rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, std::convert::Infallible>>();
        let event = |e: &crate::chat_session::ChatEvent| {
            let name = match e {
                crate::chat_session::ChatEvent::Checkpoint { .. } => "checkpoint",
                crate::chat_session::ChatEvent::Token { .. } => "token",
            };
            Event::default().event(name).json_data(e).expect("a chat event always serializes")
        };
        tokio::spawn(async move {
            let reply = Self::reply(rx, timeout);
            tokio::pin!(reply);
            let last = loop {
                tokio::select! {
                    Some(e) = events_rx.recv() => {
                        if sse_tx.send(Ok(event(&e))).is_err() {
                            return;
                        }
                    }
                    result = &mut reply => break result,
                    // The client is gone: returning drops the reply.
                    () = sse_tx.closed() => return,
                }
            };
            // Every event the worker sent precedes its reply.
            while let Ok(e) = events_rx.try_recv() {
                let _ = sse_tx.send(Ok(event(&e)));
            }
            let last = match last {
                Ok(Reply::Chat(r)) => Event::default().event("done").json_data(&r),
                Ok(_) => Event::default().event("error").json_data(
                    Failure::Internal("worker answered a chat with something else".into()).parts().1,
                ),
                Err(failure) => {
                    let (status, mut body) = failure.parts();
                    body["status"] = status.as_u16().into();
                    Event::default().event("error").json_data(body)
                }
            };
            let _ = sse_tx.send(Ok(last.expect("a chat response always serializes")));
        });
        Ok(Sse::new(tokio_stream::wrappers::UnboundedReceiverStream::new(sse_rx))
            .keep_alive(KeepAlive::default())
            .into_response())
    }
    #[allow(clippy::result_large_err)]
    pub async fn evaluate(
        &self,
        request: AdjudicateRequest,
    ) -> Result<AdjudicateResponse, Response> {
        request
            .validate()
            .map_err(|e| Failure::BadRequest(e).into_response())?;
        match self.submit(Job::Adjudicate(request), "adjudicate").await? {
            Reply::Adjudicate(r) => Ok(r),
            _ => Err(
                Failure::Internal("worker answered a generation with something else".into())
                    .into_response(),
            ),
        }
    }
    /// Validate against the menu here, so a question the daemon cannot ask
    /// is a 400 that never occupies the worker.
    #[allow(clippy::result_large_err)]
    pub async fn opine(
        &self,
        request: crate::opinion_api::OpinionRequest,
    ) -> Result<crate::opinion_api::OpinionResponse, Response> {
        request
            .validate()
            .map_err(|e| Failure::BadRequest(e).into_response())?;
        // A fast pre-check against this Handle's menu snapshot, matching
        // either id or name — not authoritative (see the field doc on
        // `menu`): the worker re-resolves `request.spec` itself and is the
        // one that actually refuses an evicted or unknown spec.
        let questions = {
            let menu = self.menu.read().expect("menu lock poisoned");
            let entry = menu
                .iter()
                .find(|e| e.id == request.spec || e.spec == request.spec)
                .ok_or_else(|| unknown_spec(request.spec.as_str()).into_response())?;
            entry
                .resolve_all(&request.questions)
                .map_err(|e| Failure::BadRequest(e).into_response())?
        };
        match self.submit(Job::Opinion(request, questions), "opinion").await? {
            Reply::Opinion(r) => Ok(r),
            _ => Err(
                Failure::Internal("worker answered an opinion with something else".into())
                    .into_response(),
            ),
        }
    }
    /// `POST /v1/probe`. Unlike [`Self::opine`] there is no spec/menu
    /// pre-check to run here — a probe names no spec, so shape validation
    /// (`ProbeRequest::validate`) is everything the handler can refuse
    /// before the worker.
    #[allow(clippy::result_large_err)]
    pub async fn probe(
        &self,
        request: crate::probe_api::ProbeRequest,
    ) -> Result<crate::probe_api::ProbeResponse, Response> {
        request
            .validate()
            .map_err(|e| Failure::BadRequest(e).into_response())?;
        match self.submit(Job::Probe(request), "probe").await? {
            Reply::Probe(r) => Ok(r),
            _ => Err(
                Failure::Internal("worker answered a probe with something else".into())
                    .into_response(),
            ),
        }
    }
    /// `POST /v1/opinion/specs`: `bytes` is the exact uploaded body — already
    /// bounded to `MAX_SPEC_BYTES` by the route's `DefaultBodyLimit` layer
    /// (`router`, below); anything over that never reaches here at all
    /// (axum answers `413` itself). The id is computed here, once,
    /// centrally — never inside a `Generator`, so it is the same content
    /// hash regardless of which engine is running (production `Adjudicator`
    /// or a test double) and cannot drift between them. Returns `201` for a
    /// genuine load, `200` for an already-loaded id (boot or upload) — no
    /// work done either way past this point. The published menu is NOT
    /// updated here — see the worker's `Job::Register` arm in `spawn`.
    #[allow(clippy::result_large_err)]
    pub async fn register(
        &self,
        bytes: Vec<u8>,
    ) -> Result<(StatusCode, crate::opinion_api::SpecMenuEntry), Response> {
        let id = crate::hash::sha256_hex_bytes(&bytes);
        let prompt: PromptSpec = serde_json::from_slice(&bytes).map_err(|e| {
            Failure::BadRequest(format!("spec is not a valid prompt spec: {e}")).into_response()
        })?;
        match self.submit(Job::Register(id, prompt), "register").await? {
            Reply::Register(outcome) => {
                let status = if outcome.newly_loaded { StatusCode::CREATED } else { StatusCode::OK };
                Ok((status, outcome.entry))
            }
            _ => Err(Failure::Internal(
                "worker answered a registration with something else".into(),
            )
            .into_response()),
        }
    }
    /// `DELETE /v1/opinion/specs/{id}`: `204` on deletion, `403` for a
    /// boot-time spec's id (refused, never deleted), `404` for an unknown
    /// id — already gone, or never loaded. The published menu is NOT
    /// updated here — see the worker's `Job::Unregister` arm in `spawn`.
    #[allow(clippy::result_large_err)]
    pub async fn unregister(&self, id: String) -> Result<StatusCode, Response> {
        match self.submit(Job::Unregister(id.clone()), "unregister").await? {
            Reply::Unregister(UnregisterOutcome::Deleted { .. }) => Ok(StatusCode::NO_CONTENT),
            Reply::Unregister(UnregisterOutcome::BootSpec) => Err(Failure::Forbidden(format!(
                "{id:?} is a boot-time spec (--opinion-spec) and cannot be deleted at runtime"
            ))
            .into_response()),
            Reply::Unregister(UnregisterOutcome::NotFound) => {
                Err(unknown_spec(&id).into_response())
            }
            _ => Err(Failure::Internal(
                "worker answered a deletion with something else".into(),
            )
            .into_response()),
        }
    }
}
/// Builds the adjudicator's router. `probe_enabled` (`--no-probe`/
/// `LFM2D_PROBE`, `config.rs`) decides whether `/v1/probe` is even ADDED to
/// the route table: with it `false`, the route is entirely absent, so a
/// request against it gets axum's plain unmatched-route 404 — never a
/// present-but-403 route, which would leak "this feature exists but is
/// turned off" to an unauthenticated prober. `/v1/tokenize` is a SEPARATE
/// router (`crate::tokenize_api::router`), merged by `main.rs` — it shares
/// no state and no toggle with this one; see that module's docs for why.
pub fn router(handle: Handle, probe_enabled: bool) -> Router {
    let mut router = Router::new()
        .route("/v1/adjudicator", get(info))
        .route("/v1/adjudicate", post(adjudicate))
        .route("/v1/opinion", post(opine))
        .route(
            "/v1/opinion/specs",
            get(specs)
                .post(register_spec)
                // Scoped to this route (not the whole router): a body over
                // MAX_SPEC_BYTES never reaches `register_spec` at all —
                // axum answers 413 itself, consistently, for every
                // oversized body rather than only the ones past its own
                // 2 MiB default (which `register_spec`'s own check used to
                // sit behind for 1-2 MiB bodies, and never saw beyond it).
                .layer(DefaultBodyLimit::max(MAX_SPEC_BYTES)),
        )
        .route("/v1/opinion/specs/{id}", axum::routing::delete(delete_spec))
        .route("/v1/chat", post(chat));
    if probe_enabled {
        router = router.route("/v1/probe", post(probe));
    }
    router
        .with_state(handle)
        .layer(axum::middleware::from_fn(
            crate::server::telemetry_middleware,
        ))
}
async fn info(State(h): State<Handle>) -> Json<AdjudicatorInfo> {
    Json(h.info)
}
async fn specs(State(h): State<Handle>) -> Json<Vec<crate::opinion_api::SpecMenuEntry>> {
    Json(h.menu())
}
#[allow(clippy::result_large_err)] // an axum handler's Err is a Response by design
async fn adjudicate(
    State(h): State<Handle>,
    crate::server::ValidJson(request): crate::server::ValidJson<AdjudicateRequest>,
) -> Result<Json<AdjudicateResponse>, Response> {
    h.evaluate(request).await.map(Json)
}
#[allow(clippy::result_large_err)]
async fn opine(
    State(h): State<Handle>,
    crate::server::ValidJson(request): crate::server::ValidJson<crate::opinion_api::OpinionRequest>,
) -> Result<Json<crate::opinion_api::OpinionResponse>, Response> {
    h.opine(request).await.map(Json)
}
#[allow(clippy::result_large_err)]
async fn chat(
    State(h): State<Handle>,
    crate::server::ValidJson(request): crate::server::ValidJson<crate::chat_session::ChatRequest>,
) -> Result<Response, Response> {
    if request.stream {
        h.chat_stream(request)
    } else {
        h.chat(request).await.map(|r| Json(r).into_response())
    }
}
#[allow(clippy::result_large_err)]
async fn probe(
    State(h): State<Handle>,
    crate::server::ValidJson(request): crate::server::ValidJson<crate::probe_api::ProbeRequest>,
) -> Result<Json<crate::probe_api::ProbeResponse>, Response> {
    h.probe(request).await.map(Json)
}
/// `POST /v1/opinion/specs`: the body IS the spec's bytes (no envelope, no
/// content-type requirement — a caller posts the file it would otherwise
/// pass via `--opinion-spec`). `201` newly loaded, `200` already loaded
/// (boot or upload, no work done), `400` unparseable, `422` a load-time
/// refusal (grammar compile, tokenization, opinion block split — the same
/// checks a boot spec passes before it ever answers a request).
#[allow(clippy::result_large_err)]
async fn register_spec(
    State(h): State<Handle>,
    bytes: axum::body::Bytes,
) -> Result<Response, Response> {
    let (status, entry) = h.register(bytes.to_vec()).await?;
    Ok((status, Json(entry)).into_response())
}
/// `DELETE /v1/opinion/specs/{id}`: `204` deleted, `403` `id` names a
/// boot-time spec (refused, not deleted), `404` unknown.
#[allow(clippy::result_large_err)]
async fn delete_spec(
    State(h): State<Handle>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Response, Response> {
    let status = h.unregister(id).await?;
    Ok(status.into_response())
}

/// `SpecStore`'s dedup/LRU/eviction rules, model-free: `Fixture` carries no
/// tokenizer, no grammar, no model state — just an id and a name — so these
/// pin the exact bookkeeping `Adjudicator::register`/`unregister` delegate
/// to, without a checkpoint. A mutation to the "already loaded" check here
/// (e.g. always calling `load`) is caught by
/// `registering_the_same_id_twice_does_no_second_load` below; see the
/// worktree's final report for the mutation round these were written for.
#[cfg(test)]
mod spec_store_tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Fixture {
        id: String,
        name: String,
        /// Only populated by `fp()`, for the `resolve_best_prefix_mut`
        /// tests below — every other fixture leaves this empty, which
        /// never matches anything (`best_prefix_match` requires
        /// `ids.len() > prefix.len()`, so an empty prefix only "matches"
        /// non-empty `ids`, and none of the id/name-keyed tests above ever
        /// call `resolve_best_prefix_mut`).
        prefix_ids: Vec<u32>,
    }
    impl SpecIdentity for Fixture {
        fn id(&self) -> &str {
            &self.id
        }
        fn name(&self) -> &str {
            &self.name
        }
    }
    fn f(id: &str) -> Fixture {
        Fixture { id: id.into(), name: format!("{id}-name"), prefix_ids: vec![] }
    }
    fn fp(id: &str, prefix_ids: &[u32]) -> Fixture {
        Fixture { id: id.into(), name: format!("{id}-name"), prefix_ids: prefix_ids.to_vec() }
    }
    fn store(boot: &[&str], capacity: usize) -> SpecStore<Fixture> {
        SpecStore::new(boot.iter().map(|id| f(id)).collect(), capacity)
    }

    #[test]
    fn resolve_mut_matches_boot_id_or_name_and_privileges_no_spec() {
        let mut s = store(&["a", "b"], 4);
        assert_eq!(s.resolve_mut("a").unwrap().id, "a");
        assert_eq!(s.resolve_mut("a-name").unwrap().id, "a");
        assert_eq!(s.resolve_mut("b").unwrap().id, "b");
        assert!(s.resolve_mut("nope").is_none(), "an unknown key is a clean miss, never a fallback to a boot spec");
    }

    /// Booting with no `--opinion-spec` is valid: the store starts empty and
    /// resolves nothing until an upload lands, and whatever lands is found
    /// only by its own id.
    #[test]
    fn an_empty_boot_store_resolves_nothing_until_an_upload_lands() {
        let mut s = store(&[], 4);
        assert!(s.resolve_mut("anything").is_none());
        assert!(s.resolve_best_prefix_mut(&[1, 2, 3], |f| &f.prefix_ids).is_none());
        s.register_or_load("up1", || Ok::<_, ()>(f("up1"))).unwrap();
        assert_eq!(s.resolve_mut("up1").unwrap().id, "up1");
        assert_eq!(s.iter().count(), 1);
    }

    #[test]
    fn resolve_mut_matches_an_uploaded_spec_by_id_only_and_touches_it_to_mru() {
        let mut s = store(&["boot"], 4);
        s.register_or_load("up1", || Ok::<_, ()>(f("up1"))).unwrap();
        s.register_or_load("up2", || Ok::<_, ()>(f("up2"))).unwrap();
        // An uploaded spec does NOT answer to its "name" (which equals its
        // id here, but resolve_mut's uploaded branch only checks `id()`).
        assert!(s.resolve_mut("up1-name").is_none());
        assert_eq!(s.resolve_mut("up1").unwrap().id, "up1");
        // Serving up1 touched it to the back; up2 is now the LRU one.
        assert_eq!(s.uploaded.front().unwrap().id, "up2");
        assert_eq!(s.uploaded.back().unwrap().id, "up1");
    }

    #[test]
    fn registering_the_same_id_twice_does_no_second_load() {
        let mut s = store(&["boot"], 4);
        let loads = std::cell::Cell::new(0);
        let load = || {
            loads.set(loads.get() + 1);
            Ok::<_, ()>(f("up1"))
        };
        let (e1, new1, ev1) = s.register_or_load("up1", load).unwrap();
        assert_eq!(e1.id, "up1");
        assert!(new1);
        assert!(ev1.is_none());
        assert_eq!(loads.get(), 1);
        let (e2, new2, ev2) = s.register_or_load("up1", load).unwrap();
        assert_eq!(e2.id, "up1");
        assert!(!new2, "a re-registration of an already-loaded id must not load again");
        assert!(ev2.is_none());
        assert_eq!(
            loads.get(),
            1,
            "second registration of the same id must not call the loader a second time"
        );
    }

    #[test]
    fn registering_a_boot_specs_id_is_also_a_free_no_op() {
        let mut s = store(&["boot"], 4);
        let loads = std::cell::Cell::new(0);
        let (entry, newly_loaded, evicted) = s
            .register_or_load("boot", || {
                loads.set(loads.get() + 1);
                Ok::<_, ()>(f("boot"))
            })
            .unwrap();
        assert_eq!(entry.id, "boot");
        assert!(!newly_loaded);
        assert!(evicted.is_none());
        assert_eq!(loads.get(), 0, "the boot spec's own id never reaches the loader");
        assert_eq!(s.uploaded.len(), 0, "it stays a boot spec, never duplicated into uploads");
    }

    #[test]
    fn capacity_plus_one_uploads_evicts_the_least_recently_used() {
        let mut s = store(&["boot"], 2);
        s.register_or_load("a", || Ok::<_, ()>(f("a"))).unwrap();
        s.register_or_load("b", || Ok::<_, ()>(f("b"))).unwrap();
        // Touch `a` so `b` becomes the LRU one.
        assert!(s.resolve_mut("a").is_some());
        let (entry, newly_loaded, evicted) = s.register_or_load("c", || Ok::<_, ()>(f("c"))).unwrap();
        assert_eq!(entry.id, "c");
        assert!(newly_loaded);
        assert_eq!(evicted.as_deref(), Some("b"), "the least recently used upload is evicted, not `a`");
        assert!(s.resolve_mut("b").is_none(), "an evicted spec is a clean miss afterward");
        assert!(s.resolve_mut("a").is_some(), "a recently-used upload survives the eviction");
        assert!(s.resolve_mut("c").is_some());
        assert_eq!(s.uploaded.len(), 2, "capacity is never exceeded");
    }

    /// The sibling of `capacity_plus_one_uploads_evicts_the_least_recently_used`,
    /// but touching `a` via a RE-REGISTRATION (`register_or_load`'s own
    /// "already uploaded" branch) rather than via `resolve_mut` — a
    /// distinct code path (`POST /v1/opinion/specs` of the same bytes
    /// again, not a `/v1/opinion` read) that must ALSO count as use, per
    /// the ruling ("registered or served"). This is the one the mutation
    /// round targets: dropping the touch from `register_or_load`'s
    /// already-uploaded branch (so a re-registration doesn't move the
    /// entry to the back) makes this fail while the sibling test above
    /// still passes, since that one never re-registers anything.
    #[test]
    fn reregistering_an_upload_also_counts_as_use_and_protects_it_from_eviction() {
        let mut s = store(&["boot"], 2);
        s.register_or_load("a", || Ok::<_, ()>(f("a"))).unwrap();
        s.register_or_load("b", || Ok::<_, ()>(f("b"))).unwrap();
        // Re-register `a`'s own id — the already-uploaded branch, not a
        // resolve_mut read — so `b` becomes the LRU one.
        let loads = std::cell::Cell::new(0);
        let (entry, newly_loaded, evicted) = s
            .register_or_load("a", || {
                loads.set(loads.get() + 1);
                Ok::<_, ()>(f("a"))
            })
            .unwrap();
        assert_eq!(entry.id, "a");
        assert!(!newly_loaded, "re-registering an already-uploaded id does no load");
        assert!(evicted.is_none());
        assert_eq!(loads.get(), 0, "the already-uploaded branch never calls the loader");
        let (entry, newly_loaded, evicted) = s.register_or_load("c", || Ok::<_, ()>(f("c"))).unwrap();
        assert_eq!(entry.id, "c");
        assert!(newly_loaded);
        assert_eq!(
            evicted.as_deref(),
            Some("b"),
            "b is the least recently used upload -- a was protected by its RE-REGISTRATION, not a read"
        );
        assert!(s.resolve_mut("a").is_some(), "a survives");
        assert!(s.resolve_mut("b").is_none(), "b was evicted");
    }

    #[test]
    fn boot_specs_are_never_evicted_however_many_uploads_arrive() {
        let mut s = store(&["boot"], 1);
        for id in ["a", "b", "c", "d"] {
            s.register_or_load(id, || Ok::<_, ()>(f(id))).unwrap();
        }
        assert_eq!(s.boot.len(), 1);
        assert_eq!(s.resolve_mut("boot").unwrap().id, "boot");
        assert_eq!(s.uploaded.len(), 1, "capacity 1 holds exactly the most recent upload");
        assert_eq!(s.resolve_mut("d").unwrap().id, "d");
    }

    #[test]
    fn remove_distinguishes_deleted_boot_and_not_found() {
        let mut s = store(&["boot"], 4);
        s.register_or_load("up1", || Ok::<_, ()>(f("up1"))).unwrap();
        assert!(matches!(s.remove("boot"), RemoveOutcome::Boot));
        assert!(s.resolve_mut("boot").is_some(), "refusing a boot delete must not remove it");
        match s.remove("up1") {
            RemoveOutcome::Removed(spec) => assert_eq!(spec.id, "up1"),
            _ => panic!("up1 was loaded and must be removable"),
        }
        assert!(s.resolve_mut("up1").is_none(), "404 after delete");
        assert!(matches!(s.remove("up1"), RemoveOutcome::NotFound), "deleting twice is a clean not-found");
        assert!(matches!(s.remove("never-loaded"), RemoveOutcome::NotFound));
    }

    #[test]
    fn remove_refuses_a_boot_spec_by_its_name_too_not_only_its_id() {
        // resolve_mut matches a boot spec by id OR name (a boot spec
        // "also answers to its file-stem name"); remove must refuse the
        // same set of keys, or DELETE by name would 404 a spec that is
        // still very much loaded and answerable by that same name.
        let mut s = store(&["boot"], 4);
        assert!(
            matches!(s.remove("boot-name"), RemoveOutcome::Boot),
            "a boot spec's name must also be refused as Boot, not treated as unknown"
        );
        assert!(s.resolve_mut("boot").is_some(), "refusing by name must not remove it");
    }

    #[test]
    fn iter_lists_boot_before_uploaded() {
        let mut s = store(&["boot1", "boot2"], 4);
        s.register_or_load("up1", || Ok::<_, ()>(f("up1"))).unwrap();
        let ids: Vec<&str> = s.iter().map(|f| f.id.as_str()).collect();
        assert_eq!(ids, ["boot1", "boot2", "up1"]);
    }

    // ------------------------------------------ resolve_best_prefix_mut (F2/F4/F5)

    #[test]
    fn resolve_best_prefix_mut_picks_the_longest_match_across_boot_and_uploaded() {
        let mut s = SpecStore::new(vec![fp("short", &[1, 2])], 4);
        s.register_or_load("long", || Ok::<_, ()>(fp("long", &[1, 2, 3, 4]))).unwrap();
        let ids = [1, 2, 3, 4, 5];
        let winner = s.resolve_best_prefix_mut(&ids, |f| &f.prefix_ids).unwrap();
        assert_eq!(winner.id, "long", "the longer prefix must win even though it was registered second");
    }

    #[test]
    fn resolve_best_prefix_mut_requires_a_strict_prefix_not_an_exact_match() {
        let mut s = SpecStore::new(vec![fp("exact", &[1, 2, 3]), fp("shorter", &[1, 2])], 4);
        let ids = [1, 2, 3];
        let winner = s.resolve_best_prefix_mut(&ids, |f| &f.prefix_ids).unwrap();
        assert_eq!(winner.id, "shorter", "a prefix equal to ids leaves nothing to bulk-forward");
    }

    #[test]
    fn resolve_best_prefix_mut_is_none_when_nothing_strictly_prefixes() {
        let mut s = SpecStore::new(vec![fp("exact", &[1, 2, 3])], 4);
        assert!(s.resolve_best_prefix_mut(&[1, 2, 3], |f| &f.prefix_ids).is_none());
        assert!(s.resolve_best_prefix_mut(&[9, 9], |f| &f.prefix_ids).is_none());
    }

    #[test]
    fn resolve_best_prefix_mut_touches_an_uploaded_winner_to_the_back_of_the_lru() {
        // capacity 2: registering a third upload evicts the LRU front
        // unless something already touched it to the back.
        let mut s = SpecStore::new(vec![], 2);
        s.register_or_load("a", || Ok::<_, ()>(fp("a", &[1, 2]))).unwrap();
        s.register_or_load("b", || Ok::<_, ()>(fp("b", &[9, 9]))).unwrap();
        // "a" is now LRU-front (registered first, never served since). A
        // probe resuming it must touch it to the back — "served-or-
        // registered = use" (F5) — exactly as `resolve_mut` already does
        // for a request that names a spec by id.
        let winner = s.resolve_best_prefix_mut(&[1, 2, 3], |f| &f.prefix_ids).unwrap();
        assert_eq!(winner.id, "a");
        s.register_or_load("c", || Ok::<_, ()>(fp("c", &[3, 3]))).unwrap();
        let ids: Vec<&str> = s.iter().map(|f| f.id.as_str()).collect();
        assert_eq!(
            ids, ["a", "c"],
            "resuming 'a' must have touched it past 'b', which is now the one evicted"
        );
    }

    #[test]
    fn resolve_best_prefix_mut_does_not_touch_a_boot_spec_since_boot_is_never_evicted() {
        let mut s = SpecStore::new(vec![fp("boot", &[1, 2])], 4);
        let winner = s.resolve_best_prefix_mut(&[1, 2, 3], |f| &f.prefix_ids).unwrap();
        assert_eq!(winner.id, "boot");
        // Boot specs are never evicted regardless of order, so there is
        // nothing to assert about position here beyond: it is still there.
        assert!(s.resolve_mut("boot").is_some());
    }
}

#[cfg(test)]
mod prompt_cache_tests {
    use super::*;
    use std::cell::Cell;
    fn fixture() -> (Model, PromptCache) {
        let mut f = std::io::Cursor::new(include_bytes!("../tests/fixtures/lfm2-moe/tiny.gguf"));
        let ct = candle_core::quantized::gguf_file::Content::read(&mut f).unwrap();
        let model = Model::from_gguf(ct, &mut f, &candle_core::Device::Cpu).unwrap();
        let mut prefix = model.new_state();
        model.forward(&[1, 2, 3], &mut prefix).unwrap();
        (
            model,
            PromptCache {
                prefix,
                prefix_ids: vec![1, 2, 3],
                ready: None,
            },
        )
    }
    fn values(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }
    #[test]
    fn prepared_prompt_keeps_logits_and_hybrid_state_isolated() {
        let (model, mut cache) = fixture();
        let tokens = [1, 2, 3, 4, 5];
        let mut first = cache.prepare(&model, &tokens, true, &|| Ok(())).unwrap();
        assert_eq!(first.cached_tokens, 3);
        let expected = values(&first.logits);
        let expected_next = values(&model.forward(&[6], &mut first.state).unwrap());
        model.forward(&[7, 8], &mut first.state).unwrap();
        let mut second = cache.prepare(&model, &tokens, true, &|| Ok(())).unwrap();
        assert_eq!(second.cached_tokens, tokens.len());
        assert_eq!(second.state.len(), tokens.len());
        assert_eq!(values(&second.logits), expected);
        assert_eq!(
            values(&model.forward(&[6], &mut second.state).unwrap()),
            expected_next
        );
        assert_eq!(cache.prefix.len(), 3);
    }
    /// What `Adjudicator::generate` does across its prefill pauses, on the
    /// tiny model: look up, forward chunk by chunk with a read served at
    /// every pause (`prepare` on the SAME cache, publishing its own ready
    /// entry), and publish only after the last chunk. The paused prefill
    /// computes exactly what it computes alone; nothing of it is visible in
    /// the cache until it publishes; a ready entry the paused job cloned out
    /// is unaffected by the read that replaces it; and the reads compute
    /// what they compute alone.
    #[test]
    fn reads_at_a_paused_prefill_neither_see_nor_disturb_it() {
        let (model, mut cache) = fixture();
        let ok = || Ok(());
        let a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let reads: [&[u32]; 3] = [&[1, 2, 3, 10], &[1, 2, 3, 11, 12], &[1, 2, 3, 13]];
        // Alone: `a` forwarded in chunks of two from the resident prefix,
        // and each read prepared on a cache of its own.
        let fresh = |cache: &PromptCache| PromptCache {
            prefix: cache.prefix.clone(),
            prefix_ids: cache.prefix_ids.clone(),
            ready: None,
        };
        let Lookup::Cold { mut state, start } = fresh(&cache).lookup(&model, &a, true).unwrap() else {
            panic!("an empty cache has nothing ready")
        };
        let alone = forward_chunks(&model, &mut state, &a[start..], 2, &mut || Ok(())).unwrap();
        let alone_next = model.forward(&[14], &mut state).unwrap();
        let reads_alone: Vec<Vec<f32>> = reads
            .iter()
            .map(|r| values(&fresh(&cache).prepare(&model, r, true, &ok).unwrap().logits))
            .collect();

        // Paused: the same prefill with a read at every pause.
        let Lookup::Cold { mut state, start } = cache.lookup(&model, &a, true).unwrap() else {
            panic!("nothing ready yet")
        };
        let served = std::cell::RefCell::new(Vec::new());
        let pauses = Cell::new(0);
        let logits = forward_chunks(&model, &mut state, &a[start..], 2, &mut || {
            let n = pauses.get();
            pauses.set(n + 1);
            if let Some(r) = reads.get(n) {
                served.borrow_mut().push(values(&cache.prepare(&model, r, true, &ok)?.logits));
                // Every pause sees the read's entry, never the paused job's.
                assert_eq!(cache.ready.as_ref().map(|p| p.token_ids.as_slice()), Some(*r));
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(pauses.get(), 3, "six suffix tokens, chunks of two");
        assert_eq!(values(&logits), values(&alone), "the paused prefill is the one it computes alone");
        assert_eq!(*served.borrow(), reads_alone, "and the reads are the ones they compute alone");
        cache.publish(&a, &state, &logits);
        let hit = cache.prepare(&model, &a, true, &ok).unwrap();
        assert_eq!(hit.cached_tokens, a.len(), "published once complete");
        assert_eq!(values(&hit.logits), values(&alone));

        // A ready entry cloned out before a read replaces it decodes on as
        // if nothing had happened: clones share no mutable state.
        let Lookup::Ready(mut paused) = cache.lookup(&model, &a, true).unwrap() else {
            panic!("`a` is ready")
        };
        cache.prepare(&model, reads[0], true, &ok).unwrap();
        let next = model.forward(&[14], &mut paused.state).unwrap();
        assert_eq!(values(&next), values(&alone_next));
    }

    #[test]
    fn ready_input_cannot_cross_model_instances() {
        let (model, mut cache) = fixture();
        let (other, _) = fixture();
        cache
            .prepare(&model, &[1, 2, 3, 4], true, &|| Ok(()))
            .unwrap();
        assert!(
            cache
                .prepare(&other, &[1, 2, 3, 4], true, &|| Ok(()))
                .is_err()
        );
    }

    #[test]
    fn cache_is_exact_bounded_and_cold_requests_bypass_it() {
        let (model, mut cache) = fixture();
        let a = [1, 2, 3, 4];
        let b = [1, 2, 3, 5];
        cache.prepare(&model, &a, true, &|| Ok(())).unwrap();
        assert_eq!(
            cache
                .prepare(&model, &b, false, &|| Ok(()))
                .unwrap()
                .cached_tokens,
            0
        );
        assert_eq!(
            cache
                .prepare(&model, &a, true, &|| Ok(()))
                .unwrap()
                .cached_tokens,
            a.len()
        );
        assert_eq!(
            cache
                .prepare(&model, &b, true, &|| Ok(()))
                .unwrap()
                .cached_tokens,
            3
        );
        assert_eq!(
            cache
                .prepare(&model, &a, true, &|| Ok(()))
                .unwrap()
                .cached_tokens,
            3
        );
    }
    #[test]
    fn step_distribution_over_real_model_logits_is_deterministic_and_internally_consistent() {
        // Exercises crate::types::step_distribution against an ACTUAL Tensor
        // -> Vec<f32> extraction from the real (tiny) model, not just
        // hand-built float arrays — the same `flatten_all().to_vec1()` path
        // the decode-loop hunk in `generate` uses. Two independent forward
        // passes from a fresh state on identical tokens must agree exactly:
        // this stack has been bit-deterministic elsewhere (see
        // `prepared_prompt_keeps_logits_and_hybrid_state_isolated` above),
        // and if it were NOT deterministic here, that would be a finding to
        // report, not paper over.
        let (model, _cache) = fixture();
        let mut state_a = model.new_state();
        let raw_a = values(&model.forward(&[1, 2, 3, 4], &mut state_a).unwrap());
        let mut state_b = model.new_state();
        let raw_b = values(&model.forward(&[1, 2, 3, 4], &mut state_b).unwrap());
        assert_eq!(
            raw_a, raw_b,
            "identical prompt tokens from a fresh state must produce bit-identical logits"
        );

        let text = |id: u32| Some(format!("<{id}>"));
        let mut sets = std::collections::BTreeMap::new();
        sets.insert("probe".to_string(), vec![0u32, 1]);
        let step_a = crate::types::step_distribution(&raw_a, 0, 5, &sets, text).unwrap();
        let step_b = crate::types::step_distribution(&raw_b, 0, 5, &sets, text).unwrap();
        assert_eq!(step_a.logprob, step_b.logprob);
        assert_eq!(
            step_a
                .top_logprobs
                .iter()
                .map(|t| (t.token, t.logprob))
                .collect::<Vec<_>>(),
            step_b
                .top_logprobs
                .iter()
                .map(|t| (t.token, t.logprob))
                .collect::<Vec<_>>()
        );
        assert_eq!(step_a.set_mass["probe"].logprob, step_b.set_mass["probe"].logprob);

        assert!(step_a.logprob <= 0.0);
        for w in step_a.top_logprobs.windows(2) {
            assert!(w[0].logprob >= w[1].logprob, "{:?} not descending", step_a.top_logprobs);
        }
        let total: f32 = raw_a
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(total.is_finite(), "tiny fixture must produce finite logits");
    }

    #[test]
    fn described_cache_is_keyed_by_prompt_and_field_and_evicts_the_oldest() {
        let (model, _) = fixture();
        let entry = |prompt: &[u32], field: &str, generated: &[u32]| {
            let mut state = model.new_state();
            let logits = model.forward(prompt, &mut state).unwrap();
            DescribedEntry {
                prompt_ids: prompt.to_vec(),
                field: field.into(),
                generated: generated.to_vec(),
                text: format!("{field}:{}", generated.len()),
                state,
                logits,
            }
        };
        let mut cache = DescribedCache::new(2);
        cache.insert(entry(&[1, 2, 3], "verdict", &[4, 5]));
        cache.insert(entry(&[1, 2, 3], "scope", &[4]));
        // Same prompt, different field: a different slot, so a different entry.
        assert_eq!(cache.len(), 2);
        let (generated, text, state, _) = cache.get(&[1, 2, 3], "verdict").unwrap();
        assert_eq!(generated, [4, 5]);
        assert_eq!(text, "verdict:2");
        assert_eq!(state.len(), 3);
        assert!(cache.get(&[1, 2, 3], "undo").is_none());
        assert!(cache.get(&[1, 2, 4], "verdict").is_none());
        // The hit above made `verdict` most recent, so a third entry evicts `scope`.
        cache.insert(entry(&[9, 9, 9], "verdict", &[1]));
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&[1, 2, 3], "scope").is_none());
        assert!(cache.get(&[1, 2, 3], "verdict").is_some());
        // Re-inserting a key replaces rather than duplicates.
        cache.insert(entry(&[9, 9, 9], "verdict", &[1, 2, 3]));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&[9, 9, 9], "verdict").unwrap().0, [1, 2, 3]);
        let mut none = DescribedCache::new(0);
        none.insert(entry(&[1], "verdict", &[2]));
        assert_eq!(none.len(), 0);
    }

    /// The resume path rebuilds the sampler with `prompt + resumed` as
    /// history instead of sampling the resumed tokens. Candle's history is a
    /// seen-set that every selection marks, so the two must agree; this pins
    /// it with logits where the penalty decides the pick.
    #[test]
    fn a_sampler_built_from_resumed_history_selects_as_the_incremental_one_would() {
        let device = candle_core::Device::Cpu;
        let vocab = 8usize;
        // `top` leads by less than the penalty takes off; `runner` is never
        // in any history here.
        let logits = |top: usize, runner: usize| {
            let mut v = vec![0.5f32; vocab];
            v[top] = 3.0;
            v[runner] = 2.9;
            Tensor::new(v.as_slice(), &device).unwrap().unsqueeze(0).unwrap()
        };
        let prompt = [1u32, 2];
        // Incremental: sample 5 then 6 (each becomes history).
        let mut incremental =
            crate::constrain::Decoder::new(None, &device, vocab, &prompt, 1.05).unwrap();
        let mut resumed_tokens = Vec::new();
        for top in [5usize, 6] {
            let t = incremental.sample(&logits(top, 7)).unwrap();
            assert_eq!(t as usize, top);
            resumed_tokens.push(t);
        }
        // Rebuilt: the same tokens as constructor history, walked by `advance`.
        let history: Vec<u32> = prompt.iter().chain(&resumed_tokens).copied().collect();
        let mut rebuilt =
            crate::constrain::Decoder::new(None, &device, vocab, &history, 1.05).unwrap();
        rebuilt.advance(&resumed_tokens).unwrap();
        // 5 is the raw argmax but penalised in both, so 7 wins; a sampler
        // that forgot the resumed tokens would pick 5.
        let l = logits(5, 7);
        let a = incremental.sample(&l).unwrap();
        let b = rebuilt.sample(&l).unwrap();
        assert_eq!(a, b);
        let mut forgetful =
            crate::constrain::Decoder::new(None, &device, vocab, &prompt, 1.05).unwrap();
        assert_eq!(forgetful.sample(&l).unwrap(), 5);
        assert_ne!(a, 5, "the penalty must have moved the pick off the resumed token");
    }

    #[test]
    fn get_deepest_returns_the_longest_description_for_the_prompt() {
        let (model, _) = fixture();
        let entry = |field: &str, generated: &[u32]| {
            let mut state = model.new_state();
            let logits = model.forward(&[1, 2, 3], &mut state).unwrap();
            DescribedEntry {
                prompt_ids: vec![1, 2, 3],
                field: field.into(),
                generated: generated.to_vec(),
                text: String::new(),
                state,
                logits,
            }
        };
        let mut cache = DescribedCache::new(4);
        cache.insert(entry("scope", &[4]));
        cache.insert(entry("verdict", &[4, 5, 6]));
        cache.insert(entry("undo", &[4, 5]));
        assert_eq!(cache.get_deepest(&[1, 2, 3]).unwrap().0, [4, 5, 6]);
        assert!(cache.get_deepest(&[1, 2, 4]).is_none());
    }

    #[test]
    fn a_cached_description_reads_the_same_state_it_was_generated_from() {
        // The cache hands back a clone of the state at the slot; scoring off it
        // must equal scoring off the state that produced it, and must not
        // advance what the cache holds.
        let (model, _) = fixture();
        let mut state = model.new_state();
        model.forward(&[1, 2, 3], &mut state).unwrap();
        let logits = model.forward(&[4, 5], &mut state).unwrap();
        let direct = crate::opinion::score_continuations(&model, &state, &logits, &[vec![6, 7], vec![8]], &|| Ok(())).unwrap();
        let mut cache = DescribedCache::new(1);
        cache.insert(DescribedEntry {
            prompt_ids: vec![1, 2, 3],
            field: "verdict".into(),
            generated: vec![4, 5],
            text: String::new(),
            state,
            logits,
        });
        for _ in 0..2 {
            let (_, _, hit_state, hit_logits) = cache.get(&[1, 2, 3], "verdict").unwrap();
            let cached = crate::opinion::score_continuations(&model, &hit_state, &hit_logits, &[vec![6, 7], vec![8]], &|| Ok(())).unwrap();
            assert_eq!(direct, cached);
            assert_eq!(hit_state.len(), 5);
        }
    }

    #[test]
    fn parse_described_reads_the_fields_before_the_slot_in_order() {
        let fields = |names: &[&str]| names.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let out = parse_described(
            "{\"effect\": \"Removes it.\", \"scope\": \"project\", \"undo\": \"easy\", \"verdict\": \"",
            "\"verdict\": \"",
            &fields(&["effect", "scope", "undo"]),
        )
        .unwrap();
        let got: Vec<(String, serde_json::Value)> = out.into_iter().map(|d| (d.field, d.value)).collect();
        assert_eq!(
            got,
            [
                ("effect".to_string(), serde_json::json!("Removes it.")),
                ("scope".to_string(), serde_json::json!("project")),
                ("undo".to_string(), serde_json::json!("easy")),
            ]
        );
        // The first field asked: nothing described.
        assert!(parse_described("{\"verdict\": \"", "\"verdict\": \"", &[]).unwrap().is_empty());
        // Booleans arrive unquoted and stay booleans.
        let out = parse_described("{\"writes\": true, \"verdict\": \"", "\"verdict\": \"", &fields(&["writes"])).unwrap();
        assert_eq!(out[0].value, serde_json::json!(true));
        // Loud on every mismatch between the spec's field list and the text.
        assert!(parse_described("{\"effect\": \"x\", \"verdict\": \"", "\"verdict\": \"", &[]).is_err());
        assert!(parse_described("{\"effect\": \"x\", \"verdict\": \"", "\"verdict\": \"", &fields(&["effect", "scope"])).is_err());
        assert!(parse_described("{\"effect\": \"x\", \"verdict\": \"", "\"verdict\": \"", &fields(&["scope"])).is_err());
        assert!(parse_described("{\"effect\": \"x\"\"verdict\": \"", "\"verdict\": \"", &fields(&["effect"])).is_err());
        assert!(parse_described("{\"effect\": \"x\", ", "\"verdict\": \"", &fields(&["effect"])).is_err());
    }

    #[test]
    fn failed_or_cancelled_prefill_does_not_replace_ready_input() {
        let (model, mut cache) = fixture();
        let a = [1, 2, 3, 4];
        cache.prepare(&model, &a, true, &|| Ok(())).unwrap();
        assert!(
            cache
                .prepare(&model, &[1, 2, 3, 16], true, &|| Ok(()))
                .is_err()
        );
        assert!(
            cache
                .prepare(&model, &[1, 2, 9, 4], true, &|| Ok(()))
                .is_err()
        );
        let calls = Cell::new(0);
        let check = || {
            calls.set(calls.get() + 1);
            if calls.get() > 2 {
                Err(Failure::Cancelled)
            } else {
                Ok(())
            }
        };
        assert!(matches!(
            cache.prepare(&model, &[1, 2, 3, 5], true, &check),
            Err(Failure::Cancelled)
        ));
        assert_eq!(
            cache
                .prepare(&model, &a, true, &|| Ok(()))
                .unwrap()
                .cached_tokens,
            a.len()
        );
        assert!(matches!(
            cache.prepare(&model, &a, true, &|| Err(Failure::Deadline)),
            Err(Failure::Deadline)
        ));
    }
}

#[cfg(test)]
mod priority_tests {
    use super::*;

    /// `ALL` is the class order the pick and the pause walk, and `Ord` is
    /// what `next_above` compares: a class added out of place in either
    /// would serve the wrong jobs at a pause.
    #[test]
    fn all_lists_every_class_once_lowest_first() {
        let mut sorted = Priority::ALL.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted, Priority::ALL);
        for (i, class) in Priority::ALL.iter().enumerate() {
            assert_eq!(class.index(), i);
        }
    }

    /// Only reads and probes overtake. The F8 read travels as an
    /// `AdjudicateRequest`, so the flag, not the job type, decides.
    #[test]
    fn reads_and_probes_are_interactive_and_everything_that_removes_generates_or_chats_is_not() {
        let adjudicate = |opinion: bool| -> AdjudicateRequest {
            serde_json::from_value(serde_json::json!({"spec": "s", "input": "x", "opinion": opinion})).unwrap()
        };
        let opinion: crate::opinion_api::OpinionRequest = serde_json::from_value(serde_json::json!({
            "spec": "s", "state": {"input": "x"}, "questions": [{"field": "f"}]
        }))
        .unwrap();
        let probe: crate::probe_api::ProbeRequest =
            serde_json::from_value(serde_json::json!({"text": "x"})).unwrap();
        let prompt: PromptSpec = serde_json::from_value(serde_json::json!({
            "input_label": "Input", "system": "Judge."
        }))
        .unwrap();
        let chat: crate::chat_session::ChatRequest =
            serde_json::from_value(serde_json::json!({"messages": [{"role": "user", "content": "hi"}]})).unwrap();
        for (job, class) in [
            (Job::Adjudicate(adjudicate(true)), Priority::Interactive),
            (Job::Opinion(opinion, vec![]), Priority::Interactive),
            (Job::Probe(probe), Priority::Interactive),
            (Job::Adjudicate(adjudicate(false)), Priority::Generative),
            (Job::Register("id".into(), prompt), Priority::Generative),
            (Job::Unregister("id".into()), Priority::Generative),
            (Job::Chat(chat, None), Priority::Generative),
        ] {
            assert_eq!(job.priority(), class, "{}", job.operation());
        }
    }
}

/// Whether the live `Handle.menu` a caller reads stays in step with the
/// worker's own `specs` — the property "MENU CAN FALL BEHIND THE WORKER"
/// named. A job whose caller is ALREADY gone before the worker even looks
/// at it is correctly refused up front (`check()`'s pre-check, unchanged —
/// see `no_work_happens_for_a_caller_that_was_already_gone_before_the_worker_looked`
/// below, which pins that this stays true) and is not what these tests
/// are about. The bug (and the fix) is about a caller that disconnects
/// AFTER the worker has already started mutating the store: the mutation
/// must complete and be published regardless.
#[cfg(test)]
mod handle_menu_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn info() -> PrefixInfo {
        PrefixInfo {
            model_id: "fixture".into(),
            weight_hash: "hash".into(),
            tokenizer_hash: "tok".into(),
            template_version: "test".into(),
            snapshot_id: "snapshot".into(),
            prefix_tokens: 7,
            input_cache_capacity: 1,
            context_limit: 128,
            backend: "cpu".into(),
            device: "cpu".into(),
            candle_rev: "test".into(),
            dtype: "f32".into(),
            sampling: "greedy".into(),
            weight_dtypes: vec!["F32".into()],
        }
    }
    fn prompt(system: &str) -> PromptSpec {
        serde_json::from_value(serde_json::json!({"input_label": "Input", "system": system})).unwrap()
    }
    fn body_of(p: &PromptSpec) -> Vec<u8> {
        serde_json::to_vec(p).unwrap()
    }

    /// A generator whose `register`/`unregister` are real enough to build a
    /// menu entry from the id/prompt it's handed, count calls, and — when
    /// `delay` is set — sleep (on this worker OS thread, `check()` is never
    /// polled during it, same as the real `LoadedSpec::load`'s prefill
    /// loop) AFTER counting the call but BEFORE mutating `registered`. That
    /// window is where a test aborts the caller: the abort is guaranteed to
    /// land while the "load" is in flight and strictly before the mutation,
    /// so the mutation the test later observes published genuinely
    /// happened after the caller was already gone.
    struct CountingRegistrar {
        calls: Arc<AtomicUsize>,
        unregister_calls: Arc<AtomicUsize>,
        /// Accumulated, exactly like the real `Adjudicator`'s `menu()`
        /// (boot + uploaded) — NOT a fresh single-entry vec each call. The
        /// tests using this depend on `RegisterOutcome::menu` and
        /// `UnregisterOutcome::Deleted { menu, .. }` being the FULL current
        /// menu, since that's what the worker publishes wholesale.
        registered: Vec<crate::opinion_api::SpecMenuEntry>,
        delay: Option<Duration>,
    }
    impl Generator for CountingRegistrar {
        fn generate(
            &mut self,
            _: &AdjudicateRequest,
            _: &dyn YieldPoint<Self>,
        ) -> Result<AdjudicateResponse, Failure> {
            Err(Failure::Internal("not exercised here".into()))
        }
        fn opine(
            &mut self,
            _: &crate::opinion_api::OpinionRequest,
            _: &[crate::opinion_api::ResolvedQuestion],
            _: &dyn Fn() -> Result<(), Failure>,
        ) -> Result<crate::opinion_api::OpinionResponse, Failure> {
            Err(Failure::Internal("not exercised here".into()))
        }
        fn register(
            &mut self,
            id: String,
            prompt: PromptSpec,
            _: &dyn Fn() -> Result<(), Failure>,
        ) -> Result<RegisterOutcome, Failure> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some(delay) = self.delay {
                std::thread::sleep(delay);
            }
            let entry = crate::opinion_api::SpecMenuEntry::from_prompt(&id, &id, &prompt, "snap", 16)
                .expect("test fixture spec is always a valid menu entry");
            self.registered.retain(|e| e.id != entry.id);
            self.registered.push(entry.clone());
            Ok(RegisterOutcome {
                entry,
                newly_loaded: true,
                evicted: None,
                load_ms: 0.,
                menu: self.registered.clone(),
            })
        }
        fn unregister(&mut self, id: &str) -> UnregisterOutcome {
            self.unregister_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(delay) = self.delay {
                std::thread::sleep(delay);
            }
            match self.registered.iter().position(|e| e.id == id) {
                Some(pos) => {
                    let entry = self.registered.remove(pos);
                    UnregisterOutcome::Deleted { entry, menu: self.registered.clone() }
                }
                None => UnregisterOutcome::NotFound,
            }
        }
        fn probe(
            &mut self,
            _: &crate::probe_api::ProbeRequest,
            _: &dyn Fn() -> Result<(), Failure>,
        ) -> Result<crate::probe_api::ProbeResponse, Failure> {
            Err(Failure::Internal("not exercised here".into()))
        }
    }

    /// Not a regression test for the bug — a control, pinning the behavior
    /// the fix must NOT change: a caller whose reply receiver is already
    /// gone before the worker ever looks at the job is refused up front by
    /// `check()`'s pre-check, and the generator is never even called. If
    /// this stopped being true (e.g. someone "fixed" the bug by dropping
    /// the pre-check entirely), unrelated already-abandoned work would
    /// start silently consuming worker time forever.
    #[tokio::test]
    async fn no_work_happens_for_a_caller_that_was_already_gone_before_the_worker_looked() {
        let calls = Arc::new(AtomicUsize::new(0));
        let handle = Handle::spawn(
            CountingRegistrar {
                calls: calls.clone(),
                unregister_calls: Arc::new(AtomicUsize::new(0)),
                registered: Vec::new(),
                delay: None,
            },
            (&info()).into(),
        );
        let (reply, rx) = oneshot::channel();
        drop(rx);
        handle.queues[Priority::Generative.index()]
            .send(Work {
                job: Job::Register("id1".into(), prompt("one")),
                reply,
                enqueued: Instant::now(),
                span: tracing::info_span!("test"),
            })
            .unwrap();
        // Serialize on a second, normal registration so the first has
        // certainly been dequeued (the worker is single-threaded, FIFO)
        // before asserting. `calls` counts BOTH jobs if job 1 wrongly
        // reaches the generator, so 1 (not 0) is the "job 1 was skipped"
        // reading — this registration's own call is the one that's there.
        let (status, _) = handle.register(body_of(&prompt("two"))).await.unwrap();
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "an already-abandoned job must never reach the generator (only job 2's own call should count)"
        );
        assert_eq!(handle.menu().len(), 1, "only the second registration is on the menu");
    }

    /// The actual regression test: the caller aborts WHILE the worker is
    /// in the middle of the (simulated, slow) load — after `check()`'s
    /// pre-check already passed, so the generator's `register` was called
    /// and is in flight — but before the store mutation happens. Under the
    /// old shape, `Adjudicator::register` ran `check()` again right after
    /// the mutation and turned the now-cancelled caller into a lost
    /// `Failure::Cancelled`, and `Handle::register`'s `replace_menu` call
    /// (gated on receiving a successful reply) never ran because there was
    /// no live task left to receive it. The fix: no `check()` after a
    /// mutation, and the WORKER publishes synchronously as part of
    /// processing the job — so this must pass regardless.
    #[tokio::test]
    async fn the_worker_publishes_a_registration_whose_caller_disconnects_mid_flight() {
        let calls = Arc::new(AtomicUsize::new(0));
        let handle = Handle::spawn(
            CountingRegistrar {
                calls: calls.clone(),
                unregister_calls: Arc::new(AtomicUsize::new(0)),
                registered: Vec::new(),
                delay: Some(Duration::from_millis(80)),
            },
            (&info()).into(),
        );
        let bytes = body_of(&prompt("mid-flight"));
        let id = sha256_hex_bytes(&bytes);
        let task = {
            let handle = handle.clone();
            tokio::spawn(async move { handle.register(bytes).await })
        };
        // Wait until the worker has actually entered `register` (so the
        // pre-check already passed) before severing the caller.
        tokio::time::timeout(Duration::from_secs(2), async {
            while calls.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the generator must start processing");
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled(), "the caller task was actually aborted");

        // The worker is still asleep inside `register`'s simulated load,
        // with no caller left. Poll for the publish rather than guessing a
        // sleep long enough to outlast it.
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if handle.menu().iter().any(|e| e.id == id) {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the worker must publish the registration even though its caller disconnected mid-flight");
    }

    /// The delete side of the same property.
    #[tokio::test]
    async fn the_worker_publishes_a_deletion_whose_caller_disconnects_mid_flight() {
        let calls = Arc::new(AtomicUsize::new(0));
        let unregister_calls = Arc::new(AtomicUsize::new(0));
        let handle = Handle::spawn(
            CountingRegistrar {
                calls: calls.clone(),
                unregister_calls: unregister_calls.clone(),
                registered: Vec::new(),
                delay: Some(Duration::from_millis(80)),
            },
            (&info()).into(),
        );
        // Register normally first (awaited to completion, not aborted —
        // this one just pays the 80ms delay), so it's on the menu.
        let (status, entry) = handle.register(body_of(&prompt("to-delete"))).await.unwrap();
        assert_eq!(status, StatusCode::CREATED);
        assert!(handle.menu().iter().any(|e| e.id == entry.id));

        let id = entry.id.clone();
        let task = {
            let handle = handle.clone();
            tokio::spawn(async move { handle.unregister(id).await })
        };
        tokio::time::timeout(Duration::from_secs(2), async {
            while unregister_calls.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the generator must start processing the deletion");
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if !handle.menu().iter().any(|e| e.id == entry.id) {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the worker must publish the deletion even though its caller disconnected mid-flight");
    }
}
