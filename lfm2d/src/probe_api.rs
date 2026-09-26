//! `POST /v1/probe`: raw inference over exact text, on the daemon's own
//! stack. An INSTRUMENT, not a judgement API — it has no calibration
//! contract, and `docs/integration.md` invariants 5 and 8-11 do not apply
//! to it (invariant 13). It takes request text by design (exact bytes, or a
//! rendered chat turn), unlike `/v1/opinion`, whose questions come only from
//! a loaded spec's menu — see that module's docs for why that restriction
//! exists there and does not here.
//!
//! Ruled 2026-09-23, `docs/system1-split-plan.md` (git f9ca081) "Tokenize and probe
//! endpoints". Types and request validation live here, mirroring
//! [`crate::opinion_api`]; the engine (rendering, warm-prefix resume,
//! continuation scoring, greedy generation) lives in `adjudicator.rs`
//! beside `Adjudicator::opinion`/`describe_then_read`, which it reuses code
//! from — see [`crate::opinion::score_continuations_verbose`] and
//! [`crate::types::step_distribution_under`].
use serde::{Deserialize, Serialize};

/// Hard cap on `continuations`. Not a soft default — a request above this
/// is rejected, not silently truncated (matches
/// [`crate::types::MAX_DISTRIBUTION_TOP_K`]'s "reject, never clamp" rule).
pub const MAX_CONTINUATIONS: usize = 32;
/// Hard cap on `generate`. Deliberately far below `/v1/adjudicate`'s 2048:
/// a probe's `generate` is a diagnostic look at a few next steps under the
/// SAME sampling policy, not a substitute for a real generation.
pub const MAX_GENERATE: usize = 256;

fn default_top_k() -> usize {
    5
}
fn yes() -> bool {
    true
}
fn default_timeout() -> u64 {
    30000
}

/// One chat turn for `messages`. Rendered through the same
/// `<|im_start|>{role}\n{content}<|im_end|>\n` shape
/// [`crate::adjudicator::PromptSpec`] uses for its own turns — see
/// [`ProbeRequest::render`].
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeMessage {
    pub role: String,
    pub content: String,
}

/// Roles the renderer accepts. A probe's `messages` render into the SAME
/// `<|im_start|>{role}\n` shape a boot prompt spec's system/user turns use;
/// admitting an arbitrary role string would let a caller spell a turn
/// boundary the checkpoint's template was never validated against.
const ALLOWED_ROLES: &[&str] = &["system", "user", "assistant"];

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeRequest {
    /// Exact bytes, fed as-is: no chat template, no control-token
    /// restriction (unlike every other prompt path in this crate, which
    /// refuses [`crate::chat::CONTROL_MARKERS`] in content —
    /// [`crate::adjudicator::validate_text`]). A probe exists partly to let
    /// a caller see what the model does with exactly the bytes it hands it,
    /// including ones the generative/opinion paths would refuse to render.
    ///
    /// **The text-form caveat (kaibo review, 2026-09-23, F3):** `text` and
    /// `messages` are both tokenized FRESH by this endpoint. That is exact
    /// for bytes a caller wrote itself (there is only one canonical
    /// encoding of a fixed string), but NOT for bytes that include a
    /// model's own generated output: BPE is not injective in the other
    /// direction, so decoding a token sequence to text and re-encoding
    /// that text is not guaranteed to reproduce the SAME token ids the
    /// model actually sampled — `describe_then_read`'s own "the model
    /// leaving the canonical path" case exists for exactly this reason.
    /// Measured to matter in practice on `decode_from`'s replay of
    /// `/v1/opinion`'s decode loop; use [`Self::ids`] instead whenever the
    /// bytes being replayed came out of a generation (`/v1/opinion`'s
    /// `rendered_token_ids`, present when `rendered: true` was asked).
    /// Exactly one of `text`/`messages`/`ids` must be given.
    #[serde(default)]
    pub text: Option<String>,
    /// Rendered through the checkpoint's turn shape and an opened assistant
    /// turn; see [`ProbeRequest::render`]. Same text-form caveat as `text`.
    /// Exactly one of `text`/`messages`/`ids` must be given.
    #[serde(default)]
    pub messages: Option<Vec<ProbeMessage>>,
    /// Exact token ids, fed as-is: no tokenization at all, so this is
    /// immune to the text-form caveat above — the ids a caller holds are
    /// teacher-forced VERBATIM, never re-derived from decoded text. The
    /// response still echoes a decoded `rendered` string (for a human to
    /// read) and its `rendered_sha256`, but neither is used for anything
    /// but that echo. `decode_from_token` is this form's schedule split
    /// (a token INDEX, not a byte offset — see [`Self::decode_from_token`]).
    /// Exactly one of `text`/`messages`/`ids` must be given.
    #[serde(default)]
    pub ids: Option<Vec<u32>>,
    /// Text appended after the opened assistant turn, before anything is
    /// read or generated. Only valid alongside `messages`.
    #[serde(default)]
    pub assistant_prefill: Option<String>,
    /// Full-vocabulary top-k logprobs at the last input position. `0` is a
    /// valid request meaning "no top-k" — same convention as
    /// [`crate::types::DistributionRequest::top_k`], whose cap this reuses.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Each teacher-forced after the input and scored like an opinion
    /// option: per-token logprobs, sequence logprob, first-token logprob.
    /// An instrument like `text` and `ids`, so control markers are allowed:
    /// scoring `<|im_end|>` is how a caller asks whether the turn ends here.
    #[serde(default)]
    pub continuations: Vec<String>,
    /// Greedy steps to take after the input, under the SAME deterministic
    /// sampling policy (repetition penalty, no grammar) production uses.
    /// Stops early at end-of-text.
    #[serde(default)]
    pub generate: usize,
    /// When true and the input's token ids start with a loaded spec's
    /// resident prefix, resume from that spec's prefix state instead of
    /// running cold from zero. Always reported back, never silently assumed
    /// — see [`crate::opinion_api`]'s cache-outcome convention, which this
    /// mirrors at one layer instead of three.
    #[serde(default = "yes")]
    pub use_cache: bool,
    /// The `text`/`messages` form's schedule split: a byte offset into the
    /// rendered input (the same bytes `rendered_sha256` hashes) marking
    /// where the daemon's schedule switches from bulk prefill to replaying
    /// the decode loop's own forward call, one token at a time — see
    /// [`ProbeCache`]. Must land exactly on a token boundary of THIS
    /// endpoint's own fresh tokenization of the input (see the text-form
    /// caveat on [`Self::text`] for what that can and can't promise); a
    /// non-boundary offset is refused with the nearest boundaries named,
    /// never rounded silently. Only valid with `text`/`messages` — use
    /// [`Self::decode_from_token`] with `ids`. `None` (the default) is
    /// today's behavior: the whole input is bulk-forwarded, matching
    /// `POST /v1/adjudicate {"opinion": true}`'s own schedule, NOT
    /// `POST /v1/opinion`'s — reproducing that one needs a schedule split
    /// set to where its generation began (`docs/lfm25-adjudicator.md`
    /// "Probe and tokenize").
    #[serde(default)]
    pub decode_from: Option<usize>,
    /// The `ids` form's schedule split: a TOKEN INDEX (not a byte offset)
    /// into `ids` — trivially exact, since `ids` carries no tokenization
    /// ambiguity to land wrong on. `0..=ids.len()`; out of range is
    /// refused. Only valid with `ids` — use [`Self::decode_from`] with
    /// `text`/`messages`.
    #[serde(default)]
    pub decode_from_token: Option<usize>,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
}

impl ProbeRequest {
    /// Shape checks that need no model: everything a handler can refuse
    /// before ever reaching the worker queue.
    pub fn validate(&self) -> Result<(), String> {
        let given = [self.text.is_some(), self.messages.is_some(), self.ids.is_some()];
        if given.iter().filter(|&&g| g).count() != 1 {
            return Err("give exactly one of text, messages, or ids".into());
        }
        if let Some(text) = &self.text {
            if text.is_empty() {
                return Err("text must not be empty".into());
            }
            if text.len() > crate::adjudicator::MAX_INPUT_BYTES {
                return Err(format!("text exceeds {} bytes", crate::adjudicator::MAX_INPUT_BYTES));
            }
            if self.assistant_prefill.is_some() {
                return Err("assistant_prefill only applies to messages, not text".into());
            }
        }
        if let Some(messages) = &self.messages {
            if messages.is_empty() {
                return Err("messages must not be empty".into());
            }
            for m in messages {
                if !ALLOWED_ROLES.contains(&m.role.as_str()) {
                    return Err(format!("message role {:?} must be one of {ALLOWED_ROLES:?}", m.role));
                }
                crate::adjudicator::validate_text(&m.content).map_err(|e| format!("message content: {e}"))?;
            }
            if let Some(prefill) = &self.assistant_prefill {
                crate::adjudicator::validate_text(prefill).map_err(|e| format!("assistant_prefill: {e}"))?;
            }
        }
        if let Some(ids) = &self.ids {
            if ids.is_empty() {
                return Err("ids must not be empty".into());
            }
            if self.assistant_prefill.is_some() {
                return Err("assistant_prefill only applies to messages, not ids".into());
            }
            if self.decode_from.is_some() {
                return Err("decode_from applies to text/messages; use decode_from_token with ids".into());
            }
            if let Some(at) = self.decode_from_token
                && at > ids.len()
            {
                return Err(format!("decode_from_token must be 0..={} (ids.len()), got {at}", ids.len()));
            }
        } else if self.decode_from_token.is_some() {
            return Err("decode_from_token applies to ids; use decode_from with text/messages".into());
        }
        if self.top_k > crate::types::MAX_DISTRIBUTION_TOP_K {
            return Err(format!(
                "top_k must be 0..={}",
                crate::types::MAX_DISTRIBUTION_TOP_K
            ));
        }
        if self.continuations.len() > MAX_CONTINUATIONS {
            return Err(format!("continuations must have at most {MAX_CONTINUATIONS}"));
        }
        if self.continuations.iter().any(String::is_empty) {
            return Err("continuations must not contain an empty string".into());
        }
        if self.generate > MAX_GENERATE {
            return Err(format!("generate must be 0..={MAX_GENERATE}"));
        }
        if self.timeout_ms == 0 || self.timeout_ms > 120000 {
            return Err("timeout_ms must be 1..=120000".into());
        }
        Ok(())
    }

    /// The exact bytes fed to the model: `text` verbatim, or `messages`
    /// rendered through the checkpoint's turn shape (`<|startoftext|>` once,
    /// then one `<|im_start|>{role}\n{content}<|im_end|>\n` per message,
    /// then an opened, unclosed assistant turn — optionally continued by
    /// `assistant_prefill`). The SAME renderer
    /// [`crate::adjudicator::PromptSpec::render_prefix`]/`render_user_turn`
    /// use for a spec's system/user turns, reused rather than reimplemented
    /// — see that module's docs on why a hand-rolled second template
    /// renderer is exactly the mistake `docs/lfm25-adjudicator.md`'s
    /// harness rules warn against ("An eval harness renders the prompt
    /// exactly as the daemon does").
    ///
    /// # Panics
    /// If neither `text` nor `messages` is set — [`Self::validate`] refuses
    /// that shape first, so a caller that calls this without validating
    /// first has already broken its own contract.
    pub fn render(&self) -> String {
        match (&self.text, &self.messages) {
            (Some(text), None) => text.clone(),
            (None, Some(messages)) => {
                render_messages(messages, self.assistant_prefill.as_deref())
            }
            _ => panic!("ProbeRequest::render called on an unvalidated request"),
        }
    }
}

/// `<|startoftext|>`, then one turn per message, then an opened assistant
/// turn optionally continued by `prefill`. Exposed (not just inlined into
/// [`ProbeRequest::render`]) so a test can pin the exact bytes independent
/// of request validation.
pub fn render_messages(messages: &[ProbeMessage], prefill: Option<&str>) -> String {
    let mut out = String::from("<|startoftext|>");
    for m in messages {
        out.push_str("<|im_start|>");
        out.push_str(&m.role);
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    if let Some(prefill) = prefill {
        out.push_str(prefill);
    }
    out
}

/// Model/tokenizer/backend identity, independent of any spec — a probe is
/// not bound to one. Mirrors [`crate::adjudicator::PrefixInfo`]'s identity
/// fields minus the spec-specific ones (`snapshot_id`, `prefix_tokens`,
/// `context_limit`), which have no meaning for arbitrary probed text.
#[derive(Clone, Debug, Serialize)]
pub struct ProbeIdentity {
    pub model_id: String,
    pub weight_hash: String,
    pub tokenizer_hash: String,
    pub backend: String,
    pub device: String,
    pub candle_rev: String,
    pub dtype: String,
    pub sampling: String,
}

/// Which schedule this probe actually ran, always stated — never left for
/// the caller to infer from `use_cache` alone, since a `use_cache: true`
/// request against text with no matching resident prefix still runs cold.
#[derive(Clone, Debug, Serialize)]
pub struct ProbeCache {
    /// Whether ANY warm-prefix resume happened. `false` whenever
    /// `use_cache: false` was requested OR no loaded spec's prefix matched.
    pub used_cache: bool,
    /// The spec whose resident prefix was resumed from, when one was.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resumed_spec: Option<String>,
    /// How many of the input's tokens came from the resumed prefix rather
    /// than being forwarded now. `0` on a cold run.
    pub cached_tokens: usize,
    /// Tokens forwarded IN BULK (chunk-sized pieces): the resumed prefix's
    /// own `cached_tokens` plus any further tokens bulk-forwarded up to
    /// `decode_from` (or the whole input, when `decode_from` was not
    /// given — today's default schedule, the one
    /// `POST /v1/adjudicate {"opinion": true}` also runs).
    pub prefill_tokens: usize,
    /// Tokens forwarded ONE AT A TIME after `decode_from`, replaying
    /// `describe_then_read`'s own decode-loop forward call. `0` when
    /// `decode_from` was not given.
    pub stepwise_tokens: usize,
}

/// One continuation's teacher-forced read.
#[derive(Clone, Debug, Serialize)]
pub struct ProbeContinuationScore {
    pub text: String,
    /// The continuation's own canonical tokens (`suffix_is_stable`'s
    /// alone-encoding) — what was actually teacher-forced.
    pub tokens: Vec<u32>,
    /// RAW per-token logprob, aligned with `tokens`.
    pub token_logprobs: Vec<f32>,
    /// Sum of `token_logprobs`. `<= 0`.
    pub sequence_logprob: f32,
    /// `token_logprobs[0]` — the F9 slot-score number, named directly so a
    /// caller doesn't have to know that convention to find it.
    pub first_logprob: f32,
    /// Whether this continuation's tokens, encoded alone, are exactly the
    /// tail of the rendered input's tokens with this text appended
    /// ([`crate::adjudicator::suffix_is_stable`]). `false` means the tokens
    /// scored are NOT the ones the model would actually write continuing
    /// this exact input — still reported (never hidden), because that fact
    /// is itself diagnostic.
    pub canonical: bool,
}

/// Every requested continuation's read, plus the one renormalized number
/// this endpoint allows: `prob` over `continuations`, beside the raw
/// `sequence_mass` it is blind to "never asked" without
/// (`docs/lfm25-adjudicator.md` "Opinion reads").
#[derive(Clone, Debug, Serialize)]
pub struct ProbeContinuations {
    pub options: Vec<ProbeContinuationScore>,
    /// `logsumexp` of the options' sequence logprobs.
    pub sequence_mass: f32,
    /// Renormalized over `options`, same order.
    pub prob: Vec<f32>,
}

/// One greedy-generated step, decoded under the same policy production
/// uses (repetition penalty, no grammar).
#[derive(Clone, Debug, Serialize)]
pub struct ProbeGeneratedStep {
    pub token: u32,
    pub text: String,
    /// RAW logprob of the sampled token (before the repetition penalty,
    /// which only affects which token was CHOSEN, not the reported number —
    /// same convention as `/v1/adjudicate`'s `distributions`).
    pub logprob: f32,
    pub top_logprobs: Vec<crate::types::TokenLogprob>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProbeResponse {
    #[serde(flatten)]
    pub identity: ProbeIdentity,
    /// The rendered text, only when `messages` was used — `text` requests
    /// already have it verbatim in their own request body, so echoing it
    /// back would be pure duplication; `rendered_sha256` below is always
    /// present either way (the audit trail this crate uses everywhere
    /// instead of repeating text it doesn't need to).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered: Option<String>,
    pub rendered_sha256: String,
    pub input_tokens: usize,
    pub cache: ProbeCache,
    /// Top-k at the last input position. Empty when `top_k: 0` was asked.
    pub top_logprobs: Vec<crate::types::TokenLogprob>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continuations: Option<ProbeContinuations>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub generated: Vec<ProbeGeneratedStep>,
    pub queue_ms: f64,
    pub prefill_ms: f64,
    pub score_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_req() -> ProbeRequest {
        ProbeRequest {
            text: Some("cargo clean".into()),
            messages: None,
            ids: None,
            assistant_prefill: None,
            top_k: 5,
            continuations: vec![],
            generate: 0,
            use_cache: true,
            decode_from: None,
            decode_from_token: None,
            timeout_ms: 30000,
        }
    }

    fn ids_req() -> ProbeRequest {
        ProbeRequest { text: None, ids: Some(vec![1, 2, 3]), ..text_req() }
    }

    #[test]
    fn requires_exactly_one_of_text_or_messages() {
        let mut req = text_req();
        req.text = None;
        assert!(req.validate().unwrap_err().contains("exactly one"));

        let mut req = text_req();
        req.messages = Some(vec![ProbeMessage { role: "user".into(), content: "hi".into() }]);
        assert!(req.validate().unwrap_err().contains("exactly one"), "both given must be refused too");

        let mut req = text_req();
        req.ids = Some(vec![1, 2, 3]);
        assert!(req.validate().unwrap_err().contains("exactly one"), "text and ids together must be refused");

        assert!(text_req().validate().is_ok());
        assert!(ids_req().validate().is_ok());
    }

    #[test]
    fn empty_ids_is_refused() {
        let mut req = ids_req();
        req.ids = Some(vec![]);
        assert!(req.validate().is_err());
    }

    #[test]
    fn decode_from_and_decode_from_token_are_form_specific() {
        let mut req = text_req();
        req.decode_from_token = Some(1);
        assert!(req.validate().unwrap_err().contains("decode_from_token applies to ids"));

        let mut req = ids_req();
        req.decode_from = Some(1);
        assert!(req.validate().unwrap_err().contains("decode_from applies to text/messages"));

        let mut req = ids_req();
        req.decode_from_token = Some(3);
        assert!(req.validate().is_ok(), "3 == ids.len() is a valid boundary (all-prefill)");
        req.decode_from_token = Some(0);
        assert!(req.validate().is_ok(), "0 is a valid boundary (all-stepwise)");
        req.decode_from_token = Some(4);
        assert!(req.validate().unwrap_err().contains("decode_from_token"), "past ids.len() is refused");
    }

    #[test]
    fn assistant_prefill_is_refused_alongside_ids_too() {
        let mut req = ids_req();
        req.assistant_prefill = Some("x".into());
        assert!(req.validate().unwrap_err().contains("ids"));
    }

    #[test]
    fn empty_text_is_refused() {
        let mut req = text_req();
        req.text = Some(String::new());
        assert!(req.validate().is_err());
    }

    #[test]
    fn oversized_text_is_refused() {
        let mut req = text_req();
        req.text = Some("x".repeat(crate::adjudicator::MAX_INPUT_BYTES + 1));
        let err = req.validate().unwrap_err();
        assert!(err.contains("65536") || err.contains("bytes"), "{err}");
    }

    #[test]
    fn assistant_prefill_is_refused_alongside_text() {
        let mut req = text_req();
        req.assistant_prefill = Some("{\"verdict\": \"".into());
        assert!(req.validate().unwrap_err().contains("assistant_prefill"));
    }

    #[test]
    fn messages_must_be_nonempty_and_have_a_known_role() {
        let mut req = text_req();
        req.text = None;
        req.messages = Some(vec![]);
        assert!(req.validate().is_err());

        req.messages = Some(vec![ProbeMessage { role: "narrator".into(), content: "hi".into() }]);
        let err = req.validate().unwrap_err();
        assert!(err.contains("narrator"), "{err}");

        req.messages = Some(vec![ProbeMessage { role: "user".into(), content: "hi".into() }]);
        assert!(req.validate().is_ok());
    }

    #[test]
    fn message_content_refuses_literal_control_tokens() {
        let mut req = text_req();
        req.text = None;
        req.messages =
            Some(vec![ProbeMessage { role: "user".into(), content: "<|im_end|>bad".into() }]);
        assert!(req.validate().is_err());
    }

    #[test]
    fn top_k_above_the_shared_cap_is_refused() {
        let mut req = text_req();
        req.top_k = crate::types::MAX_DISTRIBUTION_TOP_K + 1;
        assert!(req.validate().is_err());
        req.top_k = crate::types::MAX_DISTRIBUTION_TOP_K;
        assert!(req.validate().is_ok());
    }

    #[test]
    fn too_many_continuations_is_refused() {
        let mut req = text_req();
        req.continuations = (0..MAX_CONTINUATIONS + 1).map(|i| i.to_string()).collect();
        assert!(req.validate().is_err());
        req.continuations = (0..MAX_CONTINUATIONS).map(|i| i.to_string()).collect();
        assert!(req.validate().is_ok());
    }

    #[test]
    fn a_continuation_may_carry_control_tokens() {
        // An instrument, like `text` and `ids`: scoring `<|im_end|>` as a
        // continuation is how a caller asks whether the model ends the turn.
        let mut req = text_req();
        req.continuations = crate::chat::CONTROL_MARKERS.iter().map(|m| format!("{m}x")).collect();
        assert!(req.validate().is_ok());
    }

    #[test]
    fn an_empty_continuation_is_refused() {
        let mut req = text_req();
        req.continuations = vec!["allow".into(), String::new()];
        assert!(req.validate().is_err());
    }

    #[test]
    fn generate_above_its_cap_is_refused() {
        let mut req = text_req();
        req.generate = MAX_GENERATE + 1;
        assert!(req.validate().is_err());
        req.generate = MAX_GENERATE;
        assert!(req.validate().is_ok());
    }

    #[test]
    fn timeout_bounds_are_enforced() {
        let mut req = text_req();
        req.timeout_ms = 0;
        assert!(req.validate().is_err());
        req.timeout_ms = 120_001;
        assert!(req.validate().is_err());
        req.timeout_ms = 120_000;
        assert!(req.validate().is_ok());
    }

    #[test]
    fn render_messages_opens_the_assistant_turn_and_appends_the_prefill() {
        let messages = vec![
            ProbeMessage { role: "system".into(), content: "sys".into() },
            ProbeMessage { role: "user".into(), content: "hi".into() },
        ];
        let rendered = render_messages(&messages, None);
        assert_eq!(
            rendered,
            "<|startoftext|><|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        let with_prefill = render_messages(&messages, Some("{\"verdict\": \""));
        assert!(with_prefill.ends_with("<|im_start|>assistant\n{\"verdict\": \""));
    }

    #[test]
    fn render_dispatches_to_text_or_rendered_messages() {
        let req = text_req();
        assert_eq!(req.render(), "cargo clean");

        let mut req = text_req();
        req.text = None;
        req.messages = Some(vec![ProbeMessage { role: "user".into(), content: "hi".into() }]);
        assert_eq!(req.render(), render_messages(req.messages.as_ref().unwrap(), None));
    }
}
