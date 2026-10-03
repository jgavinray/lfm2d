//! `/v1/opinion`: the typed-decision surface over the resident adjudicator.
//!
//! An opinion request names a loaded prompt spec, hands over the state (the
//! input and, optionally, the facts an app built for it) and asks one or
//! more of the spec's choice fields. The daemon renders the same prompt the
//! generative path renders, generates the fields that precede each question
//! under the output grammar — the *description* — stands at each asked
//! value slot as the walk passes it, and teacher-forces every option there
//! ([`crate::opinion`]). Between asked slots the walk writes what the model
//! writes; nothing after the LAST asked slot is decoded.
//!
//! Why describe first: a read at the verdict slot before the model has
//! described the input carries nothing on this checkpoint (F9, measured on
//! shell commands: AUC ~0.5 at 99% mass); after three described fields the
//! same slot separates the gold at AUC 0.74. The description is therefore part of the answer and is
//! echoed, so a caller can see and log what the read was conditioned on.
//!
//! What the response never carries: a winner. The caller picks, from its own
//! thresholds on its own data, reading the renormalised `prob` beside the
//! raw `sequence_mass` (a low mass means the model was never asked this
//! question here, and a renormalised number over it is noise that looks
//! like an answer). Questions come from the spec on disk, never from request
//! text: framing words in a rendered prompt were measured to move label
//! tokens by an order of magnitude, and a prompt the instruments never
//! scored is not one this daemon should serve.
//!
//! Vocabulary: this is the *opinion read* in code; "System 1" is discussion
//! language and belongs in the README.
use crate::adjudicator::{PrefixInfo, PromptSpec, validate_text};
use crate::opinion::OpinionRead;
use serde::{Deserialize, Serialize};

pub(crate) const MAX_STATE_BYTES: usize = 65536;

fn yes() -> bool {
    true
}
fn default_timeout() -> u64 {
    30000
}

/// The state an opinion is asked about. Rendered into the user turn exactly
/// as the evaluation harnesses render it, so a paired generative run and an
/// opinion read see the same bytes: the facts block verbatim (an app builds
/// it; the daemon never does), then the spec's `input_label`, a colon, a
/// newline, and the input.
///
/// A generative `/v1/adjudicate` escalation resumes from the described state
/// only on the exact same bytes, so a caller escalating an opinion sends
/// [`OpinionState::render`]'s output — label included — as its `input`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpinionState {
    pub input: String,
    /// Evidence about the input, prepended verbatim. Ends with a newline if
    /// the caller wants one between it and the label.
    #[serde(default)]
    pub facts: Option<String>,
}

impl OpinionState {
    /// `{facts}{label}:\n{input}`, with `label` from the spec
    /// ([`PromptSpec::input_label`]), never from the request.
    pub fn render(&self, label: &str) -> String {
        format!(
            "{}{label}:\n{}",
            self.facts.as_deref().unwrap_or(""),
            self.input
        )
    }
    fn validate(&self) -> Result<(), String> {
        validate_text(&self.input).map_err(|e| format!("state.input: {e}"))?;
        if let Some(facts) = &self.facts
            && crate::chat::control_marker_in(facts).is_some()
        {
            return Err("state.facts: literal model control tokens are not allowed".into());
        }
        // The caller's bytes only: the label is the spec's, and the rendered
        // prompt's length is checked against the context limit anyway.
        if self.facts.as_deref().map_or(0, str::len) + self.input.len() > MAX_STATE_BYTES {
            return Err(format!("state exceeds {MAX_STATE_BYTES} bytes"));
        }
        Ok(())
    }
}

/// A tail read's chat. The read renders, after the checkpoint's ids:
///
/// `<|im_start|>user\n{spec block}\n\n{facts}{input_label}:\n{input}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n`
///
/// where the spec block is the spec's system turn content (its `system`, then
/// `Return exactly one JSON object matching this schema: ` and the schema).
/// That shape was measured after chats with preserved reasoning: `{"` at
/// rank 1-2 (chat-tail probe, 2026-09-26). A spec with `tools` is refused:
/// its read turn was never measured. The checkpoint must be held; an
/// evicted one is a 404, never a rebuild.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpinionContext {
    pub checkpoint: String,
}

/// One question: a choice field of the spec, optionally narrowed to a subset
/// of its options. The subset is scored in the spec's order, not the
/// request's.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Question {
    pub field: String,
    #[serde(default)]
    pub options: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpinionRequest {
    /// A loaded spec's name (`GET /v1/opinion/specs`).
    pub spec: String,
    pub state: OpinionState,
    /// A chat to read the tail of: `{"checkpoint": id}`, a chat checkpoint
    /// `POST /v1/chat` left (`crate::chat_session`). The read's user turn
    /// then follows the chat instead of the spec's system turn, carrying
    /// the spec's instructions and schema itself ([`OpinionContext`]).
    /// `null` or absent: the spec's own prompt, as before.
    #[serde(default)]
    pub context: Option<OpinionContext>,
    /// Several held contexts to read after, 1 to [`MAX_CONTEXTS`] distinct
    /// ids (chat checkpoints or `/v1/contexts`): the same read as `context`
    /// once per id, serially in request order, then each question pooled
    /// across them ([`crate::pool`]). The answer is an
    /// [`OpinionMultiResponse`]. Exclusive with `context`.
    #[serde(default)]
    pub contexts: Option<Vec<String>>,
    /// How `contexts` pool; defaults to linear, uniform. Only with `contexts`.
    #[serde(default)]
    pub pool: Option<crate::pool::PoolSettings>,
    /// One or more choice fields. They share one description: the engine
    /// walks the grammar once and reads each slot as it passes it, so the
    /// answers come back in emission order whatever order they were asked.
    pub questions: Vec<Question>,
    /// Return the exact text the options continue (`rendered` on the
    /// response). Off by default: `rendered_sha256` is the audit trail, and
    /// the text carries the whole spec prefix on every request.
    #[serde(default)]
    pub rendered: bool,
    #[serde(default = "yes")]
    pub use_cache: bool,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
}

impl OpinionRequest {
    /// Shape checks that need no spec: the menu check is
    /// [`SpecMenuEntry::resolve`].
    pub fn validate(&self) -> Result<(), String> {
        if self.spec.is_empty() {
            return Err("spec must name a loaded prompt spec".into());
        }
        self.state.validate()?;
        if let Some(context) = &self.context
            && !crate::chat_session::is_checkpoint_id(&context.checkpoint)
        {
            return Err("context.checkpoint must be a checkpoint id: 64 lowercase hex digits".into());
        }
        match &self.contexts {
            Some(_) if self.context.is_some() => {
                return Err("give `context` (one) or `contexts` (several), not both".into());
            }
            Some(ids) => {
                if ids.is_empty() || ids.len() > MAX_CONTEXTS {
                    return Err(format!("contexts takes 1 to {MAX_CONTEXTS} ids, got {}", ids.len()));
                }
                if let Some(bad) = ids.iter().find(|id| !crate::chat_session::is_checkpoint_id(id)) {
                    return Err(format!("contexts: {bad:?} is not a checkpoint id (64 lowercase hex digits)"));
                }
                let distinct: std::collections::BTreeSet<&String> = ids.iter().collect();
                if distinct.len() != ids.len() {
                    return Err("contexts must be distinct: the same content is one context, and the pool \
                                would count its voice twice"
                        .into());
                }
                if let Some(pool) = &self.pool {
                    pool.validate(ids.len()).map_err(|e| format!("pool: {e}"))?;
                }
            }
            None if self.pool.is_some() => {
                return Err("pool takes `contexts`; one read has nothing to pool".into());
            }
            None => {}
        }
        if self.questions.is_empty() {
            return Err("ask at least one question".into());
        }
        if self.timeout_ms == 0 || self.timeout_ms > 120000 {
            return Err("timeout_ms must be 1..=120000".into());
        }
        Ok(())
    }
}

/// What kind of field a spec's schema declares, as the menu reports it.
/// Only `choice` fields can be asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldKind {
    Text,
    Choice,
    Boolean,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInfo {
    pub field: String,
    pub kind: FieldKind,
    /// The enum, in schema order; empty unless `kind` is `choice`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
}

/// One loaded spec as `GET /v1/opinion/specs` lists it. Read this at
/// runtime; never hard-code a field name or an option.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpecMenuEntry {
    /// Lowercase hex sha256 of the exact bytes this spec was loaded from —
    /// kaibo's CAS digest and `lfm2d::hash::sha256_hex_bytes`. Identity for
    /// every spec, boot or uploaded; `POST /v1/opinion` and
    /// `POST /v1/adjudicate` accept it wherever they accept `spec`. See
    /// `docs/system1-split-plan.md` (git f9ca081) "Runtime spec registration".
    pub id: String,
    /// A boot-time spec's file stem (`--opinion-spec` specs are named this
    /// way and answer to it as well as to `id`). An uploaded spec has no
    /// file, so this equals `id`.
    pub spec: String,
    /// What the spec's user turn calls its input ([`PromptSpec::input_label`]):
    /// `state.input` renders after `{input_label}:` and a newline. Read it
    /// here rather than assuming one.
    pub input_label: String,
    pub snapshot_id: String,
    /// How many described states the daemon keeps for this spec.
    pub described_cache_capacity: usize,
    /// The schema's fields in `required` order, which is emission order.
    pub fields: Vec<FieldInfo>,
}

/// A question checked against its spec: what the engine generates before
/// the slot, the options it scores there, and the bytes that close each one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedQuestion {
    pub field: String,
    /// Fields generated before the question, in order.
    pub describe: Vec<String>,
    /// Options to score, in spec order.
    pub options: Vec<String>,
    /// Written after each option: the grammar's separator when another field
    /// follows, the object's close when this is the last field.
    pub close: String,
}

impl ResolvedQuestion {
    /// The text the generation must end with to stand at the slot: the key
    /// as the grammar writes it (`serde_json` escaping, so a quote or a
    /// backslash in a field name still matches), the separator, the value's
    /// opening quote.
    pub fn slot_text(&self) -> String {
        format!(
            "{}: \"",
            serde_json::to_string(&self.field).expect("a string always serializes")
        )
    }
}

impl SpecMenuEntry {
    /// Read the menu entry off a spec. A spec without an output schema lists
    /// no fields and can answer no question. `id` is the content hash
    /// (`lfm2d::hash::sha256_hex_bytes` of the exact uploaded/loaded bytes);
    /// `name` is the boot-time file stem, or `id` again for an upload.
    pub fn from_prompt(
        id: &str,
        name: &str,
        prompt: &PromptSpec,
        snapshot_id: &str,
        described_cache_capacity: usize,
    ) -> Result<Self, String> {
        let mut fields = Vec::new();
        if let Some(schema) = &prompt.output_schema {
            let order = schema["required"]
                .as_array()
                .ok_or("output_schema has no required list")?;
            let properties = schema["properties"]
                .as_object()
                .ok_or("output_schema has no properties")?;
            for name in order {
                let name = name
                    .as_str()
                    .ok_or("required field names must be strings")?;
                // The engine finds a slot by the text ending with the key as
                // the grammar writes it. An escaped quote can make an EARLIER
                // key's text end with a later field's slot (`"x\"b": "`
                // ends with `"b": "`), and the walk would stop there and
                // read the wrong state without an error. Refused at load.
                if name.contains('"') || name.contains('\\') {
                    return Err(format!(
                        "field name {name:?} contains a quote or backslash; slot detection \
                         cannot tell its key from a later field's"
                    ));
                }
                let spec = properties
                    .get(name)
                    .ok_or_else(|| format!("required names a missing property {name:?}"))?;
                let (kind, options) = if spec["type"] == "boolean" {
                    (FieldKind::Boolean, vec![])
                } else if let Some(choices) = spec.get("enum").and_then(|e| e.as_array()) {
                    let options = choices
                        .iter()
                        .map(|c| c.as_str().map(str::to_string))
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| format!("field {name:?} has a non-string enum"))?;
                    (FieldKind::Choice, options)
                } else {
                    (FieldKind::Text, vec![])
                };
                fields.push(FieldInfo {
                    field: name.to_string(),
                    kind,
                    options,
                });
            }
        }
        Ok(Self {
            id: id.to_string(),
            spec: name.to_string(),
            input_label: prompt.input_label.clone(),
            snapshot_id: snapshot_id.to_string(),
            described_cache_capacity,
            fields,
        })
    }

    /// Resolve every question and put them in emission order, the order the
    /// engine meets their slots. A field asked twice is refused: its two
    /// reads would be the same slot.
    pub fn resolve_all(&self, questions: &[Question]) -> Result<Vec<ResolvedQuestion>, String> {
        if questions.is_empty() {
            return Err("ask at least one question".into());
        }
        let mut resolved = Vec::with_capacity(questions.len());
        for question in questions {
            if resolved
                .iter()
                .any(|r: &ResolvedQuestion| r.field == question.field)
            {
                return Err(format!("field {:?} is asked more than once", question.field));
            }
            resolved.push(self.resolve(question)?);
        }
        resolved.sort_by_key(|r| r.describe.len());
        Ok(resolved)
    }

    /// Check a question against this spec and fix what the engine needs.
    pub fn resolve(&self, question: &Question) -> Result<ResolvedQuestion, String> {
        let position = self
            .fields
            .iter()
            .position(|f| f.field == question.field)
            .ok_or_else(|| format!("spec {:?} has no field {:?}", self.spec, question.field))?;
        let field = &self.fields[position];
        if field.kind != FieldKind::Choice {
            return Err(format!(
                "field {:?} is {:?}; only a choice field can be asked",
                question.field,
                serde_json::to_value(field.kind).map_err(|e| e.to_string())?
            ));
        }
        let options: Vec<String> = match &question.options {
            None => field.options.clone(),
            Some(subset) => {
                let mut seen = std::collections::BTreeSet::new();
                for option in subset {
                    if !field.options.contains(option) {
                        return Err(format!(
                            "field {:?} has no option {option:?}",
                            question.field
                        ));
                    }
                    if !seen.insert(option) {
                        return Err(format!("option {option:?} repeats"));
                    }
                }
                field
                    .options
                    .iter()
                    .filter(|o| seen.contains(o))
                    .cloned()
                    .collect()
            }
        };
        if options.len() < 2 {
            return Err("a question needs at least two options".into());
        }
        let last = position + 1 == self.fields.len();
        Ok(ResolvedQuestion {
            field: question.field.clone(),
            describe: self.fields[..position]
                .iter()
                .map(|f| f.field.clone())
                .collect(),
            options,
            close: if last { "\"}" } else { "\"," }.into(),
        })
    }
}

/// One described field, in emission order. A list rather than an object
/// because `serde_json` sorts object keys, and the order here is evidence.
#[derive(Clone, Debug, Serialize)]
pub struct DescribedField {
    pub field: String,
    pub value: serde_json::Value,
}

/// One question's read at its slot. `margin` is the renormalised gap between
/// the top two options — the one Jev-shaped number added here — and is blind
/// to "never asked", which is what `sequence_mass` beside it is for.
#[derive(Clone, Debug, Serialize)]
pub struct Answer {
    pub field: String,
    #[serde(flatten)]
    pub read: OpinionRead,
    pub margin: f32,
}

/// Each cache layer's outcome for this request: `hit`, `miss`, `bypass`
/// (the request asked for a cold path) or `skipped` (a described hit never
/// consults the prompt checkpoint).
#[derive(Clone, Debug, Serialize)]
pub struct CacheOutcome {
    /// The spec's resident system prefix; `checkpoint` for a tail read,
    /// which forks a chat checkpoint and never uses the spec's prefix.
    pub prefix: String,
    /// The exact rendered prompt, before any description.
    pub state: String,
    /// The prompt plus its generated description, at the slot.
    pub described: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpinionResponse {
    #[serde(flatten)]
    pub prefix: PrefixInfo,
    pub spec: String,
    /// The request's `context`, echoed: which chat checkpoint this read
    /// forked. Absent for a read of the spec's own prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<OpinionContext>,
    /// The fields generated before the LAST asked slot, as the model wrote
    /// them; an earlier answer was read on a prefix of this description.
    pub described: Vec<DescribedField>,
    pub answers: Vec<Answer>,
    /// Only when the request asked: the rendered prompt, chat-template
    /// control tokens included, plus the description up to and including
    /// the LAST asked slot. Each answer's `rendered_sha256` hashes this text
    /// cut at the end of its own slot, so only the last answer's hash is of
    /// the whole string. Absent, not null, when unasked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered: Option<String>,
    /// The SAME text `rendered` carries, as the model's own token ids —
    /// prefix + user turn (a plain tokenization: this part was never
    /// generated, so encoding it is unambiguous) followed by every field
    /// generated before the LAST asked slot, EXACTLY as sampled, never a
    /// re-tokenization of the decoded text. Present under the same
    /// condition as `rendered` (kaibo review, 2026-09-23, F3): BPE is not
    /// injective, so a caller that re-encodes `rendered`'s TEXT can
    /// silently land on a different token path than the model actually
    /// took — these ids are what closes that gap, e.g. as
    /// `POST /v1/probe`'s `ids`/`decode_from_token` (`crate::probe_api`,
    /// "the text-form caveat").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered_token_ids: Option<Vec<u32>>,
    pub cache: CacheOutcome,
    /// Tokens in the rendered prompt (prefix and user turn), before the
    /// description.
    pub prompt_tokens: usize,
    /// How many of those were already resident.
    pub cached_tokens: usize,
    /// Tokens the description took, whether generated now or served from
    /// the described cache (`cache.described` says which).
    pub described_tokens: usize,
    pub queue_ms: f64,
    pub prefill_ms: f64,
    pub describe_ms: f64,
    pub read_ms: f64,
}

/// The most contexts one `/v1/opinion` reads after.
pub const MAX_CONTEXTS: usize = 8;

impl OpinionRequest {
    /// This request as one tail read after `checkpoint`: what each of a
    /// multi-context read's reads computes.
    pub fn for_context(&self, checkpoint: &str) -> Self {
        Self {
            context: Some(OpinionContext { checkpoint: checkpoint.to_owned() }),
            contexts: None,
            pool: None,
            ..self.clone()
        }
    }
}

/// One question pooled across a multi-context read's reads.
#[derive(Clone, Debug, Serialize)]
pub struct PooledAnswer {
    pub field: String,
    /// The options `probs` (and each `leave_one_out` row) follow.
    pub options: Vec<String>,
    #[serde(flatten)]
    pub pooled: crate::pool::Pooled,
}

/// `/v1/opinion` with `contexts`: every context's own read, in request
/// order, and each question pooled across them. Like a single read, no
/// winner: the caller picks from its own thresholds, and a pooled
/// probability is not calibrated because its inputs were.
#[derive(Clone, Debug, Serialize)]
pub struct OpinionMultiResponse {
    pub spec: String,
    pub contexts: Vec<String>,
    /// `reads[c]` is the read after `contexts[c]`, exactly the answer
    /// `context: {checkpoint: contexts[c]}` gives.
    pub reads: Vec<OpinionResponse>,
    /// One per question, in the reads' answer order.
    pub pooled: Vec<PooledAnswer>,
    /// The pool settings, defaults filled in.
    pub pool: crate::pool::PoolSettings,
    pub queue_ms: f64,
}

/// Pool each question across `reads` (one per context, request order):
/// each context's raw option logprobs and raw mass (`exp(sequence_mass)`).
/// Every read must have answered the same questions with the same options.
pub fn pool_reads(
    reads: &[OpinionResponse],
    settings: &crate::pool::PoolSettings,
) -> Result<Vec<PooledAnswer>, String> {
    let first = reads.first().ok_or("no reads to pool")?;
    first
        .answers
        .iter()
        .enumerate()
        .map(|(q, answer)| {
            let options: Vec<String> = answer.read.options.iter().map(|o| o.option.clone()).collect();
            let mut logprobs = Vec::with_capacity(reads.len());
            let mut mass = Vec::with_capacity(reads.len());
            for (c, read) in reads.iter().enumerate() {
                let other = read
                    .answers
                    .get(q)
                    .filter(|a| a.field == answer.field)
                    .ok_or_else(|| format!("read {c} did not answer {:?} in place {q}", answer.field))?;
                let names: Vec<&String> = other.read.options.iter().map(|o| &o.option).collect();
                if names != options.iter().collect::<Vec<_>>() {
                    return Err(format!("read {c} scored other options for {:?}", answer.field));
                }
                logprobs.push(other.read.options.iter().map(|o| f64::from(o.logprob)).collect());
                mass.push(f64::from(other.read.sequence_mass).exp());
            }
            let pooled = crate::pool::pool(&logprobs, &mass, settings)
                .map_err(|e| format!("{:?}: can't pool these reads: {e}", answer.field))?;
            Ok(PooledAnswer { field: answer.field.clone(), options, pooled })
        })
        .collect()
}

/// `prob[top] - prob[runner-up]` over renormalised option probabilities.
pub fn margin(probs: &[f32]) -> f32 {
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    match sorted.as_slice() {
        [top, next, ..] => top - next,
        [_] => 1.0,
        [] => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An opinion escalates by sending its rendered state as
    /// `/v1/adjudicate`'s `input`, so the largest state this route accepts,
    /// rendered under the longest label a spec may carry, must fit there.
    #[test]
    fn a_full_size_state_still_fits_adjudicate_once_rendered() {
        let largest_rendered = MAX_STATE_BYTES + crate::adjudicator::MAX_INPUT_LABEL_BYTES + ":\n".len();
        assert!(
            crate::adjudicator::MAX_INPUT_BYTES >= largest_rendered,
            "{} < {largest_rendered}: a full-size opinion could not be escalated",
            crate::adjudicator::MAX_INPUT_BYTES
        );
    }

    #[test]
    fn slot_text_is_the_key_as_the_grammar_writes_it() {
        let q = |field: &str| ResolvedQuestion {
            field: field.into(),
            describe: vec![],
            options: vec!["a".into(), "b".into()],
            close: "\"}".into(),
        };
        assert_eq!(q("verdict").slot_text(), "\"verdict\": \"");
        // A quote or a backslash in a field name is escaped exactly as the
        // grammar (serde_json) writes the key, so the stop still matches.
        assert_eq!(q("a\"b").slot_text(), "\"a\\\"b\": \"");
        assert_eq!(q("a\\b").slot_text(), "\"a\\\\b\": \"");
    }

    fn entry() -> SpecMenuEntry {
        let choice = |f: &str, o: &[&str]| FieldInfo {
            field: f.into(),
            kind: FieldKind::Choice,
            options: o.iter().map(|s| s.to_string()).collect(),
        };
        SpecMenuEntry {
            id: "id-s".into(),
            spec: "s".into(),
            input_label: "Input".into(),
            snapshot_id: "x".into(),
            described_cache_capacity: 16,
            fields: vec![
                FieldInfo { field: "effect".into(), kind: FieldKind::Text, options: vec![] },
                choice("scope", &["nothing", "project", "home"]),
                choice("undo", &["easy", "hard"]),
                choice("verdict", &["allow", "ask", "review"]),
            ],
        }
    }

    #[test]
    fn several_questions_resolve_into_emission_order() {
        let q = |f: &str| Question { field: f.into(), options: None };
        let got = entry()
            .resolve_all(&[q("verdict"), q("scope"), q("undo")])
            .unwrap();
        let fields: Vec<&str> = got.iter().map(|r| r.field.as_str()).collect();
        assert_eq!(fields, ["scope", "undo", "verdict"], "answers follow the slots, not the request");
        assert_eq!(got[2].describe, ["effect", "scope", "undo"]);
        assert_eq!(got[2].close, "\"}");
        let err = entry().resolve_all(&[q("undo"), q("undo")]).unwrap_err();
        assert!(err.contains("more than once"), "{err}");
        assert!(entry().resolve_all(&[]).is_err());
        assert!(entry().resolve_all(&[q("scope"), q("effect")]).is_err(), "text is never asked");
    }

    #[test]
    fn margin_is_the_gap_between_the_top_two() {
        assert!((margin(&[0.7, 0.2, 0.1]) - 0.5).abs() < 1e-6);
        assert!((margin(&[0.1, 0.2, 0.7]) - 0.5).abs() < 1e-6);
        assert_eq!(margin(&[1.0]), 1.0);
        assert_eq!(margin(&[]), 0.0);
    }

    #[test]
    fn a_request_needs_at_least_one_question_and_a_well_formed_context() {
        let parse = |s: &str| serde_json::from_str::<OpinionRequest>(s).unwrap();
        let good = parse(r#"{"spec":"s","state":{"input":"x"},"questions":[{"field":"v"}]}"#);
        assert!(good.validate().is_ok());
        assert!(good.use_cache && good.timeout_ms == 30000);
        let two = parse(
            r#"{"spec":"s","state":{"input":"x"},"questions":[{"field":"v"},{"field":"w"}]}"#,
        );
        assert!(two.validate().is_ok(), "several questions share one description");
        let none = parse(r#"{"spec":"s","state":{"input":"x"},"questions":[]}"#);
        assert!(none.validate().is_err());
        for bad in [r#"1"#, r#"{"checkpoint":"abc"}"#, r#"{"checkpoint":"ABCDEF"}"#] {
            let json = format!(r#"{{"spec":"s","state":{{"input":"x"}},"context":{bad},"questions":[{{"field":"v"}}]}}"#);
            match serde_json::from_str::<OpinionRequest>(&json) {
                Ok(r) => assert!(r.validate().is_err(), "{bad}"),
                Err(_) => {}
            }
        }
        let id = crate::chat_session::checkpoint_id(&[1, 2]);
        let tail = parse(&format!(
            r#"{{"spec":"s","state":{{"input":"x"}},"context":{{"checkpoint":"{id}"}},"questions":[{{"field":"v"}}]}}"#
        ));
        assert!(tail.validate().is_ok());
        assert_eq!(tail.context, Some(OpinionContext { checkpoint: id }));
        assert!(
            serde_json::from_str::<OpinionRequest>(
                r#"{"spec":"s","state":{"input":"x"},"context":{"checkpoint":"x","other":1},"questions":[{"field":"v"}]}"#
            )
            .is_err(),
            "context takes no other fields"
        );
        let null_ctx = parse(
            r#"{"spec":"s","state":{"input":"x"},"context":null,"questions":[{"field":"v"}]}"#,
        );
        assert!(null_ctx.validate().is_ok());
    }
}
