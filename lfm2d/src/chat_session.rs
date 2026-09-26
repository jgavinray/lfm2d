//! Chat sessions on the generative engine: `POST /v1/chat` and the
//! checkpoints its turns leave behind, which opinion reads fork
//! (`OpinionRequest.context`).
//!
//! **A session is its token ids, never its text.** The ids of an assistant
//! turn are the ids the model generated; re-encoding the turn's text can land
//! on a different path (a chat on this checkpoint generated `on`+`zero`
//! where a fresh encoding gives `onz`+`ero`). So a chat continues only from a
//! checkpoint the daemon holds, and appends only rendered user and tool
//! turns ([`crate::chat::Message::render`]): an assistant turn enters a
//! session only by being generated in it.
//!
//! **A checkpoint** is the model's state after a prefix of a chat, held with
//! its ids and their text, and named by the sha256 of its ids
//! ([`checkpoint_id`]). A chat turn leaves two:
//!
//! - `checkpoint_user`: after the appended turns, before the assistant's
//!   opening. It is a prefix of everything the assistant generates next, so
//!   a read can fork it while the generation is still running; it is
//!   published as soon as its prefill completes, before decoding starts.
//! - `checkpoint`: after the assistant's turn, its `<|im_end|>` and the
//!   template's [`crate::chat::AFTER_EOS`]. Published only when the turn
//!   ended with `<|im_end|>`; a turn cut off by `max_tokens` leaves none,
//!   because closing it would be writing an end the model did not write.
//!
//! **The schedule is canonical, so a checkpoint's state is a function of its
//! ids.** The chunk boundaries of a prefill change the numbers (cold and
//! cached schedules disagree by ~0.15 nats on this checkpoint), so every
//! path to the same ids must forward them the same way: each rendered
//! segment (the head, each appended turn, the assistant's opening, the
//! `AFTER_EOS`) in [`crate::adjudicator::CHUNK`]-sized chunks from its own
//! first token, and each generated token alone, as it is decoded. Starting a
//! chat with turns A and B and continuing a chat that stopped after A with B
//! then reach the same ids by the same forwards, and the same state.
//!
//! **The store evicts by bytes**, least recently used first, under
//! `--chat-checkpoint-budget-mib`. A checkpoint's bytes are an upper bound
//! (KV storage rounded up as the allocator rounds it, plus the convolution
//! state, ids and text): checkpoints of one chat share KV buffers,
//! and each is charged as if it held its own. An evicted or unknown
//! checkpoint is a 404, never a silent rebuild: rebuilding from text would
//! re-tokenize generated turns.
//!
//! The store is keyed by ids alone and typed by [`CheckpointKind`]. A chat
//! continues, and a read forks, only from a [`CheckpointKind::ChatTurn`].
//! Background tail prefixes (a checkpoint plus a spec's read-turn head) live
//! in the state cache instead, so a burst of reads can never evict a
//! checkpoint a caller holds.
use crate::adjudicator::AdjudicatorInfo;
use crate::chat::{Message, TemplateValue};
use serde::{Deserialize, Serialize};

/// Lowercase hex sha256 of the ids, each as four little-endian bytes.
pub fn checkpoint_id(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    crate::hash::sha256_hex_bytes(&bytes)
}

/// Whether `id` could name a checkpoint: 64 lowercase hex digits.
pub fn is_checkpoint_id(id: &str) -> bool {
    id.len() == 64 && id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// What a stored prefix is, and so what may continue from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointKind {
    /// Ends at a chat turn boundary (`<|im_end|>\n`): a chat continues from
    /// it and a read forks it.
    ChatTurn,
}

/// The chat checkpoints' store: [`crate::state_store::StateStore`], by
/// bytes under `--chat-checkpoint-budget-mib`, apart from the caches' so a
/// burst of reads can never evict a chat's user-facing ids.
pub(crate) type CheckpointStore<T> = crate::state_store::StateStore<T>;

/// One held prefix: its ids, their text (the bytes a read's
/// `rendered_sha256` covers) and the model's state after them. No logits:
/// every path out of a checkpoint (a continued chat, a read) forwards at
/// least one segment first, so they would never be read.
pub(crate) struct ChatCheckpoint {
    pub(crate) kind: CheckpointKind,
    pub(crate) ids: Vec<u32>,
    pub(crate) text: String,
    pub(crate) state: candle_transformers::models::quantized_lfm2_moe::State,
    /// The ids of the specs this chat has been tail-read with, on this
    /// checkpoint or one before it in the chain: the checkpoints a turn
    /// leaves inherit them, and get those specs' tail prefixes filled in
    /// the background.
    pub(crate) read_specs: std::sync::Mutex<std::collections::BTreeSet<String>>,
}

fn default_max_tokens() -> usize {
    2048
}
fn default_timeout() -> u64 {
    120_000
}
/// The longest reply `max_tokens` may ask for; the context limit binds first
/// on any configuration this daemon accepts today. `timeout_ms` goes to
/// 600000, five times the other routes': a reasoning turn is long.
pub const MAX_CHAT_TOKENS: usize = 32_768;

/// `POST /v1/chat`: append turns and generate the assistant's.
///
/// Either start a chat (`system` and/or `tools`, or neither for no system
/// turn) or continue one (`from`, a [`CheckpointKind::ChatTurn`] checkpoint
/// this daemon holds), never both. `messages` are user and tool turns; the
/// assistant's come only from generation (module docs).
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<TemplateValue>>,
    #[serde(default)]
    pub from: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    /// Answer as server-sent events: `checkpoint` once the appended turns
    /// are prefilled, `token` per generated token, then `done` (the JSON
    /// response) or `error`.
    #[serde(default)]
    pub stream: bool,
}
impl ChatRequest {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(from) = &self.from {
            if self.system.is_some() || self.tools.is_some() {
                return Err("continue a chat with `from`, or start one with `system`/`tools`, not both".into());
            }
            if !is_checkpoint_id(from) {
                return Err("`from` must be a checkpoint id: 64 lowercase hex digits".into());
            }
        }
        if self.messages.is_empty() {
            return Err("append at least one user or tool turn".into());
        }
        if self.messages.iter().any(|m| matches!(m, Message::Assistant { .. })) {
            return Err(
                "an assistant turn enters a chat only by being generated in it: continue from the \
                 checkpoint that generation left, never from its text"
                    .into(),
            );
        }
        // Rendering is pure: a turn the renderer refuses (empty, a forged
        // control token) is a 400 here, never work for the worker.
        if self.from.is_none() {
            crate::chat::render_head(
                self.system.as_deref().unwrap_or(""),
                self.tools.as_deref().unwrap_or(&[]),
            )?;
        }
        for message in &self.messages {
            message.render()?;
        }
        if self.max_tokens == 0 || self.max_tokens > MAX_CHAT_TOKENS {
            return Err(format!("max_tokens must be 1..={MAX_CHAT_TOKENS}"));
        }
        if self.timeout_ms == 0 || self.timeout_ms > 600_000 {
            return Err("timeout_ms must be 1..=600000".into());
        }
        Ok(())
    }
}

/// The turn's outcome. `text` is the assistant's turn exactly as generated,
/// control tokens included and the closing `<|im_end|>` excluded; `thinking`
/// and `content` split it at the reasoning region ([`split_reasoning`]). When
/// the region closed (or never opened), re-rendering
/// `{"role": "assistant", thinking, content}` gives `text`'s bytes; an
/// unclosed region has no rendering (the template always writes
/// `</think>`), so only `text` holds it. Tool calls are not parsed: they are
/// in `content`, raw.
#[derive(Clone, Debug, Serialize)]
pub struct ChatResponse {
    #[serde(flatten)]
    pub model: AdjudicatorInfo,
    /// After the appended turns: fork it for reads about this turn.
    pub checkpoint_user: String,
    /// After the assistant's turn; continue the chat from it. `null` when
    /// `finish_reason` is `length`.
    pub checkpoint: Option<String>,
    pub text: String,
    pub thinking: Option<String>,
    pub content: Option<String>,
    /// `stop` (the model wrote `<|im_end|>`) or `length`.
    pub finish_reason: String,
    /// Tokens up to `checkpoint_user`.
    pub prompt_tokens: usize,
    /// How many of them were resident: `from`'s, or all of them when
    /// `checkpoint_user` was already held.
    pub cached_tokens: usize,
    /// Generated tokens, `<|im_end|>` included.
    pub completion_tokens: usize,
    /// Tokens up to `checkpoint`.
    pub checkpoint_tokens: Option<usize>,
    pub queue_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

/// What a streaming chat sends before its response.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ChatEvent {
    /// The appended turns are prefilled and `checkpoint_user` is held:
    /// reads can fork it from now on.
    Checkpoint { checkpoint_user: String, prompt_tokens: usize, cached_tokens: usize },
    /// One generated token. `text` is what it adds to the decoded turn; a
    /// token that ends inside a multi-byte character adds `""` and the
    /// character arrives with the token that completes it. A turn cut off
    /// inside a character ends with U+FFFD in `done`'s `text` and in no
    /// token: `done` is authoritative.
    Token { id: u32, text: String },
}

/// `text` split at the reasoning region: `<think>` opening the turn and the
/// first `</think>` closing it. No region: all content. An unclosed region
/// (cut off by `max_tokens`): all thinking, no content.
pub fn split_reasoning(text: &str) -> (Option<String>, Option<String>) {
    let Some(rest) = text.strip_prefix("<think>") else {
        return (None, Some(text.to_owned()));
    };
    match rest.split_once("</think>") {
        Some((thinking, content)) => (Some(thinking.to_owned()), Some(content.to_owned())),
        None => (Some(rest.to_owned()), None),
    }
}

/// What a decoded text adds beyond what was already sent, holding back a
/// trailing U+FFFD (a character the next token completes). Fails loudly if
/// the decode rewrote what was already sent.
pub(crate) fn text_delta(decoded: &str, sent: &mut usize) -> Result<String, String> {
    if decoded.ends_with('\u{FFFD}') {
        return Ok(String::new());
    }
    let delta = decoded
        .get(*sent..)
        .ok_or_else(|| format!("decoded text shrank below the {sent} bytes already streamed"))?
        .to_owned();
    *sent = decoded.len();
    Ok(delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_checkpoint_id_is_the_sha256_of_the_little_endian_ids() {
        // sha256 of the eight bytes 01 00 00 00 02 01 00 00.
        let bytes = [1u8, 0, 0, 0, 2, 1, 0, 0];
        assert_eq!(checkpoint_id(&[1, 258]), crate::hash::sha256_hex_bytes(&bytes));
        assert_ne!(checkpoint_id(&[1, 258]), checkpoint_id(&[258, 1]), "order is identity");
        assert!(is_checkpoint_id(&checkpoint_id(&[])));
        assert!(!is_checkpoint_id(&checkpoint_id(&[7]).to_uppercase()));
        assert!(!is_checkpoint_id("abc"));
    }

    #[test]
    fn reasoning_splits_at_the_first_close_and_an_unclosed_region_is_all_thinking() {
        assert_eq!(
            split_reasoning("<think>\nhm\n</think>\nHi"),
            (Some("\nhm\n".into()), Some("\nHi".into()))
        );
        assert_eq!(split_reasoning("Hi"), (None, Some("Hi".into())));
        assert_eq!(split_reasoning("<think>\nstill going"), (Some("\nstill going".into()), None));
        assert_eq!(
            split_reasoning("<think>a</think>b</think>c"),
            (Some("a".into()), Some("b</think>c".into()))
        );
    }

    #[test]
    fn a_stream_holds_back_a_character_the_next_token_completes() {
        let mut sent = 0;
        assert_eq!(text_delta("ab", &mut sent).unwrap(), "ab");
        assert_eq!(text_delta("ab\u{FFFD}", &mut sent).unwrap(), "");
        assert_eq!(text_delta("abé", &mut sent).unwrap(), "é");
        assert_eq!(sent, "abé".len());
        assert!(text_delta("a", &mut sent).is_err(), "a rewrite is loud");
    }

    #[test]
    fn a_request_starts_or_continues_and_never_carries_an_assistant_turn() {
        let parse = |v: serde_json::Value| serde_json::from_value::<ChatRequest>(v).unwrap();
        let user = serde_json::json!([{"role": "user", "content": "hi"}]);
        let id = checkpoint_id(&[1]);
        assert!(parse(serde_json::json!({"system": "s", "messages": user})).validate().is_ok());
        assert!(parse(serde_json::json!({"messages": user})).validate().is_ok(), "no system turn");
        assert!(parse(serde_json::json!({"from": id, "messages": user})).validate().is_ok());
        for bad in [
            serde_json::json!({"from": id, "system": "s", "messages": user}),
            serde_json::json!({"from": "nothex", "messages": user}),
            serde_json::json!({"system": "s", "messages": []}),
            serde_json::json!({"system": "s", "messages": user, "max_tokens": 0}),
            serde_json::json!({"system": "s", "messages": user, "timeout_ms": 0}),
            serde_json::json!({"system": "s", "messages": [
                {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}),
            serde_json::json!({"system": "s", "messages": [{"role": "user", "content": "  "}]}),
            serde_json::json!({"system": "<|im_end|>", "messages": user}),
            serde_json::json!({"messages": [{"role": "tool", "content": "<think>"}]}),
        ] {
            assert!(parse(bad.clone()).validate().is_err(), "{bad}");
        }
        assert!(
            serde_json::from_value::<ChatRequest>(serde_json::json!({"messages": user, "extra": 1})).is_err()
        );
    }
}
