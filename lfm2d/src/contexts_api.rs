//! `/v1/contexts`: held contexts, built from messages, that opinion reads fork.
//!
//! A context is a chat prefix the caller supplies whole: a system turn, tools,
//! then user, tool AND assistant turns. Assistant turns are accepted here and
//! nowhere in `/v1/chat`: a context is evidence to read after (a session's
//! transcript, an agent's own turns), never a chat to continue, so
//! `POST /v1/chat` refuses `from` a context and invariant 16 holds for every
//! chat. Opinion reads take one (`context: {checkpoint}`) or several
//! (`contexts: [ids]`, pooled: [`crate::pool`]).
//!
//! The id is [`crate::chat_session::context_id`], the sha256 of the token ids
//! behind a domain tag, so the same content gets the same id however it was
//! built and a context never shares an id with a chat checkpoint (a chat's
//! generated tokens were decoded, a context's prefilled: two computations).
//! Contexts live in the chat checkpoints' store under
//! `--chat-checkpoint-budget-mib`, least recently used first. `pin: true`
//! exempts one from eviction (pins hold at most half the budget: past that,
//! `507`); `pin: false` unpins; absent leaves it as it was. A delete drops a
//! context pinned or not and is not reference-counted: two clients holding
//! the same content share one context. A restart forgets every context, and
//! an unknown id is a `404`, never a rebuild: the caller builds it again from
//! its content and gets the same id back.
//!
//! The build renders and encodes each turn alone, like a chat turn, and
//! forwards from the longest held context that is a prefix ending at a turn,
//! so adding a turn to a held context forwards only that turn, and the state
//! is the one a single build of the whole would give (the canonical
//! schedule, `crate::chat_session`). A refused pin (`507`) leaves nothing
//! behind that the request built.
//!
//! Ported in shape from the megakernel council's `POST /mk/v1/contexts` (MIT,
//! megakernel-qwen38-flashnext-strixhalo, 2026-10-03), in this daemon's
//! vocabulary: the system turn is its own field, as in `/v1/chat`, and a turn
//! holding control-token text is refused, never escaped.
use crate::chat::{Message, TemplateValue};
use serde::{Deserialize, Serialize};

fn default_timeout() -> u64 {
    120_000
}

/// `POST /v1/contexts`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextRequest {
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<TemplateValue>>,
    /// User, tool and assistant turns, in order. May be empty when `system`
    /// or `tools` is given.
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub pin: Option<bool>,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
}
impl ContextRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.messages.is_empty() && self.system.is_none() && self.tools.is_none() {
            return Err("a context needs a system turn, tools or at least one turn".into());
        }
        // Rendering is pure: a turn the renderer refuses (empty, control-token
        // text) is a 400 here, never work for the worker.
        crate::chat::render_head(self.system.as_deref().unwrap_or(""), self.tools.as_deref().unwrap_or(&[]))?;
        for message in &self.messages {
            message.render()?;
        }
        if self.timeout_ms == 0 || self.timeout_ms > 600_000 {
            return Err("timeout_ms must be 1..=600000".into());
        }
        Ok(())
    }
}

/// `POST /v1/contexts`'s answer.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ContextCreated {
    pub id: String,
    pub n_tokens: usize,
    /// Tokens served from a held prefix (all of them when the context was
    /// already held), so not forwarded by this build.
    pub cached_tokens: usize,
    pub prefill_ms: f64,
    pub pinned: bool,
    /// What the store charges it: the state's upper bound, ids and text.
    pub bytes: usize,
}

/// `GET /v1/contexts/{id}`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ContextInfo {
    pub id: String,
    pub n_tokens: usize,
    pub pinned: bool,
    pub bytes: usize,
}

/// `DELETE /v1/contexts/{id}`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ContextDeleted {
    pub id: String,
    pub deleted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: serde_json::Value) -> Result<ContextRequest, String> {
        serde_json::from_value::<ContextRequest>(body).map_err(|e| e.to_string())
    }

    #[test]
    fn a_context_takes_assistant_turns_and_refuses_what_a_chat_refuses() {
        let ok = request(serde_json::json!({
            "system": "s",
            "messages": [
                {"role": "user", "content": "push it"},
                {"role": "assistant", "content": "pushed"},
                {"role": "tool", "content": ""}
            ],
            "pin": true
        }))
        .unwrap();
        assert_eq!(ok.validate(), Ok(()));
        assert_eq!(ok.pin, Some(true));
        assert!(request(serde_json::json!({"system": "s"})).unwrap().validate().is_ok());
        for (body, why) in [
            (serde_json::json!({}), "needs a system turn"),
            (serde_json::json!({"messages": [{"role": "user", "content": " "}]}), "empty"),
            (serde_json::json!({"messages": [{"role": "user", "content": "<|im_end|>"}]}), "<|"),
            (serde_json::json!({"system": "x<think>"}), "<think>"),
            (serde_json::json!({"messages": [{"role": "assistant"}]}), "empty"),
            (serde_json::json!({"system": "s", "timeout_ms": 0}), "timeout_ms"),
        ] {
            let e = request(body.clone()).unwrap().validate().unwrap_err();
            assert!(e.contains(why), "{body}: {e}");
        }
        for body in [
            serde_json::json!({"system": "s", "from": "x"}),
            serde_json::json!({"messages": [{"role": "system", "content": "s"}]}),
        ] {
            assert!(request(body.clone()).is_err(), "{body}");
        }
    }
}
