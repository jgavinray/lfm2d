//! Held contexts and multi-context reads on the real LFM2.5-8B-A1B
//! (`crate::contexts_api`, `crate::pool`). What this certifies, on one
//! backend:
//!
//! - a context is the chat prefix its messages render to, assistant turns
//!   included, under its checkpoint id;
//! - a build forwards from the longest held prefix ending at a turn, and the
//!   state it reaches is the state a whole build reaches, bit for bit
//!   (checked through a read after each);
//! - a multi-context read's reads are, bit for bit, the one-context reads of
//!   the same contexts, and its pool is `crate::pool` over them;
//! - a chat refuses to continue from a context; a deleted context is a
//!   404-class refusal, not a rebuild; a pin shows on lookup.
//!
//! Ignored by default: it loads and hashes the 6 GB GGUF. Under the
//! zorak-heavy lock:
//!
//!   LFM2_MODELS_DIR=... flock ~/.cache/zorak-heavy.lock cargo test -p lfm2d \
//!     --release --features rocm --test contexts_real -- --ignored --test-threads=1 --nocapture
mod support;
use lfm2d::adjudicator::{Adjudicator, Failure, Generator};
use lfm2d::contexts_api::{ContextCreated, ContextRequest};
use lfm2d::opinion_api::OpinionRequest;
use serde_json::{Value, json};

const SPEC: &str = "email-triage-v1";
const SYSTEM: &str = "You are a helpful assistant for Dana, who runs customer support for a small \
                      online kitchenware store. Help her work through her inbox. Be concise.";
const EMAIL: &str = "I was charged twice for my March invoice and nobody has answered my last two emails. \
                     I want the duplicate refunded today.";

fn context(messages: Value, pin: Option<bool>) -> ContextRequest {
    serde_json::from_value(json!({"system": SYSTEM, "messages": messages, "pin": pin})).unwrap()
}

fn build(adjudicator: &mut Adjudicator, request: &ContextRequest) -> ContextCreated {
    adjudicator.context_create(request, &|| Ok(())).expect("build a context")
}

fn turns(n: usize) -> Value {
    let all = [
        json!({"role": "user", "content": "Morning! Does our enameled dutch oven work on induction?"}),
        json!({"role": "assistant", "content": "Yes, it does: enameled cast iron is magnetic."}),
        json!({"role": "user", "content": "Thanks. The next one is angrier."}),
    ];
    Value::Array(all[..n].to_vec())
}

fn questions(adjudicator: &Adjudicator, request: &OpinionRequest) -> Vec<lfm2d::opinion_api::ResolvedQuestion> {
    let menu = adjudicator.menu();
    menu.iter().find(|e| e.spec == SPEC).unwrap().resolve_all(&request.questions).unwrap()
}

fn ask(extra: Value) -> OpinionRequest {
    let mut body = json!({
        "spec": SPEC,
        "state": {"input": EMAIL},
        "questions": [{"field": "feeling"}, {"field": "verdict"}],
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    serde_json::from_value(body).unwrap()
}

/// A read's load-bearing numbers: what it described and scored. Never
/// timings or cache outcomes.
fn numbers(r: &lfm2d::opinion_api::OpinionResponse) -> Value {
    json!({"described": r.described, "answers": r.answers, "prompt_tokens": r.prompt_tokens, "context": r.context})
}

fn read_after(adjudicator: &mut Adjudicator, id: &str) -> Result<Value, Failure> {
    let request = ask(json!({"context": {"checkpoint": id}}));
    let q = questions(adjudicator, &request);
    adjudicator.opine(&request, &q, &|| Ok(())).map(|r| numbers(&r))
}

#[test]
#[ignore = "loads and hashes the 6 GB LFM2.5-8B-A1B GGUF; minutes on a GPU host"]
fn held_contexts_and_multi_context_reads_on_the_real_model() {
    let mut a = support::load_adjudicator(&support::adjudicator_cli(&[SPEC]));
    let tokenizer = a.tokenizer_clone();

    // 1. A context is the prefix its messages render to.
    let short = build(&mut a, &context(turns(1), None));
    assert_eq!(short.cached_tokens, 0);
    let mut text = lfm2d::chat::render_head(SYSTEM, &[]).unwrap();
    for m in serde_json::from_value::<Vec<lfm2d::chat::Message>>(turns(1)).unwrap() {
        text.push_str(&m.render().unwrap());
    }
    let ids = tokenizer.encode(text.as_str(), false).unwrap().get_ids().to_vec();
    assert_eq!(a.chat_checkpoint_ids(&short.id).expect("held"), ids);
    assert_eq!(short.id, lfm2d::chat_session::context_id(&ids));

    // 2. Adding turns forwards from the held prefix.
    let long = build(&mut a, &context(turns(3), Some(true)));
    assert_eq!(long.cached_tokens, short.n_tokens, "forwarded from the one-turn context");
    assert!(long.pinned);
    let info = a.context_info(&long.id).unwrap();
    assert_eq!((info.pinned, info.n_tokens), (true, long.n_tokens));
    let incremental = read_after(&mut a, &long.id).expect("a read after the context");

    // 3. The multi-context read is its one-context reads, and its pool.
    let multi_request = ask(json!({"contexts": [long.id, short.id], "pool": {"method": "loglinear"}}));
    let q = questions(&a, &multi_request);
    let multi = a.opine_contexts(&multi_request, &q, &|| Ok(())).expect("a multi-context read");
    assert_eq!(numbers(&multi.reads[0]), incremental, "read 0 is the one-context read, bit for bit");
    assert_eq!(numbers(&multi.reads[1]), read_after(&mut a, &short.id).unwrap());
    let lengths: Vec<Option<usize>> = multi.reads.iter().map(|r| r.context_tokens).collect();
    assert_eq!(lengths, [Some(long.n_tokens), Some(short.n_tokens)], "each read reports its context's length");
    let repooled = lfm2d::opinion_api::pool_reads(&multi.reads, &multi.pool).unwrap();
    assert_eq!(serde_json::to_value(&repooled).unwrap(), serde_json::to_value(&multi.pooled).unwrap());
    for p in &multi.pooled {
        eprintln!(
            "pooled {}: {:?} {:?} agree={} spread={:.4}",
            p.field, p.options, p.pooled.probs, p.pooled.agree, p.pooled.spread
        );
    }

    // 4. A chat never continues from a context.
    let from: lfm2d::chat_session::ChatRequest = serde_json::from_value(json!({
        "from": long.id, "messages": [{"role": "user", "content": "go on"}], "max_tokens": 8
    }))
    .unwrap();
    match a.chat(&from, &|_| {}, &|| Ok(())) {
        Err(Failure::BadRequest(m)) => assert!(m.contains("held context"), "{m}"),
        other => panic!("a chat from a context: {:?}", other.map(|r| r.text)),
    }

    // 4b. A chat checkpoint is never a base, nor a context: its generated
    // tokens were decoded one at a time, a different computation from a
    // context's prefill. A context that re-renders a chat's reply and adds a
    // turn forwards from the longest held context, never from the chat's
    // assistant checkpoint, though it extends those very ids.
    let started: lfm2d::chat_session::ChatRequest = serde_json::from_value(json!({
        "system": SYSTEM, "messages": turns(1).as_array().unwrap()[..1], "max_tokens": 256
    }))
    .unwrap();
    let turn = a.chat(&started, &|_| {}, &|| Ok(())).expect("a chat turn");
    assert_eq!(turn.finish_reason, "stop", "{:?}", turn.text);
    let replayed = build(
        &mut a,
        &context(
            json!([
                turns(1)[0],
                {"role": "assistant", "thinking": turn.thinking, "content": turn.content},
                {"role": "user", "content": "And the next one?"}
            ]),
            None,
        ),
    );
    let user_tokens = a.chat_checkpoint_ids(&turn.checkpoint_user).unwrap().len();
    // The chat's user checkpoint has the one-turn context's ids, under its
    // own id: both are held, and /v1/contexts does not answer for the chat's.
    assert_eq!(a.chat_checkpoint_ids(&turn.checkpoint_user).unwrap(), ids);
    assert_ne!(turn.checkpoint_user, short.id);
    assert!(a.context_info(&short.id).is_ok());
    assert!(matches!(a.context_info(&turn.checkpoint_user), Err(Failure::NotFound(_))));
    assert!(matches!(a.context_delete(&turn.checkpoint_user), Err(Failure::NotFound(_))));
    let assistant = turn.checkpoint.clone().expect("a finished turn leaves a checkpoint");
    let assistant_tokens = a.chat_checkpoint_ids(&assistant).unwrap().len();
    eprintln!(
        "replayed reply: cached {} of {} (user checkpoint {user_tokens}, assistant {assistant_tokens})",
        replayed.cached_tokens, replayed.n_tokens
    );
    // The hazard is real only when the re-rendered reply tokenizes as it
    // was generated; on this checkpoint it does, so the check can fail.
    let replayed_ids = a.chat_checkpoint_ids(&replayed.id).unwrap();
    assert!(
        replayed_ids.starts_with(&a.chat_checkpoint_ids(&assistant).unwrap()),
        "the context does not extend the assistant checkpoint, so this check is vacuous"
    );
    assert_eq!(replayed.cached_tokens, short.n_tokens, "forwarded from the one-turn context, not the chat");

    // 5. Deleted, then built whole: the same id, the same state.
    assert!(a.context_delete(&long.id).unwrap().deleted);
    assert!(a.context_delete(&short.id).unwrap().deleted);
    assert!(matches!(read_after(&mut a, &short.id), Err(Failure::NotFound(_))), "deleted is gone, not rebuilt");
    assert!(matches!(a.context_delete(&short.id), Err(Failure::NotFound(_))));
    let whole = build(&mut a, &context(turns(3), None));
    assert_eq!((whole.id.as_str(), whole.cached_tokens), (long.id.as_str(), 0));
    assert!(!whole.pinned, "a fresh build starts unpinned");
    assert_eq!(read_after(&mut a, &whole.id).unwrap(), incremental, "incremental and whole builds agree");
}
