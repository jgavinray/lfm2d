//! Interleaving on the real LFM2.5-8B-A1B: a generative adjudication with
//! opinion reads, F8 reads and probes served at its pauses produces the
//! same tokens, the same logprobs and the same report as it does alone,
//! and every job served at a pause produces the numbers it produces alone.
//! On one backend the engine is deterministic, so "the same" is bit for bit
//! (serialized `f32`s compared as text), not a tolerance.
//!
//! The schedule is driven two ways: a hand-written [`YieldPoint`] that runs
//! chosen jobs at chosen pauses (prefill chunks and decode tokens, by
//! count, so the test is exact and does not depend on timing), and the
//! production worker through a `Handle`, where a read submitted during a
//! generation must come back before it.
//!
//! Ignored by default: each test loads and hashes the 6 GB GGUF (the first
//! test twice, so that the alone and the interleaved runs start from empty
//! caches). `tests/support` names the GPU (`LFM2D_TEST_GPU`, default rocm;
//! never cpu or auto) and arms the host-memory guard. Run them one at a time:
//!
//!   cargo test -p lfm2d --release --features rocm --test interleave_real \
//!     -- --ignored --test-threads=1 --nocapture
mod support;
use lfm2d::adjudicator::{AdjudicateRequest, Adjudicator, Failure, Generator, YieldPoint};
use lfm2d::config::Cli;
use lfm2d::opinion_api::{OpinionRequest, OpinionState, ResolvedQuestion};
use lfm2d::probe_api::ProbeRequest;
use serde_json::{Value, json};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

/// Describes `gist` and `feeling` before the `verdict` slot: the opinion
/// reads and the generation run here.
const FIELDS_FIRST: &str = "email-triage-v1";
/// Carries an `opinion` block: the F8 read (`/v1/adjudicate`, `opinion:
/// true`) runs here, on a second spec's caches.
const VERDICT_ONLY: &str = "email-verdict-opinion-v1";

fn cli() -> Cli {
    support::adjudicator_cli(&[FIELDS_FIRST, VERDICT_ONLY])
}

/// One line (the spec reads the email as one line), long enough that its
/// prefill takes several chunks, so reads land between prefill chunks as
/// well as between decoded tokens.
fn long_email(topic: &str) -> String {
    let sentences = [
        "I have been a customer for six years and I have never written in before",
        "last month I ordered a standing desk, two monitor arms and a replacement power supply",
        "the desk arrived with a cracked leg and the courier left it in the rain by the side gate",
        "the monitor arms arrived a week later in a box that had clearly been opened and taped shut again",
        "one arm is missing the clamp and the other has a stripped screw that will not tighten",
        "the power supply never arrived at all even though the tracking page says it was delivered",
        "I called on the fourth and was told someone would call me back within two business days",
        "nobody called, so I called again on the ninth and was put on hold for forty minutes",
        "the second agent said the first call had not been logged and asked me to start from the beginning",
        "I sent photographs of the cracked leg and the opened box by email the same afternoon",
        "I received an automatic reply saying my message had been received and nothing since",
        "I work from home and have been using a kitchen chair and an ironing board as a desk for three weeks",
    ];
    let mut out = format!("About {topic}: ");
    for round in 0..3 {
        for s in sentences {
            out.push_str(s);
            out.push_str(if round == 2 { ". " } else { "; " });
        }
    }
    out.push_str("Please tell me what happens next.");
    out
}

/// A job served at a pause, and its load-bearing numbers: what it answered,
/// never how long it took or what the caches did (a hit and a miss must
/// answer the same, but report differently).
enum Act {
    Read(OpinionRequest, Vec<ResolvedQuestion>),
    F8(AdjudicateRequest),
    Probe(ProbeRequest),
}

fn run(adjudicator: &mut Adjudicator, act: &Act) -> Value {
    let ok = || Ok(());
    match act {
        Act::Read(request, questions) => {
            let r = adjudicator.opine(request, questions, &ok).expect("read");
            json!({
                "described": r.described,
                "answers": r.answers,
                "prompt_tokens": r.prompt_tokens,
                "described_tokens": r.described_tokens,
            })
        }
        Act::F8(request) => {
            let r = adjudicator.generate(request, &ok).expect("F8 read");
            json!({"opinion": r.opinion, "prompt_tokens": r.prompt_tokens})
        }
        Act::Probe(request) => {
            let r = adjudicator.probe(request, &ok).expect("probe");
            json!({
                "rendered_sha256": r.rendered_sha256,
                "top_logprobs": r.top_logprobs,
                "continuations": r.continuations,
                "generated": r.generated,
            })
        }
    }
}

fn generation_numbers(r: &lfm2d::adjudicator::AdjudicateResponse) -> Value {
    json!({
        "output": r.output,
        "report": r.report,
        "finish_reason": r.finish_reason,
        "completion_tokens": r.completion_tokens,
        "prompt_tokens": r.prompt_tokens,
        "distributions": r.distributions,
    })
}

/// A yield point that serves `plan[pause]` (an index into `acts`) at the
/// numbered pause, and optionally cancels the paused job at one.
struct Interleaver<'a> {
    acts: &'a [Act],
    plan: BTreeMap<usize, usize>,
    cancel_at: Option<usize>,
    pauses: Cell<usize>,
    served: RefCell<Vec<(usize, Value)>>,
}
impl<'a> Interleaver<'a> {
    fn new(acts: &'a [Act], plan: &[(usize, usize)], cancel_at: Option<usize>) -> Self {
        Self {
            acts,
            plan: plan.iter().copied().collect(),
            cancel_at,
            pauses: Cell::new(0),
            served: RefCell::new(Vec::new()),
        }
    }
}
impl YieldPoint<Adjudicator> for Interleaver<'_> {
    fn check(&self) -> Result<(), Failure> {
        Ok(())
    }
    fn pause(&self, adjudicator: &mut Adjudicator) -> Result<(), Failure> {
        let n = self.pauses.get();
        self.pauses.set(n + 1);
        if self.cancel_at == Some(n) {
            return Err(Failure::Cancelled);
        }
        if let Some(&i) = self.plan.get(&n) {
            let numbers = run(adjudicator, &self.acts[i]);
            self.served.borrow_mut().push((i, numbers));
        }
        Ok(())
    }
}

fn acts(adjudicator: &Adjudicator) -> Vec<Act> {
    let menu = adjudicator.menu();
    let entry = menu.iter().find(|e| e.spec == FIELDS_FIRST).expect("fields-first spec loaded");
    let read = |input: &str, use_cache: bool| {
        let request: OpinionRequest = serde_json::from_value(json!({
            "spec": FIELDS_FIRST,
            "state": {"input": input},
            "questions": [{"field": "feeling"}, {"field": "verdict"}],
            "use_cache": use_cache,
        }))
        .unwrap();
        let questions = entry.resolve_all(&request.questions).unwrap();
        Act::Read(request, questions)
    };
    vec![
        // Cold: no cache read or written, a full prefill and description.
        read("I was charged twice for order #4471 and I want a refund today.", false),
        // Warm miss: prefills from the spec's prefix, then PUBLISHES its
        // own ready entry and described entries while the generation that
        // it interrupted holds a prefill of the same spec in hand.
        read("Hi, what are your store hours on Saturday?", true),
        Act::Probe(
            serde_json::from_value(json!({
                "text": "Someone logged into my account from another country.",
                "top_k": 5,
                "generate": 4,
                "continuations": [" Yes", " No"],
            }))
            .unwrap(),
        ),
        // Another spec's caches, through the generate entry point.
        Act::F8(
            serde_json::from_value(json!({
                "spec": VERDICT_ONLY,
                "input": "Email: Where is my parcel? The tracking page has not changed in a week.",
                "max_tokens": 512,
                "opinion": true,
            }))
            .unwrap(),
        ),
        // The same bytes as the warm miss: a described-cache hit now.
        read("Hi, what are your store hours on Saturday?", true),
    ]
}

fn generation(input: &str, distributions: bool) -> AdjudicateRequest {
    let mut v = json!({"spec": FIELDS_FIRST, "input": input, "max_tokens": 256});
    if distributions {
        v["distributions"] = json!({"top_k": 5});
    }
    serde_json::from_value(v).unwrap()
}

fn state(email: &str) -> String {
    OpinionState { input: email.into(), facts: None }.render("Email")
}

#[test]
#[ignore = "loads and hashes the 6 GB LFM2.5-8B-A1B GGUF twice; minutes on a GPU host"]
fn generation_and_reads_are_bit_identical_with_and_without_interleaving() {
    let ok = || Ok(());
    let long = state(&long_email("the desk order"));
    let traced = generation(&long, true);
    let plain = generation(&long, false);

    // Alone, from empty caches: each generation, then each act in order.
    let (alone_traced, alone_plain, alone_acts, prefill_chunks, prefix_tokens) = {
        let mut adjudicator = support::load_adjudicator(&cli());
        let acts = acts(&adjudicator);
        let t = adjudicator.generate(&traced, &ok).expect("generate alone");
        let p = adjudicator.generate(&plain, &ok).expect("generate alone, plain");
        assert_eq!(p.cached_tokens, p.prompt_tokens, "the plain run resumes the traced run's prefill");
        let chunks = (t.prompt_tokens - t.cached_tokens).div_ceil(128);
        assert!(chunks >= 3, "the input spans several prefill chunks: {chunks}");
        let order = [0, 1, 2, 3, 4];
        let numbers: Vec<Value> = order.iter().map(|&i| run(&mut adjudicator, &acts[i])).collect();
        (generation_numbers(&t), generation_numbers(&p), numbers, chunks, t.cached_tokens)
    };
    eprintln!("prefill chunks: {prefill_chunks}; prefix tokens: {prefix_tokens}");
    // What "does not move" compares is every step's token, logprob and top
    // five, and every read's options: never an empty list.
    let steps = alone_traced["distributions"].as_array().expect("traced").len();
    assert!(steps > 10 && steps as u64 == alone_traced["completion_tokens"].as_u64().unwrap());
    assert!(alone_acts.iter().all(|a| a != &Value::Null && a.as_object().is_some_and(|o| !o.is_empty())));

    // Interleaved, from empty caches: the same jobs in the same order, the
    // acts now served at pauses inside the traced generation (prefill and
    // decode) and inside the plain one (decode only, a ready hit).
    let mut adjudicator = support::load_adjudicator(&cli());
    let acts = acts(&adjudicator);
    let at = Interleaver::new(
        &acts,
        &[(0, 0), (1, 1), (2, 2), (prefill_chunks + 1, 3), (prefill_chunks + 20, 4)],
        None,
    );
    let traced_run = adjudicator.generate(&traced, &at).expect("generate interleaved");
    assert_eq!(at.served.borrow().len(), 5, "every act ran at its pause");
    assert_eq!(traced_run.cached_tokens, prefix_tokens, "a cold prefill, paused three times");
    assert_eq!(generation_numbers(&traced_run), alone_traced, "the generation does not move");
    for (i, numbers) in at.served.borrow().iter() {
        assert_eq!(numbers, &alone_acts[*i], "act {i} does not move when served at a pause");
    }
    // The paused prefill was published once complete, over the ready entry
    // the warm-miss read published at pause 1.
    let at = Interleaver::new(&acts, &[(0, 0), (3, 2), (7, 4)], None);
    let plain_run = adjudicator.generate(&plain, &at).expect("plain interleaved");
    assert_eq!(plain_run.cached_tokens, plain_run.prompt_tokens, "a ready hit: the paused prefill was published");
    assert_eq!(generation_numbers(&plain_run), alone_plain);
    for (i, numbers) in at.served.borrow().iter() {
        assert_eq!(numbers, &alone_acts[*i], "act {i}, at a decode pause of a resumed prefill");
    }

    // Cancelled at its second prefill pause, after a read ran at its first:
    // nothing of the cancelled prefill reaches the cache, so the same input
    // afterwards prefills from the spec's prefix again.
    let other = generation(&state(&long_email("the chair order")), false);
    let at = Interleaver::new(&acts, &[(0, 1)], Some(1));
    assert!(matches!(adjudicator.generate(&other, &at), Err(Failure::Cancelled)));
    let after = adjudicator.generate(&other, &ok).expect("the same input, uncancelled");
    assert_eq!(after.cached_tokens, prefix_tokens, "the cancelled prefill was never published");
}

/// Through the production worker: a read submitted while a real generation
/// runs comes back before the generation does, and the generation's
/// output is the one it gives alone.
#[tokio::test]
#[ignore = "loads and hashes the 6 GB LFM2.5-8B-A1B GGUF; minutes on a GPU host"]
async fn a_read_overtakes_a_real_generation_through_the_worker() {
    let adjudicator = support::load_adjudicator(&cli());
    let menu = adjudicator.menu();
    let info = adjudicator.info();
    let handle = lfm2d::adjudicator::Handle::spawn(adjudicator, info).with_menu(menu);
    let long = state(&long_email("the desk order"));
    let alone = handle.evaluate(generation(&long, false)).await.expect("alone");

    let generating = tokio::spawn({
        let handle = handle.clone();
        let request = generation(&long, false);
        async move {
            let r = handle.evaluate(request).await;
            (r, std::time::Instant::now())
        }
    });
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let request: OpinionRequest = serde_json::from_value(json!({
        "spec": FIELDS_FIRST,
        "state": {"input": "Hi, what are your store hours on Saturday?"},
        "questions": [{"field": "verdict"}],
    }))
    .unwrap();
    let read = handle.opine(request).await.expect("read");
    let read_done = std::time::Instant::now();
    let (generated, generated_done) = generating.await.unwrap();
    let generated = generated.expect("the paused generation finishes");
    eprintln!(
        "generation {:.0} ms decode for {} tokens; read {:.0} ms queued + {:.0} ms work, done {:.0} ms before the generation",
        generated.decode_ms,
        generated.completion_tokens,
        read.queue_ms,
        read.prefill_ms + read.describe_ms + read.read_ms,
        (generated_done - read_done).as_secs_f64() * 1000.
    );
    assert!(read_done < generated_done, "the read came back first");
    assert_eq!(generated.output, alone.output);
    assert_eq!(generated.completion_tokens, alone.completion_tokens);
}
