//! The adjudicator worker interleaves: an opinion read (or a probe) that
//! arrives while a generative adjudication runs is served at the
//! generation's next pause, and the generation then resumes. Over a stub
//! generator whose "generation" is a loop of pauses, so the schedule is the
//! whole subject. That the real engine's output does not move when reads
//! interleave is `interleave_real.rs`; that a cache is never published from
//! a half-finished job is `prompt_cache_tests` in `adjudicator.rs`.
use lfm2d::adjudicator::{
    AdjudicateRequest, AdjudicateResponse, Failure, Generator, Handle, PrefixInfo, YieldPoint,
};
use lfm2d::opinion::{OpinionRead, OptionScore};
use lfm2d::opinion_api::{
    Answer, CacheOutcome, FieldInfo, FieldKind, OpinionRequest, OpinionResponse, ResolvedQuestion,
    SpecMenuEntry,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

fn menu() -> Vec<SpecMenuEntry> {
    vec![SpecMenuEntry {
        id: "deadbeef".repeat(8),
        spec: "fixture".into(),
        input_label: "Input".into(),
        snapshot_id: "snapshot".into(),
        described_cache_capacity: 16,
        fields: vec![FieldInfo {
            field: "verdict".into(),
            kind: FieldKind::Choice,
            options: vec!["allow".into(), "ask".into()],
        }],
    }]
}

/// What the worker ran, in the order it ran it.
#[derive(Debug, Clone, PartialEq)]
enum Event {
    /// Generation `name` took step `i` (after its pause).
    Step(String, usize),
    /// Generation `name` finished all its steps.
    Done(String),
    Read(String),
    Probe,
    Register,
    /// Generation `name` stopped at a pause: cancelled or out of time.
    Stopped(String),
}

#[derive(Clone, Default)]
struct Log(Arc<Mutex<Vec<Event>>>);
impl Log {
    fn push(&self, e: Event) {
        self.0.lock().unwrap().push(e);
    }
    fn events(&self) -> Vec<Event> {
        self.0.lock().unwrap().clone()
    }
    fn position(&self, e: &Event) -> Option<usize> {
        self.events().iter().position(|x| x == e)
    }
    async fn wait_for(&self, e: Event) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while self.position(&e).is_none() {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("{e:?} never happened: {:?}", self.events()));
    }
}

/// A "generation" named by its input: `name:steps:ms` takes `steps` steps
/// of `ms` each, pausing before every one as the real engine pauses before
/// every chunk and token. `bg:name:steps:ms` instead queues that as the
/// stub's own background work (as a chat turn queues tail prefills) and
/// returns at once. A read named `slow` takes 400 ms, checking its own
/// cancellation as it goes.
struct Stub(Log, std::collections::VecDeque<String>);
impl Stub {
    /// `name:steps:ms`, pausing before every step; logs how it ended.
    fn steps(&mut self, script: &str, at: &dyn YieldPoint<Self>) -> Result<(), Failure> {
        let mut parts = script.split(':');
        let name = parts.next().unwrap().to_string();
        let steps: usize = parts.next().unwrap().parse().unwrap();
        let ms: u64 = parts.next().unwrap().parse().unwrap();
        for i in 0..steps {
            if let Err(e) = at.pause(self) {
                self.0.push(Event::Stopped(name));
                return Err(e);
            }
            self.0.push(Event::Step(name.clone(), i));
            std::thread::sleep(Duration::from_millis(ms));
        }
        at.check()?;
        self.0.push(Event::Done(name));
        Ok(())
    }
}
impl Generator for Stub {
    fn generate(
        &mut self,
        r: &AdjudicateRequest,
        at: &dyn YieldPoint<Self>,
    ) -> Result<AdjudicateResponse, Failure> {
        if r.opinion {
            at.check()?;
            self.0.push(Event::Read(format!("f8 {}", r.input)));
            return Ok(response(r));
        }
        if let Some(task) = r.input.strip_prefix("bg:") {
            self.1.push_back(task.to_string());
            return Ok(response(r));
        }
        self.steps(&r.input, at)?;
        Ok(response(r))
    }
    fn background(&mut self, at: &dyn YieldPoint<Self>) -> bool {
        let Some(task) = self.1.pop_front() else {
            return false;
        };
        let _ = self.steps(&task, at);
        true
    }
    fn opine(
        &mut self,
        request: &OpinionRequest,
        questions: &[ResolvedQuestion],
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<OpinionResponse, Failure> {
        check()?;
        if request.state.input.starts_with("slow") {
            let begin = Instant::now();
            while begin.elapsed() < Duration::from_millis(400) {
                check()?;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        self.0.push(Event::Read(request.state.input.clone()));
        Ok(OpinionResponse {
            prefix: info(),
            context_tokens: None,
            spec: request.spec.clone(),
            context: request.context.clone(),
            described: vec![],
            answers: questions
                .iter()
                .map(|q| Answer {
                    field: q.field.clone(),
                    read: OpinionRead {
                        options: q
                            .options
                            .iter()
                            .map(|o| OptionScore {
                                option: o.clone(),
                                logprob: -0.7,
                                first_logprob: -0.7,
                                prob: 0.5,
                                tokens: vec![1],
                            })
                            .collect(),
                        sequence_mass: 0.,
                        first_token_mass: 0.,
                        shared_tokens: 1,
                        scored_tokens: 2,
                        rendered_sha256: "0".repeat(64),
                    },
                    margin: 0.,
                })
                .collect(),
            rendered: None,
            rendered_token_ids: None,
            cache: CacheOutcome {
                prefix: "hit".into(),
                state: "miss".into(),
                described: "miss".into(),
            },
            prompt_tokens: 1,
            cached_tokens: 0,
            described_tokens: 0,
            queue_ms: 0.,
            prefill_ms: 0.,
            describe_ms: 0.,
            read_ms: 0.,
        })
    }
    fn register(
        &mut self,
        _: String,
        _: lfm2d::adjudicator::PromptSpec,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::adjudicator::RegisterOutcome, Failure> {
        check()?;
        self.0.push(Event::Register);
        Err(Failure::Unprocessable("the stub registers nothing".into()))
    }
    fn unregister(&mut self, _: &str) -> lfm2d::adjudicator::UnregisterOutcome {
        lfm2d::adjudicator::UnregisterOutcome::NotFound
    }
    fn probe(
        &mut self,
        _: &lfm2d::probe_api::ProbeRequest,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<lfm2d::probe_api::ProbeResponse, Failure> {
        check()?;
        self.0.push(Event::Probe);
        // A marker, not a response: this file is about when a probe runs.
        Err(Failure::Forbidden("probe served".into()))
    }
}

fn response(r: &AdjudicateRequest) -> AdjudicateResponse {
    AdjudicateResponse {
        prefix: info(),
        output: r.input.clone(),
        report: None,
        report_error: None,
        finish_reason: "stop".into(),
        prompt_tokens: 1,
        cached_tokens: 0,
        completion_tokens: 1,
        queue_ms: 0.,
        prefill_ms: 0.,
        decode_ms: 0.,
        distributions: None,
        opinion: None,
        resumed_tokens: None,
    }
}

fn spawn() -> (Handle, Log) {
    let log = Log::default();
    (Handle::spawn(Stub(log.clone(), Default::default()), (&info()).into()).with_menu(menu()), log)
}

fn generation(input: &str, timeout_ms: u64) -> AdjudicateRequest {
    serde_json::from_value(serde_json::json!({
        "spec": "fixture", "input": input, "timeout_ms": timeout_ms
    }))
    .unwrap()
}

fn read(input: &str) -> OpinionRequest {
    serde_json::from_value(serde_json::json!({
        "spec": "fixture", "state": {"input": input}, "questions": [{"field": "verdict"}]
    }))
    .unwrap()
}

fn steps_of(events: &[Event], name: &str) -> Vec<usize> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Step(n, i) if n == name => Some(*i),
            _ => None,
        })
        .collect()
}

/// The property: every kind of read submitted mid-generation is answered
/// within a step or two, while the generation is still running, and the
/// generation then finishes every one of its steps, once each, in order.
#[tokio::test]
async fn reads_and_probes_overtake_a_generation_at_its_next_pause_and_it_resumes() {
    let (h, log) = spawn();
    // 1000 steps of 2 ms: two seconds that a read must not wait out.
    let gen_task = tokio::spawn({
        let h = h.clone();
        async move { h.evaluate(generation("long:1000:2", 60_000)).await }
    });
    log.wait_for(Event::Step("long".into(), 5)).await;

    let begin = Instant::now();
    let answer = h.opine(read("mid-generation")).await.expect("the read is served");
    let read_ms = begin.elapsed().as_millis();
    assert_eq!(answer.answers.len(), 1);

    let f8: AdjudicateRequest = serde_json::from_value(serde_json::json!({
        "spec": "fixture", "input": "tail", "opinion": true
    }))
    .unwrap();
    let f8 = h.evaluate(f8).await.expect("the F8 read is served");
    assert_eq!(f8.output, "tail");

    let probe: lfm2d::probe_api::ProbeRequest =
        serde_json::from_value(serde_json::json!({"text": "x"})).unwrap();
    let probe = h.probe(probe).await.expect_err("the stub's probe answers with a marker");
    assert_eq!(probe.status(), 403, "the probe ran and returned its marker");
    let reads_ms = begin.elapsed().as_millis();

    assert!(
        !gen_task.is_finished(),
        "all three reads came back ({reads_ms} ms) while the generation still runs"
    );
    assert!(read_ms < 250, "an opinion read waited {read_ms} ms behind a generation");
    let generated = gen_task.await.unwrap().expect("the paused generation resumes and finishes");
    assert_eq!(generated.output, "long:1000:2");

    let events = log.events();
    assert_eq!(steps_of(&events, "long"), (0..1000).collect::<Vec<_>>(), "every step once, in order");
    let done = log.position(&Event::Done("long".into())).unwrap();
    for e in [Event::Read("mid-generation".into()), Event::Read("f8 tail".into()), Event::Probe] {
        let at = log.position(&e).unwrap_or_else(|| panic!("{e:?} never ran"));
        assert!(at < done, "{e:?} ran inside the generation, before it finished");
        assert!(
            !steps_of(&events[at..], "long").is_empty(),
            "the generation took steps after {e:?}: it paused, not ended"
        );
    }
}

/// Only a higher class overtakes. A second generation and a registration
/// submitted during a generation wait for it, in the order they arrived,
/// exactly as before; a read submitted after both still goes first.
#[tokio::test]
async fn generations_and_admin_keep_their_order_and_never_run_at_a_pause() {
    let (h, log) = spawn();
    let first = tokio::spawn({
        let h = h.clone();
        async move { h.evaluate(generation("first:400:2", 60_000)).await }
    });
    log.wait_for(Event::Step("first".into(), 3)).await;
    let second = tokio::spawn({
        let h = h.clone();
        async move { h.evaluate(generation("second:3:1", 60_000)).await }
    });
    let register = tokio::spawn({
        let h = h.clone();
        async move { h.register(br#"{"input_label":"Input","system":"Judge."}"#.to_vec()).await }
    });
    // Both are queued before the read is submitted.
    tokio::time::sleep(Duration::from_millis(20)).await;
    h.opine(read("late")).await.expect("the read is served");
    first.await.unwrap().expect("first");
    second.await.unwrap().expect("second");
    assert_eq!(register.await.unwrap().unwrap_err().status(), 422);

    let at = |e: Event| log.position(&e).unwrap_or_else(|| panic!("{e:?}: {:?}", log.events()));
    let first_done = at(Event::Done("first".into()));
    assert!(at(Event::Read("late".into())) < first_done, "the read overtook the running generation");
    assert!(first_done < at(Event::Step("second".into(), 0)), "generations never nest");
    assert!(at(Event::Done("second".into())) < at(Event::Register), "admin keeps its place in line");
}

/// Reads have a queue of their own: a generation backlog that fills the
/// generative queue (503 for the next generation) does not refuse a read.
#[tokio::test]
async fn a_full_generation_queue_does_not_refuse_a_read() {
    let (h, log) = spawn();
    let running = tokio::spawn({
        let h = h.clone();
        async move { h.evaluate(generation("running:2000:1", 60_000)).await }
    });
    log.wait_for(Event::Step("running".into(), 1)).await;
    let mut backlog = Vec::new();
    let mut overloaded = 0;
    for i in 0..12 {
        let h = h.clone();
        backlog.push(tokio::spawn(async move {
            h.evaluate(generation(&format!("queued{i}:1:0"), 60_000)).await
        }));
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    for task in &backlog {
        if task.is_finished() {
            overloaded += 1;
        }
    }
    assert!(overloaded >= 1, "the generative queue is bounded");
    h.opine(read("despite the backlog")).await.expect("a read is not refused by a generation backlog");
    assert!(
        log.position(&Event::Done("running".into())).is_none(),
        "and it was served mid-generation: {:?}",
        log.events().last()
    );
    running.abort();
    for task in backlog {
        task.abort();
    }
}

/// Time spent serving reads at a pause counts against the paused
/// generation's own deadline, which is still checked right after the pause
/// and before the next step: the read gets its answer, the generation its
/// 504, and no step runs past the deadline.
#[tokio::test]
async fn a_read_served_at_a_pause_spends_the_generations_deadline_not_its_own() {
    let (h, log) = spawn();
    let gen_task = tokio::spawn({
        let h = h.clone();
        async move { h.evaluate(generation("short:1000:1", 250)).await }
    });
    log.wait_for(Event::Step("short".into(), 2)).await;
    // The slow read takes 400 ms, past the generation's 250 ms deadline.
    h.opine(read("slow")).await.expect("the read keeps its own deadline");
    assert_eq!(gen_task.await.unwrap().unwrap_err().status(), 504);
    let events = log.events();
    let read_at = log.position(&Event::Read("slow".into())).unwrap();
    assert!(
        !events[read_at..].iter().any(|e| matches!(e, Event::Step(n, _) if n == "short")),
        "no generation step after the pause that overran its deadline: {:?}",
        &events[read_at..]
    );
}

/// The paused generation's own check runs between the jobs served at its
/// pause, not only once they run dry: out of time after the first of three
/// queued reads, it stops there, and the other two are served after it.
#[tokio::test]
async fn a_paused_generation_is_checked_between_the_reads_served_at_its_pause() {
    let (h, log) = spawn();
    let gen_task = tokio::spawn({
        let h = h.clone();
        async move { h.evaluate(generation("between:1000:1", 250)).await }
    });
    log.wait_for(Event::Step("between".into(), 2)).await;
    let reads: Vec<_> = ["slow 1", "slow 2", "slow 3"]
        .into_iter()
        .map(|input| {
            let h = h.clone();
            tokio::spawn(async move { h.opine(read(input)).await })
        })
        .collect();
    for task in reads {
        task.await.unwrap().expect("every read is served");
    }
    assert_eq!(gen_task.await.unwrap().unwrap_err().status(), 504);
    let at = |e: Event| log.position(&e).unwrap_or_else(|| panic!("{e:?}: {:?}", log.events()));
    let stopped = at(Event::Stopped("between".into()));
    let reads: Vec<usize> = ["slow 1", "slow 2", "slow 3"].map(|r| at(Event::Read(r.into()))).into();
    assert_eq!(
        reads.iter().filter(|&&r| r < stopped).count(),
        1,
        "stopped after the first read, not after all three: {:?}",
        log.events().iter().filter(|e| !matches!(e, Event::Step(..))).collect::<Vec<_>>()
    );
}

/// The stop signal reaches both the read being served at a pause and the
/// generation paused under it.
#[tokio::test]
async fn stop_cancels_the_read_at_a_pause_and_the_paused_generation() {
    let (h, log) = spawn();
    let gen_task = tokio::spawn({
        let h = h.clone();
        async move { h.evaluate(generation("stopped:1000:1", 60_000)).await }
    });
    log.wait_for(Event::Step("stopped".into(), 2)).await;
    let read_task = tokio::spawn({
        let h = h.clone();
        async move { h.opine(read("slow")).await }
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    h.stop_signal().store(true, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(read_task.await.unwrap().unwrap_err().status(), 408);
    assert_eq!(gen_task.await.unwrap().unwrap_err().status(), 408);
    assert!(log.position(&Event::Read("slow".into())).is_none(), "the cancelled read never finished");
}

/// Background work runs only when every queue is empty, reads overtake it
/// at its pauses, and nothing Generative (a registration here, which can
/// remove a spec by eviction) ever runs at them: it waits for the task.
#[tokio::test]
async fn background_work_runs_when_idle_reads_overtake_it_and_admin_waits_for_it() {
    let (h, log) = spawn();
    // Queued behind a generation: it must not start until the generation is done.
    let first = tokio::spawn({
        let h = h.clone();
        async move { h.evaluate(generation("gen:20:2", 60_000)).await }
    });
    log.wait_for(Event::Step("gen".into(), 0)).await;
    h.evaluate(generation("bg:bg:500:2", 60_000)).await.expect("queued");
    first.await.unwrap().expect("gen");
    log.wait_for(Event::Step("bg".into(), 3)).await;
    h.opine(read("during background")).await.expect("the read is served");
    let register = tokio::spawn({
        let h = h.clone();
        async move { h.register(br#"{"input_label":"Input","system":"Judge."}"#.to_vec()).await }
    });
    assert_eq!(register.await.unwrap().unwrap_err().status(), 422);
    let at = |e: Event| log.position(&e).unwrap_or_else(|| panic!("{e:?}: {:?}", log.events()));
    let bg_start = at(Event::Step("bg".into(), 0));
    let bg_done = at(Event::Done("bg".into()));
    assert!(at(Event::Done("gen".into())) < bg_start, "background waits for queued work");
    let read_at = at(Event::Read("during background".into()));
    assert!(bg_start < read_at && read_at < bg_done, "the read overtook the background task");
    assert!(bg_done < at(Event::Register), "admin never runs at a background pause");
}

/// Stopping the daemon ends a background task at its next pause.
#[tokio::test]
async fn stop_ends_a_background_task_at_its_next_pause() {
    let (h, log) = spawn();
    h.evaluate(generation("bg:bg:5000:1", 60_000)).await.expect("queued");
    log.wait_for(Event::Step("bg".into(), 2)).await;
    h.stop_signal().store(true, std::sync::atomic::Ordering::SeqCst);
    log.wait_for(Event::Stopped("bg".into())).await;
    assert!(log.position(&Event::Done("bg".into())).is_none());
}
