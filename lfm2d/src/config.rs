//! CLI configuration — flags with env-var fallbacks (`clap`'s `env`
//! feature). Startup fails loudly on anything incoherent; see
//! [`Cli::validate`].

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use lfm2_encoder::DType;

/// Compute dtype, as spelled on the command line. A closed enum rather than
/// a free string so `clap` rejects a typo at parse time with the valid set
/// named, instead of the daemon discovering it mid-load.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum DtypeArg {
    F32,
    F16,
    Bf16,
}

impl DtypeArg {
    pub fn to_dtype(self) -> DType {
        match self {
            DtypeArg::F32 => DType::F32,
            DtypeArg::F16 => DType::F16,
            DtypeArg::Bf16 => DType::BF16,
        }
    }
}

/// `lfm2d` — HTTP sidecar serving LFM2.5 encoder heads. Loads every
/// configured checkpoint once at startup (no lazy loading); serves the same
/// API over a Unix socket and/or TCP.
#[derive(Parser, Debug, Clone)]
#[command(name = "lfm2d", about, long_about = None)]
pub struct Cli {
    /// LFM2.5-8B-A1B GGUF for the separate adjudicator worker.
    #[arg(long, env="LFM2D_ADJUDICATOR_MODEL")]
    pub adjudicator_model: Option<PathBuf>,
    /// Matching Hugging Face tokenizer.json (checked against GGUF vocabulary).
    #[arg(long, env="LFM2D_ADJUDICATOR_TOKENIZER")]
    pub adjudicator_tokenizer: Option<PathBuf>,
    /// Adjudicator context budget, including output. The initial eager attention path is capped at 8192.
    #[arg(long, default_value_t=4096)]
    pub adjudicator_context: usize,
    /// Sign-aware repetition penalty over the full prompt and generated tokens.
    /// 1.0 disables it; 1.05 is the checkpoint author's recommendation.
    #[arg(long, default_value_t = 1.05)]
    pub adjudicator_repeat_penalty: f32,
    /// Prompt specs loaded at boot, each a JSON `{system, output_schema, ...}`
    /// prefilled into its own resident prefix. Repeatable, and optional: with
    /// none the menu starts empty and fills through `POST /v1/opinion/specs`.
    /// No spec is a default; `/v1/opinion` and `/v1/adjudicate` both name
    /// one. Named by file stem, so stems must not repeat.
    #[arg(long = "opinion-spec", env = "LFM2D_OPINION_SPECS", value_delimiter = ',')]
    pub opinion_specs: Vec<PathBuf>,
    /// How many runtime-uploaded specs (`POST /v1/opinion/specs`) stay
    /// resident at once. Each holds a resident prefix state, which is why
    /// this is bounded. Past it, the least recently used upload is evicted
    /// (a re-registration or a served request both count as use); its next
    /// request gets a 404 that tells the client to upload it again.
    /// Boot-time specs (`--opinion-spec`) are never
    /// evicted and don't count against this.
    #[arg(long = "opinion-spec-capacity", env = "LFM2D_OPINION_SPEC_CAPACITY", default_value_t = 8)]
    pub opinion_spec_capacity: usize,

    /// Directory holding an `Lfm2Embedding`-shaped checkpoint
    /// (`config.json`, `tokenizer.json`, `model.safetensors`). Backs
    /// `/embed`.
    #[arg(long, env = "LFM2D_EMBEDDER_DIR")]
    pub embedder_dir: Option<PathBuf>,

    /// Directory holding an `Lfm2SequenceRouter`-shaped checkpoint (the
    /// Prompt-Router). Backs `/v1/route`.
    #[arg(long, env = "LFM2D_ROUTER_DIR")]
    pub router_dir: Option<PathBuf>,

    /// Directory holding an `Lfm2TokenClassifier`-shaped checkpoint —
    /// REPEATABLE (pass `--token-classifier-dir` once per head, or a
    /// comma-separated list via `LFM2D_TOKEN_CLASSIFIER_DIR`), unlike
    /// `--embedder-dir`/`--router-dir` which each take
    /// at most one. Each registers under a model id derived from its
    /// directory basename, exactly like every other head — fully generic,
    /// no checkpoint-specific logic anywhere (the PII detector is not
    /// special-cased). Backs `POST /v1/spans` and
    /// `POST /v1/spans/credentials`. With 2+ loaded, a request must name
    /// which one via `"model"`; with exactly 1, it's implicit — see
    /// `engine_real.rs`'s `resolve_token_classifier`.
    #[arg(long = "token-classifier-dir", env = "LFM2D_TOKEN_CLASSIFIER_DIR", value_delimiter = ',')]
    pub token_classifier_dir: Vec<PathBuf>,

    /// Attach a hash of a `/v1/spans`/`/v1/spans/credentials` request's
    /// input TEXT (never the text itself) to that call's trace/log span,
    /// as an opt-in correlation aid — e.g. matching one detection call to
    /// the same call logged by an upstream service, without either system
    /// writing the caller's actual secret-bearing text anywhere. Defaults
    /// OFF and is deliberately `ArgAction::Set` (must be spelled
    /// `--log-input-hash true`, not just present) rather than a bare
    /// switch — flipping on ANY per-request hashing on a secrets-detection
    /// endpoint should be a considered operator choice, not a fat-fingered
    /// flag. NEVER attached as an OTLP metric label regardless of this
    /// setting — see `worker.rs`'s `spans`/`spans_credentials` and the
    /// module docs' "Observability" section for why (unbounded per-input
    /// cardinality would wreck VictoriaMetrics).
    #[arg(
        long = "log-input-hash",
        env = "LFM2D_LOG_INPUT_HASH",
        action = clap::ArgAction::Set,
        default_value_t = false
    )]
    pub log_input_hash: bool,

    /// Unix domain socket path to serve on. At least one of this or
    /// `--bind-addr` is required.
    #[arg(long, env = "LFM2D_SOCKET_PATH")]
    pub socket_path: Option<PathBuf>,

    /// TCP address to serve on, e.g. `127.0.0.1:8080`. At least one of this
    /// or `--socket-path` is required.
    #[arg(long, env = "LFM2D_BIND_ADDR")]
    pub bind_addr: Option<String>,

    /// Compute dtype every head is loaded and run at: `f32` (default),
    /// `f16`, or `bf16`.
    ///
    /// # Read this before reaching for f16
    ///
    /// **The LFM2.5 encoder checkpoints ship f32 natively** — verified
    /// 2026-08-12 from the safetensors headers: Encoder-350M,
    /// Prompt-Router and PII-Detector are all `F32`, ~1352 MiB each, with
    /// `torch_dtype: float32` in their configs. `kube_ordinal_v8` is f32
    /// too. So f16 is a genuine *loss* of shipped resolution here, not the
    /// removal of a pointless upcast.
    ///
    /// **The lone exception is `LFM2.5-Embedding-350M`, which ships
    /// `BF16`** (676 MiB). bf16→f32 is lossless; bf16→**f16 is not**, and
    /// not merely in mantissa — bf16 carries f32's 8 exponent bits against
    /// f16's 5, so values outside f16's range become inf/0 rather than
    /// rounding. Loading that checkpoint as f16 is the one combination here
    /// that can quietly produce wrong numbers instead of slow ones.
    ///
    /// **And f16 trades the constraint that binds for the one that
    /// doesn't.** Measured on this workload, f16 halves memory at ~1.6×
    /// latency, while the cost that actually hurts is a forward pass in an
    /// interactive path. Reach for this when a box is memory-bound (zorak's
    /// `system-reserved` arithmetic), not to make things faster.
    #[arg(long, env = "LFM2D_DTYPE", default_value = "f32")]
    pub dtype: DtypeArg,

    /// Execution backend. Auto tries compiled GPU backends (ROCm, CUDA,
    /// Metal), then CPU if device initialization is unavailable. Explicit
    /// backends fail instead of falling back. Model-load and inference errors
    /// are never retried on another device. The adjudicator refuses auto
    /// resolving to CPU (its CPU MoE is a reference); name `cpu` for that.
    #[arg(long, env = "LFM2D_DEVICE", default_value = "auto")]
    pub device: crate::device::DeviceArg,

    /// GPU ordinal within the selected backend.
    #[arg(long, env = "LFM2D_DEVICE_INDEX", default_value_t = 0)]
    pub device_index: usize,

    /// Size of the rayon global thread pool that candle's matmul runs on
    /// (`rayon::ThreadPoolBuilder::num_threads`), set BEFORE any model is
    /// loaded — rayon's global pool can only be built once, so this must
    /// win the race against candle's own lazy default. Defaults to
    /// `std::thread::available_parallelism()` when unset, which under a
    /// k8s CPU *limit* (a cgroup quota, not just a request) usually
    /// over-reports — see `lfm2d/deploy/k8s.yaml`'s CPU-limits comment for
    /// why this daemon's deploy guidance is "requests without limits."
    #[arg(long, env = "LFM2D_THREADS")]
    pub threads: Option<usize>,

    /// `POST /v1/probe` is on by default whenever the adjudicator is
    /// loaded — `--no-probe` (or `LFM2D_PROBE=0`/`false`) turns it off,
    /// which removes the ROUTE entirely rather than leaving it present and
    /// answering 403: an unauthenticated prober should not be able to tell
    /// "this feature exists but is disabled" from "this daemon never had
    /// it." It is an instrument with no calibration contract
    /// (`docs/lfm25-adjudicator.md` "Probe and tokenize"), which is exactly
    /// why it can be switched off without affecting `/v1/opinion` or
    /// `/v1/adjudicate` at all. No effect when the adjudicator itself is
    /// not configured — there is no route to add either way.
    #[arg(
        long = "no-probe",
        env = "LFM2D_PROBE",
        action = clap::ArgAction::SetFalse,
        value_parser = clap::builder::BoolishValueParser::new(),
        default_value_t = true,
    )]
    pub probe: bool,
}

/// Why a removed flag's env var is refused, shared by the flags one
/// removal retired together.
const NO_DEFAULT_SPEC: &str = "no spec is a default and the cascade is gone. Unset it; boot specs \
    are --opinion-spec, and /v1/opinion and /v1/adjudicate name a spec per request";
const NO_CLASSIFIER: &str = "the daemon no longer serves a sequence classifier (/predict, \
    /v1/classify). Unset it; the head remains a library type, \
    lfm2_encoder::Lfm2SequenceClassifier, for a consumer that embeds it";

/// Env vars whose flags were removed, each with why and what to do
/// instead. clap ignores an env var it has no argument for, so a
/// deployment that still sets one would start without the behaviour it
/// asked for; startup refuses instead. The flags themselves need no entry:
/// clap rejects an unknown argument on its own.
pub const RETIRED_ENV: &[(&str, &str)] = &[
    ("LFM2D_ADJUDICATOR_PROMPT", NO_DEFAULT_SPEC),
    ("LFM2D_CASCADE_ROUTES", NO_DEFAULT_SPEC),
    ("LFM2D_CASCADE_SEVERE_LABELS", NO_DEFAULT_SPEC),
    ("LFM2D_CLASSIFIER_DIR", NO_CLASSIFIER),
    ("LFM2D_CANDIDATE_CLASSIFIER_DIR", NO_CLASSIFIER),
];

/// Refuses a retired env var by name. `is_set` is the environment lookup,
/// injected so the check is testable without touching the process env.
pub fn refuse_retired_env(is_set: impl Fn(&str) -> bool) -> Result<(), String> {
    match RETIRED_ENV.iter().find(|(name, _)| is_set(name)) {
        Some((name, why)) => Err(format!("{name} is set, but its flag was removed (2026-09-24): {why}")),
        None => Ok(()),
    }
}

impl Cli {
    /// Cross-field checks `clap` itself can't express. Pure and
    /// unit-testable without ever invoking the CLI parser — see the tests
    /// below.
    ///
    /// # Errors
    /// - neither `--socket-path` nor `--bind-addr` given: nothing to serve
    ///   on.
    /// - none of `--embedder-dir`/`--router-dir`/`--token-classifier-dir`/
    ///   `--adjudicator-model` given: nothing to load, so this process
    ///   would serve `/healthz` and nothing else — almost certainly a
    ///   misconfiguration, not a deliberate deployment.
    pub fn validate(&self) -> Result<(), String> {
        if self.socket_path.is_none() && self.bind_addr.is_none() {
            return Err(
                "no transport configured: pass --socket-path and/or --bind-addr \
                 (env LFM2D_SOCKET_PATH / LFM2D_BIND_ADDR) — at least one is required"
                    .to_string(),
            );
        }
        if self.embedder_dir.is_none()
            && self.router_dir.is_none()
            && self.token_classifier_dir.is_empty()
            && self.adjudicator_model.is_none()
        {
            return Err(
                "no models configured: pass at least one of --embedder-dir/--router-dir/\
                 --token-classifier-dir/--adjudicator-model (env LFM2D_EMBEDDER_DIR/\
                 LFM2D_ROUTER_DIR/LFM2D_TOKEN_CLASSIFIER_DIR/LFM2D_ADJUDICATOR_MODEL)"
                    .to_string(),
            );
        }
        if !self.adjudicator_repeat_penalty.is_finite() || !(1.0..=2.0).contains(&self.adjudicator_repeat_penalty) {
            return Err("adjudicator repeat penalty must be finite and in 1.0..=2.0".into());
        }
        if self.adjudicator_model.is_some() != self.adjudicator_tokenizer.is_some() {
            return Err("--adjudicator-model and --adjudicator-tokenizer must be supplied together".into());
        }
        if self.adjudicator_model.is_some() && (!matches!(self.dtype, DtypeArg::F32) || !(128..=8192).contains(&self.adjudicator_context)) {
            return Err("adjudicator requires --dtype f32 and --adjudicator-context 128..=8192".into());
        }
        if !self.opinion_specs.is_empty() && self.adjudicator_model.is_none() {
            return Err("--opinion-spec needs the adjudicator (--adjudicator-model/--adjudicator-tokenizer)".into());
        }
        if self.opinion_spec_capacity == 0 {
            return Err("--opinion-spec-capacity must be at least 1".into());
        }
        Ok(())
    }

    /// Whether `POST /v1/probe` should be routed at all — the exact value
    /// `main.rs` passes as `adjudicator::router`'s `probe_enabled`
    /// argument (`router.merge(lfm2d::adjudicator::router(handle,
    /// cli.probe_route_enabled()))`). Factored out so the
    /// `--no-probe`/`LFM2D_PROBE`-flag-to-router-argument wiring is
    /// itself unit-testable via `Cli::parse_from` (real clap parsing,
    /// same as `Cli::parse` in production) without needing to load a
    /// model or bind a socket — `main.rs`'s own line is then a trivial,
    /// visibly-correct pass-through with no conditional logic of its own
    /// left untested (kaibo review, 2026-09-23). No effect when the
    /// adjudicator itself is not configured; there is no route to add
    /// either way.
    pub fn probe_route_enabled(&self) -> bool {
        self.probe
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Cli {
        Cli {
            adjudicator_model: None,
            adjudicator_tokenizer: None,
            adjudicator_context: 4096,
            adjudicator_repeat_penalty: 1.05,
            opinion_specs: Vec::new(),
            opinion_spec_capacity: 8,
            embedder_dir: None,
            router_dir: None,
            token_classifier_dir: Vec::new(),
            log_input_hash: false,
            socket_path: None,
            bind_addr: None,
            dtype: DtypeArg::F32,
            device: crate::device::DeviceArg::Cpu,
            device_index: 0,
            threads: None,
            probe: true,
        }
    }

    #[test]
    fn rejects_no_transport_configured() {
        let mut cli = base();
        cli.embedder_dir = Some("/tmp/x".into());
        let err = cli.validate().expect_err("no socket_path/bind_addr must be refused");
        assert!(err.contains("transport"), "{err}");
    }

    #[test]
    fn rejects_no_models_configured() {
        let mut cli = base();
        cli.bind_addr = Some("127.0.0.1:0".into());
        let err = cli.validate().expect_err("no models configured must be refused");
        assert!(err.contains("models"), "{err}");
    }

    #[test]
    fn accepts_one_model_and_socket_path_only() {
        let mut cli = base();
        cli.router_dir = Some("/tmp/router".into());
        cli.socket_path = Some("/tmp/lfm2d.sock".into());
        cli.validate().expect("one model + one transport is valid");
    }

    #[test]
    fn accepts_both_transports_and_all_three_models() {
        let cli = Cli {
            adjudicator_model: None,
            adjudicator_tokenizer: None,
            adjudicator_context: 4096,
            adjudicator_repeat_penalty: 1.05,
            opinion_specs: Vec::new(),
            opinion_spec_capacity: 8,
            embedder_dir: Some("/tmp/e".into()),
            router_dir: Some("/tmp/r".into()),
            token_classifier_dir: vec!["/tmp/t".into()],
            log_input_hash: true,
            socket_path: Some("/tmp/lfm2d.sock".into()),
            bind_addr: Some("0.0.0.0:8080".into()),
            dtype: DtypeArg::F32,
            device: crate::device::DeviceArg::Cpu,
            device_index: 0,
            threads: Some(4),
            probe: false,
        };
        cli.validate().expect("fully specified config is valid");
    }

    #[test]
    fn a_token_classifier_dir_alone_satisfies_the_models_requirement() {
        let mut cli = base();
        cli.token_classifier_dir = vec!["/tmp/pii".into()];
        cli.socket_path = Some("/tmp/lfm2d.sock".into());
        cli.validate().expect("--token-classifier-dir alone is a valid model set");
    }

    #[test]
    fn token_classifier_dir_is_repeatable() {
        let cli = Cli {
            token_classifier_dir: vec!["/tmp/pii".into(), "/tmp/secrets".into()],
            socket_path: Some("/tmp/lfm2d.sock".into()),
            ..base()
        };
        assert_eq!(cli.token_classifier_dir.len(), 2);
        cli.validate().expect("2+ token-classifier dirs is valid config");
    }

    #[test]
    fn log_input_hash_defaults_off() {
        assert!(!base().log_input_hash, "a secrets-detection endpoint must not hash input by default");
    }

    #[test]
    fn a_retired_env_var_is_refused_by_name() {
        for (name, _) in RETIRED_ENV {
            let err = refuse_retired_env(|k| k == *name).expect_err("a retired env var must stop startup");
            assert!(err.contains(name), "the refusal must name {name}: {err}");
        }
        refuse_retired_env(|k| k == "LFM2D_BIND_ADDR").expect("a live env var is not refused");
        refuse_retired_env(|_| false).expect("an empty environment is not refused");
    }

    /// The sequence-classifier head left the daemon (2026-09-24). A
    /// deployment still pointing at a classifier must stop at startup and
    /// be told where the head went, not boot serving nothing it expected.
    #[test]
    fn the_retired_classifier_env_vars_are_refused_with_directions() {
        for name in ["LFM2D_CLASSIFIER_DIR", "LFM2D_CANDIDATE_CLASSIFIER_DIR"] {
            let err = refuse_retired_env(|k| k == name).expect_err("a retired classifier env var must stop startup");
            assert!(err.contains(name), "the refusal must name {name}: {err}");
            assert!(err.contains("/v1/classify"), "the refusal must name the removed surface: {err}");
            assert!(err.contains("Lfm2SequenceClassifier"), "the refusal must say where the head lives now: {err}");
        }
    }

    #[test]
    fn the_retired_classifier_flags_are_refused_by_the_parser() {
        for flag in ["--classifier-dir", "--candidate-classifier-dir"] {
            let err = Cli::try_parse_from(["lfm2d", "--bind-addr", "127.0.0.1:0", flag, "/tmp/c"])
                .expect_err("a retired classifier flag must not parse");
            assert!(err.to_string().contains(flag), "the parse error must name {flag}: {err}");
        }
    }

    // ------------------------------------------------------- adjudicator
    //
    // The model and tokenizer enable the adjudicator; specs are optional at
    // boot (the menu can start empty and fill through
    // `POST /v1/opinion/specs`). There is no `--adjudicator-prompt`: no spec
    // is privileged, so none is required.

    fn adjudicator_only() -> Cli {
        Cli {
            adjudicator_model: Some("/models/lfm25.gguf".into()),
            adjudicator_tokenizer: Some("/models/tokenizer.json".into()),
            socket_path: Some("/tmp/lfm2d.sock".into()),
            ..base()
        }
    }

    #[test]
    fn the_adjudicator_boots_with_zero_specs() {
        adjudicator_only()
            .validate()
            .expect("model + tokenizer with no --opinion-spec is a valid, empty-menu adjudicator");
    }

    #[test]
    fn the_adjudicator_needs_both_model_and_tokenizer() {
        for cli in [
            Cli { adjudicator_tokenizer: None, ..adjudicator_only() },
            // A router too, so the refusal is about the adjudicator pair and
            // not "no models configured".
            Cli { adjudicator_model: None, router_dir: Some("/tmp/router".into()), ..adjudicator_only() },
        ] {
            let err = cli.validate().expect_err("half an adjudicator must be refused");
            assert!(err.contains("--adjudicator-model") && err.contains("--adjudicator-tokenizer"), "{err}");
        }
    }

    #[test]
    fn an_opinion_spec_without_the_adjudicator_is_refused() {
        let cli = Cli {
            router_dir: Some("/tmp/router".into()),
            socket_path: Some("/tmp/lfm2d.sock".into()),
            opinion_specs: vec!["/specs/a.json".into()],
            ..base()
        };
        let err = cli.validate().expect_err("a spec with no model to serve it");
        assert!(err.contains("--opinion-spec"), "{err}");
    }

    #[test]
    fn adjudicator_prompt_is_no_longer_a_flag() {
        let err = Cli::try_parse_from(["lfm2d", "--adjudicator-prompt", "/specs/a.json"])
            .expect_err("the removed flag must be a parse error, not silently ignored");
        assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument, "{err}");
    }

    // -------------------------------------------- real clap::Parser parsing
    //
    // Everything above builds a `Cli` by hand and never touches the
    // `#[arg(...)]` attributes at all — a typo in `value_delimiter` or
    // `action` would compile fine and pass every test above while silently
    // breaking real CLI parsing. These two exercise `Cli::parse_from`
    // directly, the same entry point `main.rs` uses via `Cli::parse()`.

    #[test]
    fn token_classifier_dir_parses_repeated_flags_and_comma_lists() {
        let cli = Cli::parse_from([
            "lfm2d",
            "--token-classifier-dir",
            "/models/pii",
            "--token-classifier-dir",
            "/models/secrets,/models/router-guard",
        ]);
        assert_eq!(
            cli.token_classifier_dir,
            vec![
                PathBuf::from("/models/pii"),
                PathBuf::from("/models/secrets"),
                PathBuf::from("/models/router-guard"),
            ]
        );
    }

    #[test]
    fn opinion_spec_capacity_defaults_to_eight_and_rejects_zero() {
        let cli = Cli::parse_from(["lfm2d"]);
        assert_eq!(cli.opinion_spec_capacity, 8);
        let mut cli = base();
        cli.router_dir = Some("/tmp/router".into());
        cli.socket_path = Some("/tmp/lfm2d.sock".into());
        cli.opinion_spec_capacity = 0;
        let err = cli
            .validate()
            .expect_err("a capacity of 0 leaves no room for any upload");
        assert!(err.contains("opinion-spec-capacity"), "{err}");
    }

    #[test]
    fn log_input_hash_requires_an_explicit_value_not_a_bare_flag() {
        let cli = Cli::parse_from(["lfm2d", "--log-input-hash", "true"]);
        assert!(cli.log_input_hash);
        let cli = Cli::parse_from(["lfm2d"]);
        assert!(!cli.log_input_hash, "must default to false when unset");
        assert!(
            Cli::try_parse_from(["lfm2d", "--log-input-hash"]).is_err(),
            "a bare --log-input-hash must be refused, not read as true"
        );
    }

    // ------------------------------------------------------------- --no-probe
    //
    // `probe` defaults true and only the bare `--no-probe` flag (or a
    // falsy `LFM2D_PROBE`) turns it off — the opposite shape from
    // `--log-input-hash`'s "explicit value required" rule above, and worth
    // pinning explicitly since a clap `ArgAction`/env mixup here would
    // silently leave `/v1/probe` on, or off, regardless of the flag.

    /// `std::env::set_var`/`remove_var` mutate real process state, which
    /// every `#[test]` in this binary shares — a `--no-probe`/default test
    /// that never touches the env var can still observe a STALE
    /// `LFM2D_PROBE` a concurrently-running test set a moment ago, since
    /// `cargo test` runs tests in parallel by default. So every test that
    /// cares what `probe` resolves to (env-driven or not) takes this one
    /// lock and leaves the var unset on exit — that is the only way any of
    /// them gets a clean environment to parse against. (Rust 2024 makes
    /// `set_var`/`remove_var` themselves `unsafe`: no other thread may read
    /// the environment while they run, which this lock is what guarantees.)
    static PROBE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn probe_defaults_on_and_the_bare_flag_turns_it_off() {
        let _guard = PROBE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: guarded by PROBE_ENV_LOCK; no other thread reads or
        // writes LFM2D_PROBE while this lock is held.
        unsafe {
            std::env::remove_var("LFM2D_PROBE");
        }
        let cli = Cli::parse_from(["lfm2d"]);
        assert!(cli.probe, "on by default");
        let cli = Cli::parse_from(["lfm2d", "--no-probe"]);
        assert!(!cli.probe, "the bare flag needs no value");
    }

    /// The actual value `main.rs` hands to `adjudicator::router`'s
    /// `probe_enabled` argument, exercised through real clap parsing —
    /// see `probe_route_enabled`'s doc comment for why this exists as its
    /// own test rather than folding into the one above.
    #[test]
    fn probe_route_enabled_matches_the_no_probe_flag() {
        let _guard = PROBE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: guarded by PROBE_ENV_LOCK.
        unsafe {
            std::env::remove_var("LFM2D_PROBE");
        }
        assert!(Cli::parse_from(["lfm2d"]).probe_route_enabled());
        assert!(!Cli::parse_from(["lfm2d", "--no-probe"]).probe_route_enabled());
    }

    #[test]
    fn probe_env_var_accepts_boolish_spellings() {
        let _guard = PROBE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for (value, want) in [("0", false), ("false", false), ("1", true), ("true", true)] {
            // SAFETY: guarded by PROBE_ENV_LOCK above.
            unsafe {
                std::env::set_var("LFM2D_PROBE", value);
            }
            let cli = Cli::parse_from(["lfm2d"]);
            assert_eq!(cli.probe, want, "LFM2D_PROBE={value:?}");
        }
        // SAFETY: same guard; leave the environment clean for every other
        // test in this file that parses a `Cli` and assumes `LFM2D_PROBE`
        // is unset.
        unsafe {
            std::env::remove_var("LFM2D_PROBE");
        }
    }
}
