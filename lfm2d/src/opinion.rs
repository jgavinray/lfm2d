//! System-1 reads over the resident adjudicator: one prefill, no decoding.
//!
//! The generative adjudicator writes a whole report to reach one verdict:
//! ~80 tokens at ~15 ms each, after the prefill. An opinion stands the model at
//! the verdict slot instead — the prompt spec's `opinion.prefill` is written as
//! if the assistant had already begun its answer — and scores every option in
//! `opinion.options` by teacher-forcing its full token sequence, terminator
//! included. Nothing is sampled, so there is nothing for a grammar to mask and
//! nothing to argmax away: the caller gets the distribution and decides.
//!
//! Why full sequences rather than first tokens: an option can span several
//! tokens, and forcing bytes through a grammar walks an off-canonical path
//! (gibberish and `ls -la` once scored one point apart that way). Scoring the
//! tokenizer's own encoding of `prefill + option + close` stays on the path the
//! model would have written.
//!
//! Why the tokenization is split at the longest common prefix of the options'
//! full encodings, not at the end of the prefill: a BPE merge can cross the
//! prefill/option boundary. Splitting where the encodings first diverge means
//! every option is scored on its canonical tokens and the shared part is
//! prefilled once.
use candle_core::{D, Result, Tensor};
use candle_transformers::models::quantized_lfm2_moe::{Model, Observer, State};
use serde::{Deserialize, Serialize};

/// The question a spec asks at its verdict slot.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpinionSpec {
    /// Text the assistant's turn is continued with before the options, e.g.
    /// `{"verdict": "`. Rendered after the spec's reasoning opening.
    pub prefill: String,
    /// The answer set, in the order the response reports it.
    pub options: Vec<String>,
    /// Written after each option, e.g. the closing quote. It makes an option
    /// that is a prefix of another (`ask` / `asking`) score as a whole word,
    /// and is part of every option's score.
    pub close: String,
}

impl OpinionSpec {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.options.len() < 2 {
            return Err("opinion.options needs at least two options".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        for option in &self.options {
            if option.is_empty() {
                return Err("opinion.options must not contain an empty option".into());
            }
            if !seen.insert(option) {
                return Err(format!("opinion.options repeats {option:?}"));
            }
        }
        if self.close.is_empty() {
            return Err("opinion.close must not be empty".into());
        }
        for text in [&self.prefill, &self.close].into_iter().chain(&self.options) {
            if crate::chat::control_marker_in(text).is_some() {
                return Err("opinion text must not carry model control tokens".into());
            }
        }
        Ok(())
    }
}

/// One option's standing at the slot.
#[derive(Clone, Debug, Serialize)]
pub struct OptionScore {
    pub option: String,
    /// Sum of the RAW logprobs (full-vocabulary denominators) of the option's
    /// tokens, terminator included. `<= 0`.
    pub logprob: f32,
    /// RAW logprob of the option's first token alone at the slot. The F9
    /// slot score is this number; `logprob` adds the rest of the word and
    /// the close.
    pub first_logprob: f32,
    /// Renormalized over the options. Read it beside `sequence_mass`: when the
    /// model put little mass on the answer set, this is noise that looks like
    /// an answer.
    pub prob: f32,
    pub tokens: Vec<u32>,
}

/// The whole read. The response never picks a winner; the caller does.
#[derive(Clone, Debug, Serialize)]
pub struct OpinionRead {
    pub options: Vec<OptionScore>,
    /// logsumexp of the options' sequence logprobs: the log probability that
    /// the model writes exactly one of the options and its terminator.
    pub sequence_mass: f32,
    /// Raw log mass on the distinct first tokens of the options' continuations.
    pub first_token_mass: f32,
    /// Tokens prefilled before the options diverge, and tokens teacher-forced
    /// across all options after that.
    pub shared_tokens: usize,
    pub scored_tokens: usize,
    /// sha256 of the rendered text the options continue: prefix, user turn,
    /// then on `/v1/adjudicate` (`opinion: true`) the reasoning opening and
    /// the spec's fixed prefill, on `/v1/opinion` the generated description
    /// through this answer's own slot (`rendered: true` returns the text to
    /// the last asked slot; an earlier answer hashes a prefix of it). The two
    /// endpoints hash different bytes for the same input. Results carry it
    /// so a replay can prove it rendered what the daemon ran.
    pub rendered_sha256: String,
}

/// Length of the longest prefix every sequence shares.
pub fn common_prefix_len(seqs: &[Vec<u32>]) -> usize {
    let Some(first) = seqs.first() else { return 0 };
    let mut n = first.len();
    for s in &seqs[1..] {
        n = n.min(first.iter().zip(s).take_while(|(a, b)| a == b).count());
    }
    n
}

/// The options' full encodings of `rendered + option + close`, split where
/// they diverge: the shared prefix (prefilled once) and one continuation per
/// option (teacher-forced), in `spec.options` order.
///
/// Whether the split leaves every option a continuation depends on the
/// tokenizer and the spec alone, never on the input: the input sits before
/// the assistant turn's added tokens. So the load path runs this on a probe
/// input and refuses to start on a spec whose close cannot separate its
/// options; at request time the same failure is an invariant violation.
pub struct SplitOptions {
    pub shared: Vec<u32>,
    pub continuations: Vec<Vec<u32>>,
}

pub fn split_options(
    tokenizer: &tokenizers::Tokenizer,
    rendered: &str,
    spec: &OpinionSpec,
) -> std::result::Result<SplitOptions, String> {
    let encodings = spec
        .options
        .iter()
        .map(|option| {
            tokenizer
                .encode(format!("{rendered}{option}{}", spec.close), false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| e.to_string())
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let shared = common_prefix_len(&encodings);
    let continuations: Vec<Vec<u32>> = encodings.iter().map(|e| e[shared..].to_vec()).collect();
    if let Some(i) = continuations.iter().position(Vec::is_empty) {
        return Err(format!(
            "opinion option {:?} encodes as a prefix of every other option's encoding, so \
             nothing is left to score; opinion.close must separate the options",
            spec.options[i]
        ));
    }
    Ok(SplitOptions {
        shared: encodings[0][..shared].to_vec(),
        continuations,
    })
}

fn log_softmax_rows(logits: &Tensor) -> Result<Vec<Vec<f32>>> {
    candle_nn::ops::log_softmax(logits, D::Minus1)?.to_vec2::<f32>()
}

struct LastResidual {
    layer: usize,
    residual: Option<Tensor>,
}
impl Observer for LastResidual {
    fn residual(&mut self, layer: usize, x: &Tensor) -> Result<()> {
        if layer == self.layer {
            self.residual = Some(x.clone());
        }
        Ok(())
    }
}

/// One continuation's teacher-forced read: the RAW logprob of every one of
/// its tokens, in order, and their sum. `token_logprobs.len() ==
/// continuation.len()` always; `sequence_logprob == token_logprobs.iter().sum()`
/// (up to float summation order — [`score_continuations`] takes the summed
/// value straight from here rather than re-adding, so there is no second
/// rounding to disagree with this one).
pub struct ContinuationScore {
    pub token_logprobs: Vec<f32>,
    pub sequence_logprob: f32,
}

/// Raw logprob of each continuation, teacher-forced from `state`, whose
/// next-token logits are `logits` (`(1, vocab)`). `state` is not advanced.
///
/// The first token of every continuation is read from `logits`; the rest from
/// one forward over the continuation minus its last token, projected at every
/// position through the model's own final norm and output head.
///
/// This is the one forward-pass loop every teacher-forced read in this
/// crate shares: `/v1/opinion`'s options ([`score_continuations`] below) and
/// `/v1/probe`'s `continuations` (`crate::probe_api`, which wants the
/// PER-TOKEN logprobs this returns and `score_continuations` throws away)
/// are the same computation asked for two different levels of detail, not
/// two computations.
pub fn score_continuations_verbose(
    model: &Model,
    state: &State,
    logits: &Tensor,
    continuations: &[Vec<u32>],
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<ContinuationScore>> {
    let first = log_softmax_rows(logits)?;
    let first = first
        .first()
        .ok_or_else(|| candle_core::Error::Msg("empty logits".into()))?;
    let last_layer = model.layers().len() - 1;
    let mut scores = Vec::with_capacity(continuations.len());
    for cont in continuations {
        check()?;
        let (&head, rest) = cont
            .split_first()
            .ok_or_else(|| candle_core::Error::Msg("empty continuation".into()))?;
        let mut token_logprobs = Vec::with_capacity(cont.len());
        token_logprobs.push(first[head as usize]);
        if !rest.is_empty() {
            let mut branch = state.clone();
            let mut tap = LastResidual {
                layer: last_layer,
                residual: None,
            };
            model.forward_observed(&cont[..cont.len() - 1], &mut branch, &mut tap)?;
            let residual = tap
                .residual
                .ok_or_else(|| candle_core::Error::Msg("last residual not observed".into()))?;
            let rows = log_softmax_rows(&model.project(&residual)?.squeeze(0)?)?;
            if rows.len() != rest.len() {
                return Err(candle_core::Error::Msg(format!(
                    "projected {} rows for {} continuation tokens",
                    rows.len(),
                    rest.len()
                )));
            }
            for (row, &next) in rows.iter().zip(rest) {
                token_logprobs.push(row[next as usize]);
            }
        }
        let total: f32 = token_logprobs.iter().sum();
        // A non-finite score would flow through logsumexp and the renormalized
        // probabilities as a number that looks like an answer. Crash instead,
        // as `step_distribution` does for its logits.
        if !total.is_finite() {
            return Err(candle_core::Error::Msg(format!(
                "non-finite sequence logprob {total} for continuation {cont:?}"
            )));
        }
        scores.push(ContinuationScore { token_logprobs, sequence_logprob: total });
    }
    Ok(scores)
}

/// [`score_continuations_verbose`], keeping only each continuation's summed
/// sequence logprob — what `/v1/opinion` and `/v1/adjudicate`'s `opinion:
/// true` read.
pub fn score_continuations(
    model: &Model,
    state: &State,
    logits: &Tensor,
    continuations: &[Vec<u32>],
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<f32>> {
    Ok(score_continuations_verbose(model, state, logits, continuations, check)?
        .into_iter()
        .map(|c| c.sequence_logprob)
        .collect())
}

pub fn logsumexp(values: &[f32]) -> f32 {
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return max;
    }
    max + values.iter().map(|&v| (v - max).exp()).sum::<f32>().ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Model {
        let mut f = std::io::Cursor::new(include_bytes!("../tests/fixtures/lfm2-moe/tiny.gguf"));
        let ct = candle_core::quantized::gguf_file::Content::read(&mut f).unwrap();
        Model::from_gguf(ct, &mut f, &candle_core::Device::Cpu).unwrap()
    }

    fn ok() -> Result<()> {
        Ok(())
    }

    #[test]
    fn common_prefix_stops_at_first_divergence() {
        assert_eq!(
            common_prefix_len(&[vec![1, 2, 3, 4], vec![1, 2, 5], vec![1, 2, 3]]),
            2
        );
        assert_eq!(common_prefix_len(&[vec![1, 2], vec![3]]), 0);
        assert_eq!(common_prefix_len(&[vec![7, 8, 9]]), 3);
        // The shortest sequence last: an implementation that compares each
        // sequence with the first but forgets to clamp reports 3 here.
        assert_eq!(
            common_prefix_len(&[vec![1, 2, 3], vec![1, 2, 3, 4], vec![1, 2]]),
            2
        );
        assert_eq!(common_prefix_len(&[vec![1, 2, 3], vec![1, 2, 3, 4], vec![]]), 0);
    }

    /// The batched read must equal decoding the continuation one token at a
    /// time, which is what the generative path does. A mistake in the
    /// position alignment (reading row i for token i instead of i+1) fails
    /// this by nats, not by rounding.
    #[test]
    fn batched_score_matches_token_by_token_decode() {
        let model = fixture();
        let mut state = model.new_state();
        let logits = model.forward(&[1, 2, 3], &mut state).unwrap();
        let conts = vec![vec![4, 5, 6], vec![7], vec![4, 9]];
        let got = score_continuations(&model, &state, &logits, &conts, &ok).unwrap();
        for (cont, got) in conts.iter().zip(&got) {
            let mut s = state.clone();
            let mut l = logits.clone();
            let mut want = 0f32;
            for &t in cont {
                want += log_softmax_rows(&l).unwrap()[0][t as usize];
                l = model.forward(&[t], &mut s).unwrap();
            }
            assert!(
                (got - want).abs() < 1e-3,
                "cont {cont:?}: batched {got} vs stepwise {want}"
            );
        }
    }

    /// `score_continuations` must be reading `sequence_logprob` off
    /// `score_continuations_verbose`, not recomputing it independently — a
    /// mutation that made the two disagree (e.g. the wrapper re-summing a
    /// stale copy) would pass every other test here and still be wrong.
    /// Also pins the per-token breakdown itself against the same
    /// step-by-step decode `batched_score_matches_token_by_token_decode`
    /// checks the sequence total against.
    #[test]
    fn verbose_per_token_logprobs_sum_to_the_sequence_logprob_and_match_score_continuations() {
        let model = fixture();
        let mut state = model.new_state();
        let logits = model.forward(&[1, 2, 3], &mut state).unwrap();
        let conts = vec![vec![4, 5, 6], vec![7], vec![4, 9]];
        let verbose = score_continuations_verbose(&model, &state, &logits, &conts, &ok).unwrap();
        let terse = score_continuations(&model, &state, &logits, &conts, &ok).unwrap();
        assert_eq!(verbose.len(), conts.len());
        for ((cont, v), t) in conts.iter().zip(&verbose).zip(&terse) {
            assert_eq!(v.token_logprobs.len(), cont.len(), "cont {cont:?}");
            let summed: f32 = v.token_logprobs.iter().sum();
            assert!((summed - v.sequence_logprob).abs() < 1e-6, "cont {cont:?}");
            assert_eq!(v.sequence_logprob, *t, "the wrapper must read this exact value, not re-derive it");

            let mut s = state.clone();
            let mut l = logits.clone();
            for (i, &tok) in cont.iter().enumerate() {
                let want = log_softmax_rows(&l).unwrap()[0][tok as usize];
                assert!(
                    (v.token_logprobs[i] - want).abs() < 1e-3,
                    "cont {cont:?} token {i}: batched {} vs stepwise {want}",
                    v.token_logprobs[i]
                );
                l = model.forward(&[tok], &mut s).unwrap();
            }
        }
    }

    #[test]
    fn scoring_does_not_advance_the_shared_state() {
        let model = fixture();
        let mut state = model.new_state();
        let logits = model.forward(&[1, 2, 3], &mut state).unwrap();
        score_continuations(&model, &state, &logits, &[vec![4, 5, 6]], &ok).unwrap();
        assert_eq!(state.len(), 3);
    }

    #[test]
    fn spec_refuses_degenerate_answer_sets() {
        let spec = |options: &[&str], close: &str| OpinionSpec {
            prefill: "{\"verdict\": \"".into(),
            options: options.iter().map(|s| s.to_string()).collect(),
            close: close.into(),
        };
        assert!(spec(&["allow", "ask"], "\"").validate().is_ok());
        assert!(spec(&["allow"], "\"").validate().is_err());
        assert!(spec(&["allow", "allow"], "\"").validate().is_err());
        assert!(spec(&["allow", ""], "\"").validate().is_err());
        assert!(spec(&["allow", "ask"], "").validate().is_err());
        assert!(spec(&["allow", "<|im_end|>"], "\"").validate().is_err());
    }

    #[test]
    fn logsumexp_is_stable() {
        let v = logsumexp(&[-1000.0, -1000.0]);
        assert!((v - (-1000.0 + 2f32.ln())).abs() < 1e-3);
        assert_eq!(logsumexp(&[f32::NEG_INFINITY]), f32::NEG_INFINITY);
    }
}
