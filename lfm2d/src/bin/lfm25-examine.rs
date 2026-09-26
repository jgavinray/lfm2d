//! Examine one prompt inside LFM2.5 and write the record to disk.
//!
//! The prompt is either exact text (`--text-file`, control tokens and all) or
//! the daemon's own rendering of a prompt spec plus one input (`--prompt`,
//! `--input`), optionally continued into the assistant's turn
//! (`--assistant-prefill`) to stand at an answer slot. See `lfm2d::examine`
//! for what the record holds.
//!
//! `--inputs-file` examines many inputs against one prompt spec with the model
//! loaded once, writing one record per line to `examinations.jsonl`. With
//! `--record-from-shared-prefix` nothing is recorded for the tokens every input
//! shares: a causal model routes a shared prefix identically every time, so
//! recording it once per input is pure repetition.
//!
//! The record carries the prompt text, which may be a corpus row, and corpora
//! do not live in this repo. So `--out` is required and never defaulted.

use clap::Parser;
use lfm2d::adjudicator::{Checkpoint, PromptSpec, validate_text};
use lfm2d::examine::{ExamineSpec, examine};
use lfm2d::hash::sha256_hex_bytes;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Expert routing, residual norms and the logit lens for one LFM2.5 prompt")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    /// Directory for examination.json. Keep it outside this repo.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, value_enum, default_value = "auto")]
    device: lfm2d::device::DeviceArg,
    #[arg(long, default_value_t = 0)]
    device_index: usize,

    /// Exact text to examine, control tokens included.
    #[arg(long, conflicts_with_all = ["prompt", "input", "assistant_prefill"])]
    text_file: Option<PathBuf>,
    /// A prompt spec file, the same JSON the daemon's --opinion-spec loads
    /// and `POST /v1/opinion/specs` accepts.
    #[arg(long)]
    prompt: Option<PathBuf>,
    /// The user turn's content, exactly as `/v1/adjudicate`'s `input`: for
    /// the state an opinion reads, that is `{facts}{input_label}:\n{input}`
    /// with the spec's own label.
    #[arg(long, requires = "prompt", conflicts_with = "inputs_file")]
    input: Option<String>,
    /// JSON lines of {"name", "input", "assistant_prefill"?}: a batch against
    /// --prompt. Writes examinations.jsonl and batch.json.
    #[arg(long, requires = "prompt", conflicts_with_all = ["text_file", "assistant_prefill"])]
    inputs_file: Option<PathBuf>,
    /// Batch only: record nothing for the token prefix all inputs share.
    #[arg(long, requires = "inputs_file")]
    record_from_shared_prefix: bool,
    /// Text the assistant has already written, to stand at an answer slot.
    /// An instrument, like `/v1/probe`'s `text`: control markers are not
    /// refused (only a second reasoning region is, under a closed spec).
    #[arg(long, default_value = "")]
    assistant_prefill: String,

    /// NAME=ID,ID,... a token set to follow through the lens. Repeatable.
    #[arg(long = "set")]
    sets: Vec<String>,
    /// NAME=WORD,WORD,... like --set, taking the FIRST token of each word
    /// encoded alone: the token that discriminates it at an answer slot.
    #[arg(long = "set-first-token")]
    first_token_sets: Vec<String>,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    /// A token position, or `last`. Repeatable.
    #[arg(long = "top-k-position")]
    top_k_positions: Vec<String>,
    /// Read the lens only at this position, or `last`. Repeatable, in
    /// increasing order. Default: every recorded position.
    #[arg(long = "lens-position")]
    lens_positions: Vec<String>,
    #[arg(long)]
    router_logits: bool,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchInput {
    name: String,
    input: String,
    #[serde(default)]
    assistant_prefill: String,
}

fn named_list(arg: &str) -> Result<(String, Vec<String>), String> {
    let (name, items) = arg
        .split_once('=')
        .ok_or_else(|| format!("expected NAME=ITEM,ITEM but got {arg:?}"))?;
    let items: Vec<String> = items.split(',').map(str::to_string).collect();
    if name.is_empty() || items.iter().any(String::is_empty) {
        return Err(format!("expected NAME=ITEM,ITEM but got {arg:?}"));
    }
    Ok((name.to_string(), items))
}

fn position(arg: &str, n_tokens: usize) -> Result<usize, String> {
    if arg == "last" {
        return Ok(n_tokens - 1);
    }
    arg.parse()
        .map_err(|_| format!("a position is a token index or `last`, not {arg:?}"))
}

fn render(spec: &PromptSpec, input: &str, prefill: &str) -> Result<String, String> {
    // The daemon refuses this input, so its rendering of it does not exist to
    // be examined. Exact text goes through --text-file.
    validate_text(input)?;
    Ok(format!(
        "{}{}",
        spec.render_prefix()?,
        spec.render_user_turn_with_prefill(input, prefill)?
    ))
}

fn run(args: Args) -> Result<(), String> {
    let text = |e: std::io::Error, p: &Path| format!("{}: {e}", p.display());
    std::fs::create_dir_all(&args.out).map_err(|e| text(e, &args.out))?;
    let batch = args.inputs_file.is_some();
    let target = args.out.join(if batch { "examinations.jsonl" } else { "examination.json" });
    if target.exists() {
        return Err(format!("{} already exists", target.display()));
    }
    let prompt_spec = |path: &Path| -> Result<PromptSpec, String> {
        serde_json::from_slice(&std::fs::read(path).map_err(|e| text(e, path))?)
            .map_err(|e| format!("{}: {e}", path.display()))
    };

    // (name, rendered text, where it came from)
    let mut items: Vec<(String, String, serde_json::Value)> = Vec::new();
    match (&args.text_file, &args.prompt, &args.input, &args.inputs_file) {
        (Some(path), None, None, None) => items.push((
            "examination".into(),
            std::fs::read_to_string(path).map_err(|e| text(e, path))?,
            serde_json::json!({"text_file": path}),
        )),
        (None, Some(path), Some(input), None) => items.push((
            "examination".into(),
            render(&prompt_spec(path)?, input, &args.assistant_prefill)?,
            serde_json::json!({
                "prompt": path,
                "input": input,
                "assistant_prefill": args.assistant_prefill,
            }),
        )),
        (None, Some(path), None, Some(file)) => {
            let spec = prompt_spec(path)?;
            let lines = std::fs::read_to_string(file).map_err(|e| text(e, file))?;
            let mut names = std::collections::BTreeSet::new();
            for (n, line) in lines.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
                let row: BatchInput = serde_json::from_str(line)
                    .map_err(|e| format!("{} line {}: {e}", file.display(), n + 1))?;
                if !names.insert(row.name.clone()) {
                    return Err(format!("{}: name {:?} is used twice", file.display(), row.name));
                }
                let rendered = render(&spec, &row.input, &row.assistant_prefill)
                    .map_err(|e| format!("{} line {}: {e}", file.display(), n + 1))?;
                items.push((row.name, rendered, serde_json::json!({"prompt": path, "inputs_file": file})));
            }
            if items.is_empty() {
                return Err(format!("{} holds no inputs", file.display()));
            }
        }
        _ => return Err("give --text-file, or --prompt with --input or --inputs-file".into()),
    }

    let checkpoint = Checkpoint::load(&args.model, &args.tokenizer, args.device, args.device_index)?;
    let tokenizer = &checkpoint.tokenizer;
    let mut tokenized = Vec::with_capacity(items.len());
    for (name, rendered, _) in &items {
        let ids = tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        if ids.is_empty() {
            return Err(format!("{name}: the prompt tokenizes to nothing"));
        }
        tokenized.push(ids);
    }
    let record_from = if args.record_from_shared_prefix {
        shared_prefix(&tokenized)
    } else {
        0
    };

    let mut token_sets = BTreeMap::new();
    let mut resolution = Vec::new();
    let mut insert = |name: String, ids: Vec<u32>| match token_sets.insert(name.clone(), ids) {
        Some(_) => Err(format!("token set {name} is given twice")),
        None => Ok(()),
    };
    for arg in &args.sets {
        let (name, items) = named_list(arg)?;
        let ids = items
            .iter()
            .map(|i| i.parse().map_err(|_| format!("--set {name}: {i:?} is not a token id")))
            .collect::<Result<_, _>>()?;
        insert(name, ids)?;
    }
    for arg in &args.first_token_sets {
        let (name, words) = named_list(arg)?;
        let mut ids = Vec::new();
        for word in &words {
            let encoded = tokenizer.encode(word.as_str(), false).map_err(|e| e.to_string())?;
            let first = *encoded
                .get_ids()
                .first()
                .ok_or_else(|| format!("{word:?} tokenizes to nothing"))?;
            resolution.push(serde_json::json!({
                "set": name,
                "word": word,
                "id": first,
                "piece": tokenizer.id_to_token(first),
                "word_tokens": encoded.get_ids().len(),
            }));
            ids.push(first);
        }
        insert(name, ids)?;
    }

    let identity = serde_json::json!({
        "model_id": checkpoint.model_id,
        "weight_hash": checkpoint.weight_hash,
        "tokenizer_hash": checkpoint.tokenizer_hash,
        "weight_dtypes": checkpoint.weight_dtypes,
        "backend": checkpoint.execution.backend.as_str(),
        "device": checkpoint.execution.identity,
        "candle_rev": lfm2d::adjudicator::CANDLE_REV,
    });
    let positions = |args: &[String], n: usize| -> Result<Vec<usize>, String> {
        args.iter().map(|p| position(p, n)).collect()
    };
    let begin = std::time::Instant::now();
    let mut sink = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&target)
            .map_err(|e| text(e, &target))?,
    );
    use std::io::Write;
    for ((name, rendered, source), tokens) in items.iter().zip(&tokenized) {
        let spec = ExamineSpec {
            token_sets: token_sets.clone(),
            top_k: args.top_k,
            top_k_positions: positions(&args.top_k_positions, tokens.len())?,
            router_logits: args.router_logits,
            lens_positions: positions(&args.lens_positions, tokens.len())?,
            record_from,
        };
        let examination = examine(&checkpoint.model, tokens, &spec, |id| tokenizer.id_to_token(id))
            .map_err(|e| format!("{name}: {e}"))?;
        let mut record = serde_json::json!({
            "name": name,
            "text_sha256": sha256_hex_bytes(rendered.as_bytes()),
            "examination": examination,
        });
        if !batch {
            // One record is a whole document; a batch says these once, in batch.json.
            let mut id = identity.clone();
            id["text_sha256"] = record["text_sha256"].clone();
            record = serde_json::json!({
                "identity": id,
                "source": source,
                "text": rendered,
                "first_token_resolution": resolution,
                "spec": spec,
                "examination": record["examination"],
            });
        }
        serde_json::to_writer(&mut sink, &record).map_err(|e| e.to_string())?;
        sink.write_all(b"\n").map_err(|e| text(e, &target))?;
    }
    sink.flush().map_err(|e| text(e, &target))?;
    let seconds = begin.elapsed().as_secs_f64();
    if batch {
        let about = args.out.join("batch.json");
        let body = serde_json::json!({
            "identity": identity,
            "source": items[0].2,
            "first_token_resolution": resolution,
            "record_from": record_from,
            "inputs": items.len(),
            "seconds": seconds,
        });
        std::fs::write(&about, serde_json::to_vec_pretty(&body).map_err(|e| e.to_string())?)
            .map_err(|e| text(e, &about))?;
    }
    println!(
        "{} examined, recorded from token {record_from}, {seconds:.2}s ({:.2}s each) -> {}",
        items.len(),
        seconds / items.len() as f64,
        target.display()
    );
    Ok(())
}

/// How many leading tokens every input shares, leaving each at least one token.
fn shared_prefix(all: &[Vec<u32>]) -> usize {
    let shortest = all.iter().map(Vec::len).min().unwrap_or(0);
    let mut n = 0;
    while n + 1 < shortest && all.iter().all(|t| t[n] == all[0][n]) {
        n += 1;
    }
    n
}

fn main() {
    if let Err(e) = run(Args::parse()) {
        eprintln!("lfm25-examine: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_lists_parse_or_say_why_not() {
        assert_eq!(
            named_list("labels=268,91").unwrap(),
            ("labels".to_string(), vec!["268".to_string(), "91".to_string()])
        );
        for bad in ["labels", "=1,2", "labels=", "labels=1,,2"] {
            assert!(named_list(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_shared_prefix_stops_at_the_first_difference_and_leaves_a_token() {
        assert_eq!(shared_prefix(&[vec![1, 2, 3, 4], vec![1, 2, 9, 4]]), 2);
        assert_eq!(shared_prefix(&[vec![1, 2, 3], vec![1, 2, 3, 4]]), 2);
        assert_eq!(shared_prefix(&[vec![7, 8]]), 1);
        assert_eq!(shared_prefix(&[vec![1], vec![2]]), 0);
    }

    #[test]
    fn last_is_the_final_token_and_nothing_else_is_guessed() {
        assert_eq!(position("last", 9).unwrap(), 8);
        assert_eq!(position("3", 9).unwrap(), 3);
        assert!(position("-1", 9).is_err());
        assert!(position("end", 9).is_err());
    }
}
