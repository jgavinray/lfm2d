//! Prompt bytes pinned against history, not against the code that renders
//! them now.
//!
//! On 2026-09-26 `PromptSpec::render_prefix` moved onto `lfm2d::chat` so that
//! tools render as the template's `tojson` does (template v3). The claim that
//! specs WITHOUT tools render the same bytes, and so keep their
//! `snapshot_id`, cannot be tested by comparing two paths through the new
//! code. These hashes were rendered by the code at `b9dd9f6`, the commit
//! before the change, from a detached worktree. A mismatch here is a changed
//! prompt: every consumer's `snapshot_id` for that spec moves with it.
use lfm2d::adjudicator::PromptSpec;
use lfm2d::hash::sha256_hex_bytes;

const INPUT: &str = "Email:\nWhere is my order? It's late — ¿dónde está?";

/// (spec file, its sha256 = spec id, template_version, sha256 of
/// `render_prefix`, sha256 of `render_prefix` + `render_user_turn(INPUT)`),
/// all as `b9dd9f6` rendered them.
const PINS: [(&str, &str, &str, &str, &str); 6] = [
    (
        "lfm2d/tests/fixtures/specs/email-triage-v1.json",
        "ae158b7daaf46b8e544c630a8479aa9a0cbf4369b63fe5b7d7c1b4db8c249bd4",
        "lfm25-single-user-v2-closed",
        "a25b7d412d5f00068912d9909100d384b28927a23d5048ba5f1bb74680ecb098",
        "3be3b6782b4b64effe7ea5437a9bc0ec04a16aed34fe4e809baea62a72d5e206",
    ),
    (
        "lfm2d/tests/fixtures/specs/email-triage-opinion-v1.json",
        "f295c9cfa7b3d7f7fdb681b8b5b0a4e629b522ffb2d53301ca0693bfb04d0cf9",
        "lfm25-single-user-v2-closed",
        "a25b7d412d5f00068912d9909100d384b28927a23d5048ba5f1bb74680ecb098",
        "3be3b6782b4b64effe7ea5437a9bc0ec04a16aed34fe4e809baea62a72d5e206",
    ),
    (
        "lfm2d/tests/fixtures/specs/email-verdict-opinion-v1.json",
        "7ea8534229afda014e5daaf8709e0ce83ecb734ddd70369d6522afc68df88d5d",
        "lfm25-single-user-v2-closed",
        "4b72ca10af36d3bb2af02a0c961e91e3e445a80196e028dad8cf8bbeefb25a1c",
        "a95abec8dab88ddf188c4cd8b6e7e047fc378b7e5db3fcef9ba2f8440d167411",
    ),
    (
        "demo/specs/email-triage-v2.json",
        "1c66281d9b83b805871aa8f6cf4a499e9bb31984c9548373684084118c5b39c8",
        "lfm25-single-user-v2-closed",
        "03612fbf2d6fa21007c80993d0b89d2a18216796589e455b272b95525ce4ec09",
        "1968a9827f8edd62a7e3bd10002b114d6a3276773562c29232f825d2239ebc3d",
    ),
    (
        "demo/web/static/life-decision-v2.json",
        "0995af933c7763ff7b6de306cd3542e8ad39ad93fe0e0cdd510430f7020df042",
        "lfm25-single-user-v2-closed",
        "05a7f4dd661ee52c500a8a24b2dbf19ca8b85247a6169429f7fb6a1e82fbcd71",
        "c13343ed0617117672ed2b1909fc6d0d6f7557772e69118ab38bc5a63b813f9f",
    ),
    (
        "demo/web/static/command-verdict-enum-v1.json",
        "02bc1de7fcee01b84828134208678e4767090c32b59414f4ca22910584861e4b",
        "lfm25-single-user-v2-closed",
        "18ec8b14e81ac6f3fb2afb0e87681cdbc46e51d2dafa6a13a4090786cd099013",
        "d04d9ba8b583815b16d0e78ef178d50979b797fea9965abe004564b51a68f420",
    ),
];

fn repo() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

#[test]
fn specs_without_tools_render_the_bytes_they_rendered_before_template_v3() {
    for (file, spec_id, version, prefix_sha, whole_sha) in PINS {
        let bytes = std::fs::read(repo().join(file)).unwrap();
        // A spec edit is not a renderer change: re-pin from the new bytes.
        assert_eq!(sha256_hex_bytes(&bytes), spec_id, "{file} changed; re-pin it");
        let spec: PromptSpec = serde_json::from_slice(&bytes).unwrap();
        assert!(spec.tools.is_empty(), "{file}");
        assert_eq!(spec.template_version(), version, "{file}");
        let prefix = spec.render_prefix().unwrap();
        assert_eq!(sha256_hex_bytes(prefix.as_bytes()), prefix_sha, "{file} prefix");
        let whole = format!("{prefix}{}", spec.render_user_turn(INPUT));
        assert_eq!(sha256_hex_bytes(whole.as_bytes()), whole_sha, "{file} prefix + user turn");
    }
}

#[test]
fn email_triage_v1_prefix_is_the_276_tokens_the_2026_09_25_run_recorded() {
    // Independent of the hashes above: benchmarks/system1/results/
    // 2026-09-25-email-triage-v1-confirm.json recorded prefix_tokens 276 for
    // spec id ae158b7d…, from the daemon of that day.
    let models = std::env::var_os("LFM2_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| repo().join(".models"));
    let path = std::env::var_os("LFM2D_ADJUDICATOR_TOKENIZER")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| models.join("LFM2.5-8B-A1B/tokenizer.json"));
    assert!(path.is_file(), "missing tokenizer at {}", path.display());
    let tokenizer = tokenizers::Tokenizer::from_file(&path).unwrap();
    let spec: PromptSpec = serde_json::from_slice(
        &std::fs::read(repo().join("lfm2d/tests/fixtures/specs/email-triage-v1.json")).unwrap(),
    )
    .unwrap();
    let prefix = spec.render_prefix().unwrap();
    assert_eq!(tokenizer.encode(prefix.as_str(), false).unwrap().len(), 276);
}
