//! Reads the candle fork revision this build resolves from the workspace's
//! `Cargo.lock` and hands it to the crate as `LFM2D_CANDLE_REV`
//! (`adjudicator::CANDLE_REV`, part of every `snapshot_id`). A lockfile
//! without a candle-core entry is a build failure, never a blank identity.

use std::path::Path;

fn main() {
    // At run time, not `env!`: cargo reuses one build-script binary across
    // checkouts that share a target dir (a worktree built into main's baked
    // in its own, since-deleted path on 2026-09-26), and sets this for the
    // tree it is building now.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR for build scripts");
    let lock = Path::new(&manifest).join("../Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let text = std::fs::read_to_string(&lock)
        .unwrap_or_else(|e| panic!("reading {}: {e}", lock.display()));
    let rev = candle_rev(&text).unwrap_or_else(|| {
        panic!("{} has no candle-core package with a source; cannot name the candle build", lock.display())
    });
    println!("cargo:rustc-env=LFM2D_CANDLE_REV={rev}");
}

/// The git commit after `#` for a git source, or `crates.io:<version>` for a
/// registry one. Two `candle-core` packages in one lockfile would make "the
/// build" ambiguous, so that is a build failure rather than a first-wins pick.
fn candle_rev(lock: &str) -> Option<String> {
    let found = lock.split("[[package]]").filter(|b| b.lines().any(|l| l == "name = \"candle-core\"")).count();
    assert!(found <= 1, "Cargo.lock holds {found} candle-core packages; snapshot_id cannot name one build");
    for block in lock.split("[[package]]") {
        let field = |key: &str| {
            block.lines().find_map(|l| l.strip_prefix(&format!("{key} = \""))?.strip_suffix('"').map(str::to_string))
        };
        if field("name").as_deref() != Some("candle-core") {
            continue;
        }
        let source = field("source")?;
        return match source.split_once('#') {
            Some((_, commit)) if source.starts_with("git+") => Some(commit.to_string()),
            _ if source.starts_with("registry+") => Some(format!("crates.io:{}", field("version")?)),
            _ => None,
        };
    }
    None
}
