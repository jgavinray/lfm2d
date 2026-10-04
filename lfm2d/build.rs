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

    // The SYCL kernels link as `libcandle_sycl.so` from candle-sycl-kernels'
    // OUT_DIR. That crate adds an rpath for its own targets, but a downstream
    // binary does not inherit it: `cargo test` papers over this with a
    // LD_LIBRARY_PATH, and a daemon started by anything else fails to load the
    // library. Add the rpath here, where the final link happens. The directory
    // comes from the kernels crate's `links = "candle_sycl_kernels"` metadata
    // (`DEP_CANDLE_SYCL_KERNELS_LIB_DIR`), which cargo hands to this build
    // script precisely because lfm2d depends on that crate directly under the
    // sycl feature. This replaces globbing `build/` for a
    // `candle-sycl-kernels-<hash>` directory and taking the newest-named one
    // that holds a .so — which a stale directory from a superseded candle pin
    // could win outright, silently rpathing the wrong kernels into the
    // binary (tobert/lfm2d#3 item 8). A missing variable is a hard failure:
    // the pinned candle lacks the links key, so the sycl build must name the
    // paired candle branch instead of guessing.
    // No `cargo:rerun-if-changed={dir}` here (the old picker emitted one):
    // a changed library directory changes DEP_CANDLE_SYCL_KERNELS_LIB_DIR,
    // and that already reruns this build script.
    if std::env::var_os("CARGO_FEATURE_SYCL").is_some() {
        let dir = std::env::var("DEP_CANDLE_SYCL_KERNELS_LIB_DIR").unwrap_or_else(|_| {
            panic!(
                "the sycl feature requires candle-sycl-kernels' links metadata \
                 (DEP_CANDLE_SYCL_KERNELS_LIB_DIR); bump the candle pin to a rev \
                 carrying its links key (tobert/candle sycl-kernel-links)"
            )
        });
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
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
