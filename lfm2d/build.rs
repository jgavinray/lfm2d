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
    // library. Add the rpath here, where the final link happens.
    if std::env::var_os("CARGO_FEATURE_SYCL").is_some() {
        if let Some(dir) = sycl_kernel_dir() {
            println!("cargo:rerun-if-changed={}", dir.display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        } else {
            panic!("sycl feature is on but no candle-sycl-kernels build directory holds libcandle_sycl.so");
        }
    }
}

/// `target/<profile>/build/candle-sycl-kernels-<hash>/out`, found from this
/// build script's own OUT_DIR (`target/<profile>/build/lfm2d-<hash>/out`).
fn sycl_kernel_dir() -> Option<std::path::PathBuf> {
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").ok()?);
    let build = out.parent()?.parent()?;
    let mut found: Vec<_> = std::fs::read_dir(build)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("candle-sycl-kernels-"))
        })
        .map(|p| p.join("out"))
        .filter(|p| p.join("libcandle_sycl.so").is_file())
        .collect();
    found.sort();
    found.pop()
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
