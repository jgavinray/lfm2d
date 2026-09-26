//! Host-memory watchdog for tests that load real weights. Call
//! [`arm`] first thing in any test that loads a checkpoint; it is cheap and
//! idempotent, and one watchdog serves the whole test process.
//!
//! Why MemAvailable and not RSS: on zorak (gfx1151, unified memory) GPU
//! allocations are GTT, host RAM that never appears in the process's RSS.
//! On 2026-09-26 a cold 30k-token prefill test reached 133 GB of VM at 4 MB
//! RSS, and the kernel's OOM killer took the production lfm2d pods instead
//! of it. GTT does come out of MemAvailable, so this watches the host and
//! aborts THIS process when MemAvailable falls under a floor, before the
//! OOM killer has to choose.
//!
//! The floor is `LFM2_TEST_MEM_FLOOR_GIB` (GiB, fractions allowed), default
//! 8. It is shared by both crates' tests: the encoder's `tests/*.rs` and
//! `lfm2d/tests/support` include this file by `#[path]`.
#![allow(dead_code)]

use std::sync::OnceLock;
use std::time::Duration;

pub const FLOOR_ENV: &str = "LFM2_TEST_MEM_FLOOR_GIB";
pub const DEFAULT_FLOOR_GIB: f64 = 8.0;
/// How often the watchdog reads `/proc/meminfo` (a read is tens of µs).
pub const POLL: Duration = Duration::from_millis(5);

const KIB_PER_GIB: f64 = 1024.0 * 1024.0;

/// `MemAvailable` from `/proc/meminfo` text, in KiB.
pub fn mem_available_kib(meminfo: &str) -> Option<u64> {
    meminfo.lines().find_map(|line| {
        let rest = line.strip_prefix("MemAvailable:")?;
        rest.trim().strip_suffix("kB")?.trim().parse().ok()
    })
}

/// The floor in KiB from the environment's value; `None` is the default.
/// Anything but a positive finite number of GiB is an error, never a
/// silently disabled guard.
pub fn floor_kib_from(value: Option<&str>) -> Result<u64, String> {
    let gib = match value {
        None => DEFAULT_FLOOR_GIB,
        Some(v) => v
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|g| g.is_finite() && *g > 0.0)
            .ok_or_else(|| format!("{FLOOR_ENV}={v:?}: expected a positive number of GiB"))?,
    };
    Ok((gib * KIB_PER_GIB).round() as u64)
}

/// The host's MemAvailable now, in KiB. Panics when it cannot be read: a
/// guard that cannot see memory must not pretend to be watching it.
pub fn read_mem_available_kib() -> u64 {
    let text = std::fs::read_to_string("/proc/meminfo")
        .unwrap_or_else(|e| panic!("memory guard: cannot read /proc/meminfo: {e}"));
    mem_available_kib(&text).expect("memory guard: /proc/meminfo has no MemAvailable line")
}

/// The watchdog loop: read every `poll`, and hand the first reading under
/// `floor_kib` to `breach`, then stop. [`arm`] runs it with the real reader
/// and an abort; tests run it with fakes.
pub fn watch(
    floor_kib: u64,
    poll: Duration,
    mut read: impl FnMut() -> u64 + Send + 'static,
    breach: impl FnOnce(u64) + Send + 'static,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("memory-guard".into())
        .spawn(move || loop {
            let available = read();
            if available < floor_kib {
                breach(available);
                return;
            }
            std::thread::sleep(poll);
        })
        .expect("memory guard: spawn the watchdog thread")
}

fn own_status(field: &str) -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with(field)).map(|l| l.split_whitespace().skip(1).collect::<Vec<_>>().join(" ")))
        .unwrap_or_else(|| "?".into())
}

fn abort_under_floor(available_kib: u64, floor_kib: u64) -> ! {
    eprintln!(
        "\nmemory guard: host MemAvailable {:.2} GiB is under the {:.2} GiB floor ({FLOOR_ENV}); \
         aborting test process {} (VmSize {}, VmRSS {}; GPU memory on unified-memory hosts is \
         GTT and not in RSS) before the OOM killer picks another process\n",
        available_kib as f64 / KIB_PER_GIB,
        floor_kib as f64 / KIB_PER_GIB,
        std::process::id(),
        own_status("VmSize:"),
        own_status("VmRSS:"),
    );
    std::process::abort()
}

/// Start the process-wide watchdog (once; later calls return at once) and
/// check the floor right now, so a test never starts under it. Returns the
/// floor in KiB.
pub fn arm() -> u64 {
    static ARMED: OnceLock<u64> = OnceLock::new();
    *ARMED.get_or_init(|| {
        let value = match std::env::var(FLOOR_ENV) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => panic!("{FLOOR_ENV}: {e}"),
        };
        let floor = floor_kib_from(value.as_deref()).unwrap_or_else(|e| panic!("memory guard: {e}"));
        let now = read_mem_available_kib();
        if now < floor {
            abort_under_floor(now, floor);
        }
        watch(floor, POLL, read_mem_available_kib, move |available| abort_under_floor(available, floor));
        floor
    })
}
