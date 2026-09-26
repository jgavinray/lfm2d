//! The guards every real-model test stands behind (`tests/support/`): the
//! host-memory watchdog, and the refusal to run the 8B anywhere but a GPU.
//!
//! These run in the default suite and load no weights. The abort itself is
//! tested in a child process (this same test binary, re-run with a marker
//! in its environment), because an abort ends the process that trips it.
use std::os::unix::process::ExitStatusExt as _;
use std::process::{Command, Output};
use std::sync::mpsc;
use std::time::Duration;

use clap::Parser as _;
use lfm2d::config::Cli;

mod support;
use support::memory_guard;

/// Set in a child's environment: run the child half of a guard test.
const CHILD: &str = "LFM2D_GUARD_TEST_CHILD";
const SIGABRT: i32 = 6;

/// Re-run one test of this binary as a child, with the floor set to `floor`.
fn child(test: &str, floor: &str) -> Output {
    Command::new(std::env::current_exe().expect("this test binary"))
        .args([test, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .env(memory_guard::FLOOR_ENV, floor)
        .output()
        .expect("run the child")
}

#[test]
fn meminfo_parsing_reads_mem_available_in_kib() {
    let text = "MemTotal:       131072000 kB\nMemFree:          524288 kB\n\
                MemAvailable:   99999999 kB\nBuffers:            1234 kB\n";
    assert_eq!(memory_guard::mem_available_kib(text), Some(99_999_999));
    assert_eq!(memory_guard::mem_available_kib("MemTotal: 1 kB\nMemFree: 1 kB\n"), None);
    assert_eq!(memory_guard::mem_available_kib("MemAvailable: lots kB\n"), None);
}

#[test]
fn the_floor_defaults_to_8_gib_and_refuses_nonsense() {
    assert_eq!(memory_guard::floor_kib_from(None), Ok(8 * 1024 * 1024));
    assert_eq!(memory_guard::floor_kib_from(Some("0.5")), Ok(512 * 1024));
    assert_eq!(memory_guard::floor_kib_from(Some("24")), Ok(24 * 1024 * 1024));
    for bad in ["", "0", "-1", "NaN", "inf", "eight"] {
        assert!(memory_guard::floor_kib_from(Some(bad)).is_err(), "{bad:?} must be refused");
    }
}

/// The watchdog loop itself, with a fake reader: it reports the first
/// reading under the floor, and not before.
#[test]
fn the_watchdog_fires_on_the_first_reading_under_the_floor() {
    let readings = [20u64, 20, 15, 10, 9, 20];
    let (tx, rx) = mpsc::channel();
    let mut i = 0;
    memory_guard::watch(
        10,
        Duration::from_millis(1),
        move || {
            let r = readings[i.min(readings.len() - 1)];
            i += 1;
            r
        },
        move |seen| tx.send(seen).unwrap(),
    );
    assert_eq!(rx.recv_timeout(Duration::from_secs(5)), Ok(9));
}

/// End to end, against the real `/proc/meminfo`: a floor above anything
/// the host has makes `arm` abort the process (SIGABRT, a message naming
/// the guard), and the test body after it never runs.
#[test]
fn an_unreachable_floor_aborts_the_process() {
    if std::env::var_os(CHILD).is_some() {
        memory_guard::arm();
        std::thread::sleep(Duration::from_millis(200));
        println!("guard-child: survived");
        return;
    }
    let out = child("an_unreachable_floor_aborts_the_process", "1000000");
    let (stdout, stderr) = (String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.status.signal(), Some(SIGABRT), "status {:?}\n{stdout}\n{stderr}", out.status);
    assert!(stderr.contains("memory guard"), "{stderr}");
    assert!(!stdout.contains("survived"), "{stdout}");
}

/// The other half: armed under a floor the host clears, the test runs.
#[test]
fn a_reachable_floor_lets_the_test_run() {
    if std::env::var_os(CHILD).is_some() {
        memory_guard::arm();
        std::thread::sleep(Duration::from_millis(50));
        println!("guard-child: survived");
        return;
    }
    let out = child("a_reachable_floor_lets_the_test_run", "0.001");
    let (stdout, stderr) = (String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "status {:?}\n{stdout}\n{stderr}", out.status);
    assert!(stdout.contains("guard-child: survived"), "{stdout}");
}

#[test]
fn real_model_tests_name_a_gpu_backend_never_cpu_or_auto() {
    assert_eq!(support::gpu_backend_from(None), Ok("rocm"));
    for good in ["rocm", "cuda", "metal"] {
        assert_eq!(support::gpu_backend_from(Some(good)), Ok(good));
    }
    for bad in ["cpu", "auto", "", "ROCm"] {
        assert!(support::gpu_backend_from(Some(bad)).is_err(), "{bad:?} must be refused");
    }
}

/// The loader refuses a CPU `Cli` before it touches the GGUF: the path here
/// does not exist, so any other error means the file was reached first.
#[test]
fn the_real_model_loader_refuses_a_cpu_or_auto_device_before_loading() {
    for device in ["cpu", "auto"] {
        let cli = Cli::parse_from([
            "lfm2d",
            "--device",
            device,
            "--adjudicator-model",
            "/nonexistent/lfm2d-guard-test/model.gguf",
            "--adjudicator-tokenizer",
            "/nonexistent/lfm2d-guard-test/tokenizer.json",
        ]);
        let error = support::try_load_adjudicator(&cli).err().expect("a CPU/auto device must be refused");
        assert!(error.contains("GPU"), "{device}: {error}");
        assert!(!error.contains("nonexistent"), "{device}: reached the file before refusing: {error}");
    }
}
