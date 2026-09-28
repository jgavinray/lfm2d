//! Choose an execution device once, before loading any checkpoint.

use clap::ValueEnum;
use lfm2_encoder::{DType, Device};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DeviceArg {
    Auto,
    Cpu,
    Rocm,
    Cuda,
    Metal,
}

impl DeviceArg {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto", Self::Cpu => "cpu", Self::Rocm => "rocm",
            Self::Cuda => "cuda", Self::Metal => "metal",
        }
    }
}

/// Startup policy separated from driver probing so CPU-only hosts still test
/// successful GPU selection, unavailable drivers, and explicit-device failure.
fn select_with(
    requested: DeviceArg,
    compiled: &[DeviceArg],
    mut initialize: impl FnMut(DeviceArg) -> Result<(), String>,
) -> Result<(DeviceArg, Vec<String>), String> {
    if requested == DeviceArg::Cpu {
        return Ok((DeviceArg::Cpu, vec![]));
    }
    if requested != DeviceArg::Auto {
        if !compiled.contains(&requested) {
            return Err(format!("{} backend not compiled; rebuild with --features {}",
                               requested.as_str(), requested.as_str()));
        }
        initialize(requested).map_err(|e| format!("{} initialization failed: {e}", requested.as_str()))?;
        return Ok((requested, vec![]));
    }
    let mut reasons = Vec::new();
    for &backend in compiled {
        match initialize(backend) {
            Ok(()) => return Ok((backend, reasons)),
            Err(e) => reasons.push(format!("{} initialization failed: {e}", backend.as_str())),
        }
    }
    if compiled.is_empty() {
        reasons.push("no GPU backend compiled; rebuild with --features rocm, cuda, or metal".into());
    }
    Ok((DeviceArg::Cpu, reasons))
}

/// For a checkpoint whose CPU path is a reference, never a fallback (the
/// LFM2.5 MoE: every expert dequantized to f32, ~29 GB, then memory and time
/// linear in tokens): `auto` landing on CPU is refused. An explicit
/// `--device cpu` is the reference run and stays allowed.
pub fn refuse_implicit_cpu(requested: DeviceArg, selected: DeviceArg, what: &str) -> Result<(), String> {
    if requested == DeviceArg::Auto && selected == DeviceArg::Cpu {
        return Err(format!(
            "--device auto resolved to CPU for {what}; its CPU path is a reference, not a fallback \
             (~29 GB of f32 experts, then memory and time linear in tokens). Fix the GPU backend, \
             or pass --device cpu explicitly for a reference run"
        ));
    }
    Ok(())
}

/// The device used by all heads. Never replaced after model loading starts.
pub struct ExecutionDevice {
    pub device: Device,
    pub backend: DeviceArg,
    pub selection_reasons: Vec<String>,
    /// What the kernels were built for, as specific as the backend can say:
    /// `rocm:gfx1151:hip7.2`, `cuda:sm_121:nvcc13.0:drv580.173.02`, or the
    /// backend's bare name where there is no more to say (cpu) or no port yet
    /// (Metal). Part of every `snapshot_id`, because numbers do not transfer
    /// between targets.
    pub identity: String,
}

impl ExecutionDevice {
    pub fn select(requested: DeviceArg, ordinal: usize) -> Result<Self, String> {
        i32::try_from(ordinal).map_err(|_| "device index exceeds the GPU driver's signed 32-bit range")?;
        let compiled = [
            #[cfg(feature = "rocm")] DeviceArg::Rocm,
            #[cfg(feature = "cuda")] DeviceArg::Cuda,
            #[cfg(feature = "metal")] DeviceArg::Metal,
        ];
        let mut initialized = None;
        let (backend, selection_reasons) = select_with(requested, &compiled, |backend| {
            initialized = Some(initialize(backend, ordinal)?);
            Ok(())
        })?;
        let device = if backend == DeviceArg::Cpu { Device::Cpu } else {
            initialized.ok_or("selected GPU without an initialized device")?
        };
        let identity = identity_of(&device, backend)?;
        Ok(Self { device, backend, selection_reasons, identity })
    }

    pub fn metadata(&self, dtype: DType) -> crate::telemetry::ExecutionMetadata {
        crate::telemetry::ExecutionMetadata {
            device_type: if self.device.is_cpu() { "cpu" } else { "gpu" }.into(),
            backend: self.backend.as_str().into(),
            // The selected device's own identity, only where it names more
            // than the backend (ROCm and CUDA today); never the host's installed GPU
            // read some other way, which need not be the one selected.
            device_name: (self.identity != self.backend.as_str()).then(|| self.identity.clone()),
            dtype: format!("{dtype:?}").to_lowercase(),
        }
    }
}

fn identity_of(device: &Device, backend: DeviceArg) -> Result<String, String> {
    match device {
        #[cfg(feature = "rocm")]
        Device::Rocm(d) => Ok(format!("rocm:{}:hip{}", d.arch(), d.hip_version())),
        #[cfg(feature = "cuda")]
        Device::Cuda(d) => {
            let capability = d.cuda_stream().context().compute_capability()
                .map_err(|e| format!("cuda identity: compute capability: {e}"))?;
            let nvcc = nvcc_release(candle_kernels::AFFINE.ptx())?;
            let proc = std::fs::read_to_string(DRIVER_VERSION_FILE)
                .map_err(|e| format!("cuda identity: {DRIVER_VERSION_FILE}: {e}"))?;
            Ok(cuda_identity(capability, &nvcc, &driver_release(&proc)?))
        }
        _ => Ok(backend.as_str().to_string()),
    }
}

/// Three things decide what a CUDA device computes: the target the kernels
/// run on, the nvcc that compiled them, and the driver, whose JIT turns
/// candle's PTX into the machine code that runs.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn cuda_identity((major, minor): (i32, i32), nvcc: &str, driver: &str) -> String {
    format!("cuda:sm_{major}{minor}:nvcc{nvcc}:drv{driver}")
}

/// The nvcc release that compiled candle's kernels, from the header it
/// writes into every PTX module (`Cuda compilation tools, release 13.0, …`):
/// what the build used, not whatever nvcc the host has now.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn nvcc_release(ptx: &str) -> Result<String, String> {
    ptx.lines()
        .find_map(|l| l.strip_prefix("// Cuda compilation tools, release "))
        .and_then(|rest| rest.split(',').next())
        .map(|v| v.trim().to_string())
        .ok_or_else(|| "cuda identity: the kernels' PTX names no nvcc release".into())
}

/// The kernel module's own statement of the installed driver release.
/// `cuDriverGetVersion` reports only the API level (13000), which many
/// driver releases, and so many PTX JITs, share.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const DRIVER_VERSION_FILE: &str = "/proc/driver/nvidia/version";

/// The release (`580.173.02`) from the `NVRM version:` line of
/// [`DRIVER_VERSION_FILE`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn driver_release(proc: &str) -> Result<String, String> {
    proc.lines()
        .find_map(|l| l.strip_prefix("NVRM version:"))
        .and_then(|rest| rest.split_whitespace().find(|w| {
            w.contains('.') && w.split('.').all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
        }))
        .map(str::to_string)
        .ok_or_else(|| format!("cuda identity: no driver release on the NVRM line of {DRIVER_VERSION_FILE}"))
}

fn initialize(backend: DeviceArg, ordinal: usize) -> Result<Device, String> {
    // Suppress only the unused-variable warning in CPU-only builds; no runtime
    // fallback lives here or in the inference worker.
    let _ = ordinal;
    match backend {
        #[cfg(feature = "rocm")]
        DeviceArg::Rocm => Device::new_rocm(ordinal).map_err(|e| e.to_string()),
        #[cfg(feature = "cuda")]
        DeviceArg::Cuda => Device::new_cuda(ordinal).map_err(|e| e.to_string()),
        #[cfg(feature = "metal")]
        DeviceArg::Metal => Device::new_metal(ordinal).map_err(|e| e.to_string()),
        _ => Err(format!("{} is not a compiled GPU backend", backend.as_str())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_ordinal_cannot_wrap_to_a_different_device() {
        let error = ExecutionDevice::select(DeviceArg::Auto, usize::MAX).err()
            .expect("invalid ordinal must fail before probing or CPU fallback");
        assert!(error.contains("device index"));
    }

    #[test]
    fn actual_cpu_selection_exports_its_backend_and_dtype() {
        let execution = ExecutionDevice::select(DeviceArg::Cpu, 0).unwrap();
        let metadata = execution.metadata(DType::F16);
        assert_eq!(metadata.device_type, "cpu");
        assert_eq!(metadata.backend, "cpu");
        assert_eq!(metadata.dtype, "f16");
        // The identity snapshot_id hashes. On CPU it says no more than the
        // backend, so telemetry gets no device_name rather than a repeat.
        assert_eq!(execution.identity, "cpu");
        assert_eq!(metadata.device_name, None);
    }

    #[test]
    fn cpu_never_probes_a_gpu() {
        let (selected, errors) = select_with(DeviceArg::Cpu, &[DeviceArg::Rocm], |_| {
            panic!("CPU selection must not probe accelerators")
        }).unwrap();
        assert_eq!(selected, DeviceArg::Cpu);
        assert!(errors.is_empty());
    }

    #[test]
    fn auto_prefers_initialized_gpu_and_records_failed_probes() {
        let (selected, errors) = select_with(DeviceArg::Auto, &[DeviceArg::Rocm, DeviceArg::Cuda], |backend| {
            if backend == DeviceArg::Rocm { Err("no AMD device".into()) } else { Ok(()) }
        }).unwrap();
        assert_eq!(selected, DeviceArg::Cuda);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("no AMD device"));
    }

    #[test]
    fn auto_falls_back_loudly_when_initialization_fails() {
        let (selected, errors) = select_with(DeviceArg::Auto, &[DeviceArg::Rocm], |_| {
            Err("driver unavailable".into())
        }).unwrap();
        assert_eq!(selected, DeviceArg::Cpu);
        assert!(errors[0].contains("driver unavailable"));
    }

    #[test]
    fn cpu_only_build_reports_why_auto_selected_cpu() {
        let (selected, errors) = select_with(DeviceArg::Auto, &[], |_| unreachable!()).unwrap();
        assert_eq!(selected, DeviceArg::Cpu);
        assert!(errors[0].contains("no GPU backend compiled"));
    }

    #[test]
    fn a_moe_checkpoint_refuses_auto_resolving_to_cpu() {
        let error = refuse_implicit_cpu(DeviceArg::Auto, DeviceArg::Cpu, "the adjudicator").unwrap_err();
        assert!(error.contains("--device cpu"), "{error}");
        assert!(error.contains("the adjudicator"), "{error}");
        // An explicit CPU is the reference path; a GPU is the point.
        assert!(refuse_implicit_cpu(DeviceArg::Cpu, DeviceArg::Cpu, "x").is_ok());
        assert!(refuse_implicit_cpu(DeviceArg::Auto, DeviceArg::Rocm, "x").is_ok());
        assert!(refuse_implicit_cpu(DeviceArg::Rocm, DeviceArg::Rocm, "x").is_ok());
    }

    // Headers as nvcc 13.0 and the 580 driver on tenchi wrote them.
    const PTX_HEADER: &str = "//\n// Generated by NVIDIA NVVM Compiler\n//\n// Compiler Build ID: CL-36424714\n\
        // Cuda compilation tools, release 13.0, V13.0.88\n// Based on NVVM 20.0.0\n//\n\n.version 9.0\n";
    const PROC_VERSION: &str = "NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  580.173.02  \
        Release Build  (dvs-builder@U22-A24-5-4)  Tue Jun 23 08:34:19 UTC 2026\nGCC version:  gcc version 13.3.0\n";

    #[test]
    fn cuda_identity_names_target_compiler_and_driver() {
        let nvcc = nvcc_release(PTX_HEADER).unwrap();
        let driver = driver_release(PROC_VERSION).unwrap();
        assert_eq!(cuda_identity((12, 1), &nvcc, &driver), "cuda:sm_121:nvcc13.0:drv580.173.02");
    }

    #[test]
    fn cuda_identity_parts_fail_rather_than_guess() {
        assert!(nvcc_release(".version 9.0\n.target sm_121f\n").is_err());
        assert!(driver_release("GCC version:  gcc version 13.3.0\n").is_err());
        assert!(driver_release("NVRM version: NVIDIA UNIX Open Kernel Module for aarch64\n").is_err());
    }

    #[test]
    fn explicit_gpu_failure_never_falls_back() {
        let error = select_with(DeviceArg::Rocm, &[DeviceArg::Rocm], |_| {
            Err("driver unavailable".into())
        }).unwrap_err();
        assert!(error.contains("driver unavailable"));
        let error = select_with(DeviceArg::Cuda, &[], |_| unreachable!()).unwrap_err();
        assert!(error.contains("not compiled"));
    }
}
