# Grouped expert prefill on ROCm

The second optimization pass groups routed tokens by expert and reuses
Candle's existing quantized matrix-matrix tiles. On this machine, cold
adjudicator prefill fell from about 1.24 s to 0.57 s (median 2.19× speedup).
Decode still uses the previous vector kernels. No profiler was run.

## Hardware integration

The implementation follows the column-grouping pattern in llama.cpp's
[expert routing helper](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmid.cu)
and [quantized matrix dispatch](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu).
A GPU kernel packs routed pairs by expert. The existing MMQ tiles load the
corresponding q8_1 activation columns and scatter results back to their
original token/slot positions. Duplicate routes are retained; empty experts
exit before shared-memory barriers. Expert weights remain packed Q5K/Q6K.

Dispatch uses grouped MMQ when there are at least eight routed pairs per
expert on average: 64 tokens for LFM2.5's 32 experts and top-4 routing.
The daemon's 128-token prefill chunks qualify. Decode and shorter suffixes
retain indexed matrix-vector multiplication; other quantization formats
retain their existing paths. No model or snapshot interface changed.

The host reads only expert counts and an invalid-ID status (132 bytes per
projection for this model). That synchronization validates IDs before weight
access, checks every routed pair is accounted for, and bounds the launch to
the busiest expert. Routing scratch uses `experts × routed_pairs` indices;
it is temporary, and no snapshot storage is mutated. Removing metadata
readback and reducing scratch/zeroing are later refinements.

Q5 needs a deliberate arithmetic choice. Dense MMQ uses the saved original
activation sum for its minimum correction; the existing indexed vector
kernel recomputes the sum from quantized activation bytes. Grouped Q5 uses
the latter, preserving MoE's existing convention. A compile-time parameter
leaves dense MMQ unchanged. An isolated minimum-only test covers varying
block minima and positive, negative, and zero activation sums.

## Measurements and smoke checks

The [paired artifact](../benchmarks/lfm25/results/2026-09-13-grouped-prefill.json)
records binary hashes, all twelve requests, and numerical comparisons.
These are shared-host wall-clock measurements, not isolated GPU profiling.

| Case | Before cold prefill | After cold prefill | Speedup |
|---|---:|---:|---:|
| reset --hard | 1,239.6 ms | 566.7 ms | 2.19× |
| git restore | 1,261.3 ms | 575.4 ms | 2.19× |
| rm -i | 1,243.6 ms | 559.7 ms | 2.22× |
| sed read | 1,231.8 ms | 570.7 ms | 2.16× |

A separate projection sweep includes activation quantization, allocation,
routing, and synchronization. It alternates paths, excludes warmup, and
uses six samples per path:

| Projection | 32 tokens | 64 tokens | 128 tokens |
|---|---:|---:|---:|
| Q5K merged gate/up, 3584×2048 | 0.72× | 1.39× | 2.79× |
| Q6K down, 2048×1792 | 0.66× | 1.17× | 2.18× |

The loss at 32 tokens supports the small-batch vector dispatch. This pass
makes no decode-speed claim: generated lengths changed, so total decode
times are not directly comparable.

Hardware correctness checks passed:

- Five grouped tests cover an independent CPU reference, tight old-vector
  parity, repeatability, public dispatch, partial tiles, padded K strides,
  nonzero offsets, duplicate routes, unused/skewed experts, invalid IDs,
  invalid layouts, stream identity, and isolated Q5 minimum arithmetic.
- Eight existing dense-MMQ tests pass, including all supported dtypes and
  host/kernel tile-geometry agreement. Five LFM2 model tests pass, including
  exact merged/separate projection agreement and transactional state checks.
- A temporary full-checkpoint witness compared 264 projections using the
  same real activations in both paths. Worst maximum absolute error divided
  by the reference maximum was 5.59e-7, below the 2e-5 bound. The diagnostic
  was removed after validation; no duplicate computation remains in serving.
- Cached repeats match exactly. Deadline cancellation, reuse afterward,
  and in-flight SIGTERM complete successfully.

## CUDA (2026-09-28)

The kernels live in the shared `quantized.cu`, so CUDA had them in its PTX
all along; only the host side was ROCm-only. Fork commit `cca9bde7`
(`lfm25-cuda-grouped`) ports `rocm/moe.rs`'s grouped path to
`quantized/cuda/moe.rs` with the same dispatch rule (Q5K/Q6K, at least eight
routed pairs per expert), the same q8_1 activations and Q5 minimum, and the
AMPERE tile set nvcc compiles. The LFM2 model packs once per layer and
shares it across gate/up and down on both backends.

On the DGX Spark (GB10, sm_121) the fork's tests check grouped against the
CPU and against single-route matvecs within 2e-5 of the row scale (partial
tiles, padded K, duplicate routes, a hot expert, nonzero offsets), invalid
IDs, and prepared-routing validation; the lfm2d real 8B suite passes on it.
The port also fixed the CUDA matvec reading input and IDs from the start of
their storage regardless of layout offset. Speed was not measured: vLLM was
sharing the GPU.

## Prompt/model observations

Amy clarified the acceptance boundary: “we don't really know if our prompt
is any good” and wants acceleration to enable faster joint prompt/model
experiments. Fixture severity labels are recorded observations, not hardware
acceptance criteria. `git reset --hard` is context dependent.

Sampling remains greedy (temperature-zero equivalent), with repetition
penalty 1.05 over the full prompt and generated history. Grouped arithmetic
changes floating-point accumulation order. Cold and cached chunk schedules
now produce different long greedy generations; cached repeats remain exact.
Do not treat cold/cache text equality as a guarantee across kernel choices.

Ten of twelve optimized responses passed report-schema validation. Two
cached/repeated sed responses echoed schema structure instead of the report
fields; the daemon returned an explicit report error. Fixture grades were
7/12 versus 12/12 before. Those facts belong in prompt/model tuning; no schema
validation was relaxed and no report repair was added to hide them.

## Review and next work

Two Kaibo DeepSeek / deepseek-flash reviews checked routing, memory access,
and the Q5 arithmetic. Their full responses, Amy's direction, and our
resolution notes are in the author's private notes (not published). Focused ROCm lib/test
lint passes with `unnecessary_cast` allowed for two pre-existing core casts;
unrestricted lint reports only those existing casts.

The older indexed-vector API still trusts expert IDs. LFM2's arg-sort router
produces valid IDs, but bounds protection for arbitrary public-API callers
is recorded as a separate hardening task. Also recorded: rocm-rs 0.5.2's
sorting proc macros can write into the Cargo registry during compilation.

Continue with compatible QKV projection fusion and append-efficient KV
allocation while preserving snapshot ownership. Grouped routing reuse
between gate/up and down is another small hardware integration opportunity.
Profile once the larger structural changes are through.
