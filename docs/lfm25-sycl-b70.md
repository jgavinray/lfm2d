# SYCL on the Intel Arc Pro B70: what the decode-path kernels cost, and what moved

2026-10-02, `hyper03`, oneAPI 2026.1, candle fork `jgavinray/lfm25-sycl-xe`
`e112b15c`. First measured record for this device; before it the SYCL path had
correctness evidence only. Structured numbers:
[2026-10-02-sycl-b70.json](results/2026-10-02-sycl-b70.json).

## Device

Intel Arc Pro B70: 256 Xe2 cores, 31 GiB, sub-groups 16 and 32, max workgroup
1024. `sycl-ls` and the kernels-crate smoke test both see it; fp16 and fp64
report supported. Peak measured device bandwidth (q4_0 4096×4096 mat-vec,
`mmvq_speed`): **686 GB/s**.

## What the engine's decode actually pays

Per mat-vec call on the LFM2.5-8B-A1B decode shapes (hidden 2048, MoE
intermediate 1792, 32 experts, top-4), the old path issued 3 launches
(quantize → dot → reduce) plus 3 pooled scratch allocations. Measured:

| shape | device ms | enqueue µs |
|---|---:|---:|
| dense gate_up 14336×2048 Q5_K | 0.096 | ~22 |
| dense down 2048×7168 Q5_K | 0.056 | ~21 |
| lm_head 128256×2048 Q6_K | 0.914 | ~20 |
| routed gate_up 3584×2048 Q5_K top4 | 0.092 | ~20 |
| routed down 2048×1792 Q5_K top4 | 0.058 | ~20 |

Device time dominates everywhere — even the smallest shape (56 µs) is well
above the 20 µs enqueue. That fact decided the outcome below.

## What was tried

1. **Sub-group fused geometry** (one SG16 per row, two rows per SG when the
   block count ≤ 8, xor-shuffle reduce, single launch, no `tmp` round-trip).
   Device-equal on the routed shapes, **+6–8% slower** on the two dense shapes
   (which are the most device-bound), **−32% enqueue**. Net loss for the
   engine; rejected. The geometry is kept in this record's git history, not in
   the tree.
2. **Persistent mat-vec scratch** (kept): q8/d8/s32/tmp live in a per-queue
   cache keyed `(m, n, k, blk)`, handed out as non-owning views. Saves the
   pooled alloc round trip (~1.3 µs/call host time; ~46 mat-vecs/token).
   Device times unchanged within noise.
3. **Parallel `build.rs`** (kept): the 13 kernel translation units now compile
   concurrently instead of one blocking `icpx` at a time.

Two dead ends worth remembering: a serial one-work-item-per-row dot starves
the GPU (2048 work-items on 256 CUs → 3× slower on dense down), and
`permute_group_by_xor` with mask 8 crashes IGC 2026.1's JIT (SIGSEGV in
`libigc.so.2`); masks ≤ 4 are safe.

## Verification (all green on the B70)

- kernels-crate smoke (9), `bench_decode::q8_0_matches_reference`
  (independent hand-decoded Q8_0 reference over 1792×2048).
- candle-core `quantized_tests` sycl arm (26; QMatMul vs CPU, all dtypes),
  `indexed_moe_tests` sycl arm.
- lfm2d `device_real::all_heads_agree_between_cpu_and_explicit_gpu` with
  `LFM2D_TEST_GPU=sycl`: every head CPU vs B70, cosine > 0.999, PII spans
  exact.

## What this changes

The decode mat-vec is at the practical roofline on this device; kernel
geometry is not the lever. If decode latency on the B70 needs to come down,
the next work is launch batching across layers (host-side), not kernel
arithmetic. The harness (`tests/bench_decode.rs`) and the
`enqueue`-beside-device-time output are the instruments for that round.

## Update 2026-10-02, later: grouped MoE prefill (the prefill win, landed)

The prefill-side lever the first round pointed at is now implemented and
merged (`jgavinray/lfm25-sycl-xe` `1775caf4`): at `batch*topk >= 64` tasks the
indexed MoE call dequantizes the expert stack to f16 once (cached on the
storage), gathers task rows, runs one f16 GEMM per expert and scatters back —
**before** the row-per-task expansion, which alone cost 3.5 s of d2d copies at
batch 512.

Measured, gate_up 3584×2048 Q5_K, batch 512, B70: **29.8 ms (integer mat-vec)
→ 3.2 ms warm-cache / 2.9–3.0 ms cold**, ~9×. Correctness: the full
indexed_moe battery plus a stage-decomposition test; the f16 GEMM carries
k-scaled error (0.6·√k tolerance, dtype-gated), reviewed by DeepSeek v4.1
(dispatch-before-expansion, input_dim1 guard, gather mapping, tolerance gate)
and an internal agent pass. Both reviews' findings are applied in `1775caf4`.
