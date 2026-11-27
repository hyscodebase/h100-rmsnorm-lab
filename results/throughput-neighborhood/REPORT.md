# RMSNorm throughput-scale width neighborhood

Issue [#62](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/62). This 32,768-row BF16 experiment checks whether the two-row/four-warp win near width 4,104 persists at a throughput-relevant workload.

Runtime: NVIDIA H100 80GB HBM3 (CC [9, 0]); CUDA 12.8; PyTorch 2.7.1+cu128; Triton 3.3.1; driver 550.144.03.

The candidate is Triton four warps/two rows. Per seed, the one-row comparator is the faster of the four-warp and eight-warp kernels; the second reference is shape-specialized PyTorch. All checkpoints passed both recorded numerical gates before timing.

| Width | Candidate µs (47 / 48 / 49) | Fastest one-row µs | Torch compile µs | Gain vs one-row | Gain vs torch compile | ≥5% in all seeds vs both? |
|---:|---:|---:|---:|---:|---:|:---:|
| 4103 | 226.576 / 226.600 / 226.624 | 233.784 / 233.792 / 233.600 | 280.256 / 280.224 / 280.064 | 3.083 / 3.076 / 2.986% | 19.154 / 19.136 / 19.081% | No |
| 4104 | 183.272 / 183.216 / 183.304 | 181.072 / 181.008 / 181.136 | 187.632 / 188.128 / 187.592 | -1.215 / -1.220 / -1.197% | 2.324 / 2.611 / 2.286% | No |
| 4105 | 227.992 / 228.056 / 228.152 | 232.328 / 232.456 / 232.560 | 280.120 / 280.224 / 280.352 | 1.866 / 1.893 / 1.895% | 18.609 / 18.617 / 18.619% | No |
| 4112 | 183.424 / 183.536 / 183.432 | 181.328 / 181.336 / 181.256 | 188.224 / 187.792 / 187.736 | -1.156 / -1.213 / -1.201% | 2.550 / 2.266 / 2.293% | No |

**Memory preflight:** one maximum-width BF16 output is 269,484,032 bytes; conservative graph-output pool is 4,311,744,512 bytes (4.02 GiB), below the 8 GiB ceiling. Every checkpoint records allocated/reserved/free/total memory before and after each capture. Largest full-checkpoint allocated/reserved peaks were 4,043,317,760/4,315,938,816 bytes.

**H100 use:** 76,992 kernel invocations across 12 checkpoints; CUDA event intervals total 16.67s and host runner wall totals 90.9s. The observer recorded 125 samples over 130.1s summed checkpoint spans; median GPU busy 0.0%, memory busy 0.0%, VRAM 3262.0 MiB, power 120.77W. Telemetry gaps and per-checkpoint samples are archived; no saturation claim is inferred from sparse samples.

Attempt receipts: 12 total, 0 failed. Model: GPT-6 Luna (delegated runtime).

Sources were frozen at 0a7efc0137227452b8a3ebee5b25b7530e64fc4d. Raw data and telemetry are in [raw/](raw); audited derivations are in [summary.json](raw/summary.json).

The existing 1,024-row width 4,104 recommendation remains outside this throughput experiment's decision.
