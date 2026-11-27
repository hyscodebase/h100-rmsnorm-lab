# RMSNorm width neighborhood on H100

Issue [#60](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/60). Rows: 1,024 BF16. All checkpoints passed the strict FP32 gate (`rtol=0.005`, `atol=1e-6`) and width-sweep cross-check (`rtol=0.016`, `atol=1e-5`). The candidate is fixed at two rows/program and four warps.

Runtime: NVIDIA H100 80GB HBM3 (compute capability [9, 0]); CUDA 12.8; PyTorch 2.7.1+cu128; Triton 3.3.1; driver 550.144.03; Python 3.12.16.

Each cell below shows the 44/45/46 seed gains. The one-row comparator is chosen as the faster of the four-warp and eight-warp variants within each same-seed checkpoint.

| Width | Candidate µs (44 / 45 / 46) | Fastest one-row µs | Torch compile µs | Gain vs one-row | Gain vs torch compile | ≥5% in all seeds vs both? |
|---:|---:|---:|---:|---:|---:|:---:|
| 4103 | 7.740 / 7.669 / 7.670 | 9.479 / 9.403 / 9.406 | 11.501 / 11.426 / 11.427 | 18.346 / 18.441 / 18.456% | 32.702 / 32.881 / 32.878% | Yes |
| 4104 | 4.987 / 4.989 / 5.009 | 5.268 / 5.278 / 5.304 | 5.580 / 5.588 / 5.594 | 5.334 / 5.476 / 5.562% | 10.627 / 10.719 / 10.458% | Yes |
| 4105 | 7.754 / 7.801 / 7.769 | 9.472 / 9.503 / 9.493 | 11.426 / 11.472 / 11.450 | 18.138 / 17.910 / 18.161% | 32.137 / 32.000 / 32.148% | Yes |
| 4112 | 4.950 / 5.005 / 4.953 | 5.203 / 5.259 / 5.216 | 5.553 / 5.600 / 5.542 | 4.863 / 4.830 / 5.042% | 10.859 / 10.625 / 10.628% | No |

**Decision:** Retain the width 4,104 recommendation under the preregistered rule; all three seeds must beat both references by at least 5%. Neighbor results do not broaden the recommendation.

**H100 observations:** 86 one-second device samples from 12 checkpoint windows; summed first-to-last sample spans 91.5s. Median GPU busy 0.0%; median memory busy 0.0%; median VRAM 824.0 MiB; median power 119.1 W; largest within-checkpoint sample gap 5.657s. The largest gap across checkpoints was 91.071s, so the aggregate 0% busy medians do not measure continuous utilization. The 1,008 CUDA graph event intervals cover 0.24s in total; host runner wall across 12 checkpoints was 67.2s. These observer samples are not billed GPU time.

The raw archive contains 13 attempt receipts, including 1 failed connection attempt, followed by a successful retry.

The experiment runner and analysis were prepared with GPT-6 Luna (delegated runtime).

Next: retest widths 4,103, 4,104, 4,105, and 4,112 at 32,768 rows with seeds 47–49 and a 4.02 GiB graph-output estimate under the registered [issue #62](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/62).

Raw archive: [`raw/`](raw). Sources were frozen at `54c7d426a658d989aedbf442b24b9ba862fb1e2b`; checkpoint code revisions `113a26e, 54c7d42, bb2c59d, be4168d, d314d5e` are result-only commits whose archived source hashes audit to the same frozen code. The machine-readable derivation is [`summary.json`](raw/summary.json).
