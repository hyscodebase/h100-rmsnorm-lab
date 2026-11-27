# Sustained H100 Throughput: Progress

All 12 registered checkpoints are complete. This is the final result for the frozen width/seed matrix.

## Frozen experiment

- Source pin: `1f4dc3c38f542b7aef8f5374368f71d7df6b7fcf`
- Shape: 32,768 rows of BF16 residual RMSNorm, widths 4,103 / 4,104 / 4,105 / 4,112
- Seeds: 50, 51, 52
- Candidate: Triton 4 warps / 4 rows
- References: Triton 4 warps / 1 row, Triton 8 warps / 1 row, and `torch.compile`
- Each checkpoint records 1,604 randomized latency intervals and four 33-second sustained phases.
- Adoption requires at least a 5% throughput gain over both the fastest one-row Triton reference and `torch.compile` for every seed and width.
- Review fixes hardened the audit after the measurements; all measured checkpoints remain tied to the source pin above. Any follow-up H100 experiment needs its own frozen source revision.

## Completed checkpoints

Rates below count logical input plus output bytes. They are not measured HBM bandwidth.

| Seed / width | Candidate rate (GB/s) | Fastest one-row rate (GB/s) | `torch.compile` (GB/s) | Candidate gain vs one-row | Candidate gain vs compile |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 / 4,103 | 2,619.07 | 2,314.81 | 1,870.55 | +13.14% | +40.02% |
| 50 / 4,104 | 2,963.34 | 2,995.99 | 2,874.49 | -1.09% | +3.09% |
| 50 / 4,105 | 2,609.47 | 2,331.04 | 1,870.57 | +11.94% | +39.50% |
| 50 / 4,112 | 2,968.20 | 2,995.34 | 2,892.44 | -0.91% | +2.62% |
| 51 / 4,103 | 2,624.50 | 2,328.10 | 1,877.28 | +12.73% | +39.80% |
| 51 / 4,104 | 2,962.78 | 2,995.01 | 2,884.34 | -1.08% | +2.72% |
| 51 / 4,105 | 2,606.24 | 2,330.23 | 1,870.86 | +11.84% | +39.31% |
| 51 / 4,112 | 2,963.34 | 3,001.17 | 2,889.83 | -1.26% | +2.54% |
| 52 / 4,103 | 2,621.67 | 2,322.77 | 1,874.40 | +12.87% | +39.87% |
| 52 / 4,104 | 2,962.82 | 2,992.48 | 2,881.96 | -0.99% | +2.81% |
| 52 / 4,105 | 2,607.65 | 2,330.51 | 1,871.54 | +11.89% | +39.33% |
| 52 / 4,112 | 2,967.71 | 3,000.64 | 2,886.52 | -1.10% | +2.62% |

## Final decision

The 4-row candidate cleared the registered 5% margin over both references for widths 4,103 and 4,105 in all three seeds. At widths 4,104 and 4,112 it missed that margin in all three seeds, averaging about 1.1% slower than the fastest one-row Triton backend and only 2.6–2.8% faster than `torch.compile`. The preregistered rule therefore rejects broad adoption of the 4-row candidate.

| Width | Candidate median (GB/s) | Fastest one-row median (GB/s) | `torch.compile` median (GB/s) | Gain vs one-row | Gain vs compile | Seeds passing |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,103 | 2,621.67 | 2,322.77 | 1,874.40 | +12.87% | +39.87% | 3/3 |
| 4,104 | 2,962.82 | 2,995.01 | 2,881.96 | -1.08% | +2.81% | 0/3 |
| 4,105 | 2,607.65 | 2,330.51 | 1,870.86 | +11.89% | +39.33% | 3/3 |
| 4,112 | 2,967.71 | 3,000.64 | 2,889.83 | -1.10% | +2.62% | 0/3 |

The stable width split is a useful result: width-aware dispatch is the next hypothesis to test, routing 4,103 and 4,105 to the 4-row candidate while retaining the fastest one-row backend for 4,104 and 4,112. That follow-up should include nearby unseen widths and a representative production shape mix to check for overfitting.

## H100 activity readings

The 48 sustained phases accumulated 1,588.37 seconds (26.47 minutes) of CUDA-event time. Raw sampler medians across 1,449 samples per field were 100% GPU busy, 100% memory busy, 5,578 MiB in use, 574.66 W, and 46°C. These aggregate readings are descriptive; most phases do not meet the continuous-telemetry gate.

Including phase-start and phase-end gaps, only 3 of 48 phases pass the registered 1.5-second maximum query-gap rule. The boundary-aware audit flags one stale recorded value: seed 52 / width 4,103's 8-warps / 1-row phase was stored as eligible, but its final sample is 3.498 seconds before phase end, so the corrected audit marks it ineligible. This telemetry mismatch does not invalidate the compute result. The three qualifying phases are width 4,104's candidate and 8-warps / 1-row reference, plus width 4,112's 4-warps / 1-row reference. Full telemetry rows and their gaps remain in every raw checkpoint.

Eligible phases were width 4,104's candidate and 8-warps / 1-row reference, plus width 4,112's 4-warps / 1-row reference. Each recorded 100% median GPU busy and memory busy; their median power readings were 530.57 W, 583.66 W, and 549.27 W, respectively. This workload saturates compute activity while using a small fraction of H100 memory capacity.

## Raw evidence

- [Seed 50, width 4,103 checkpoint](raw/checkpoints/s50-w4103.json) and [attempt receipt](raw/attempts/s50-w4103-86346363f0fe4999817fff974a06c0e9.json)
- [Seed 50, width 4,104 checkpoint](raw/checkpoints/s50-w4104.json) and [attempt receipt](raw/attempts/s50-w4104-81f1a4a12ab1493abbd3fdaf6cb53777.json)
- [Seed 50, width 4,105 checkpoint](raw/checkpoints/s50-w4105.json) and [attempt receipt](raw/attempts/s50-w4105-f38ae20d4cb446da884b9cb8459abfcb.json)
- [Seed 50, width 4,112 checkpoint](raw/checkpoints/s50-w4112.json) and [attempt receipt](raw/attempts/s50-w4112-2c708404e9d24efca4f53523ee78b0d8.json)
- [Seed 51, width 4,103 checkpoint](raw/checkpoints/s51-w4103.json) and [attempt receipt](raw/attempts/s51-w4103-e3f1f0bc7aa34941be4632c7271703b2.json)
- [Seed 51, width 4,104 checkpoint](raw/checkpoints/s51-w4104.json) and [attempt receipt](raw/attempts/s51-w4104-65e141b42d7e499ea6f234894e133550.json)
- [Seed 51, width 4,105 checkpoint](raw/checkpoints/s51-w4105.json) and [attempt receipt](raw/attempts/s51-w4105-004550ddbab0403bb56707c89a635b73.json)
- [Seed 51, width 4,112 checkpoint](raw/checkpoints/s51-w4112.json) and [attempt receipt](raw/attempts/s51-w4112-4a53cb5e21544f1b8aca677f1d1ca00a.json)
- [Seed 52, width 4,103 checkpoint](raw/checkpoints/s52-w4103.json) and [attempt receipt](raw/attempts/s52-w4103-92827351f8f6457ca95714632bc31e5e.json)
- [Seed 52, width 4,104 checkpoint](raw/checkpoints/s52-w4104.json) and [attempt receipt](raw/attempts/s52-w4104-b3604a84bcf14cee8136bce8aff4d930.json)
- [Seed 52, width 4,105 checkpoint](raw/checkpoints/s52-w4105.json) and [attempt receipt](raw/attempts/s52-w4105-7f73d73acb2e49d29ee714fd0d2b2b3f.json)
- [Seed 52, width 4,112 checkpoint](raw/checkpoints/s52-w4112.json) and [attempt receipt](raw/attempts/s52-w4112-14f3fea64d3f4d3eb26bf33ce4051fa3.json)

The raw records contain all 19,248 per-operation latency intervals, event windows, correctness checks, memory snapshots, telemetry rows, source hashes, and attempt receipts. The boundary-aware [final audit summary](raw/final-audit.json) verifies the frozen source pin, all 12 complete checkpoints, and 12 successful checkpoint receipts (16 total attempt receipts including transport and recovery history). It reports the corrected activity-telemetry eligibility separately from the original runtime flag.
