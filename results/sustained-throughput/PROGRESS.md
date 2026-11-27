# Sustained H100 Throughput: Progress

This is an interim record. The registered study has 12 checkpoints; 9 are complete. No final kernel-adoption decision is supported yet.

## Frozen experiment

- Source pin: `1f4dc3c38f542b7aef8f5374368f71d7df6b7fcf`
- Shape: 32,768 rows of BF16 residual RMSNorm, widths 4,103 / 4,104 / 4,105 / 4,112
- Seeds: 50, 51, 52
- Candidate: Triton 4 warps / 4 rows
- References: Triton 4 warps / 1 row, Triton 8 warps / 1 row, and `torch.compile`
- Each checkpoint records 1,604 randomized latency intervals and four 33-second sustained phases.
- Adoption requires at least a 5% throughput gain over both the fastest one-row Triton reference and `torch.compile` for every seed and width.

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

The candidate has cleared the registered margin for width 4,103 in all three seeds so far, and for width 4,105 in seeds 50 and 51. At widths 4,104 and 4,112 it did not meet the margin in seeds 50 and 51. This is a repeatable width-dependent result; seed 52 still needs widths 4,104, 4,105, and 4,112. The current evidence does not support broad kernel adoption.

## H100 activity readings

The sustained phases sampled median GPU busy at 100% and roughly 5.58 GiB of GPU memory in use. All required telemetry fields were present, but only 3 of 36 phases pass the 1.5-second maximum query-gap rule when I include the start-to-first-sample and last-sample-to-end gaps from the full raw records. The current audit marks one additional phase as eligible even though its final sample is 3.498 seconds before phase end. The audit code currently checks internal gaps but not phase boundaries; that review finding will be fixed before final reporting. Activity summaries from the other 33 phases are retained as raw observations and are not treated as eligible utilization evidence.

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

The raw records contain the per-operation latency samples, event windows, correctness checks, memory snapshots, telemetry rows, source hashes, and attempt receipts. The partial audit passed with 9 of 12 checkpoints complete.
