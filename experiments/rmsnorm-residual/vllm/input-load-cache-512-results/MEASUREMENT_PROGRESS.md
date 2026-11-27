# Input-load cache policy: completed measurements

Seed 209 passed exactness and PTX qualification. Seeds 210–212 completed both separately checkpointed periods. Every output matched bitwise, and input and weight fingerprints matched across the two backend periods in each seed.

| Seed | Order | Baseline median rows/s | Candidate median rows/s | Candidate gain |
| ---: | --- | ---: | ---: | ---: |
| 210 | Baseline then candidate | 91,547,406 | 91,548,119 | +0.000779% |
| 211 | Candidate then baseline | 91,544,022 | 91,544,924 | +0.000985% |
| 212 | Baseline then candidate | 91,535,863 | 91,542,773 | +0.007549% |

Every candidate gain was below 0.01%, far short of the 5% adoption gate. Retain the eight-warp baseline for this shape and workload.

The six fixed-work periods processed 12,884,901,888 normalized rows in 140.792 seconds, with 18.146 seconds of warm-up. All periods had 100% median GPU and memory busy, 8,431.25 MiB median VRAM, and 1,980 MHz median SM clock. The observer recorded 133 samples; median power ranged from 514.843 to 518.510 W. The largest sample gap was 2.499 seconds.

| Seed-period | Backend | Samples | GPU / memory busy median | VRAM median | Power median | Largest gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 210-p0 | Baseline | 22 | 100% / 100% | 8,431.25 MiB | 514.955 W | 1.789 s |
| 210-p1 | Candidate | 23 | 100% / 100% | 8,431.25 MiB | 515.072 W | 1.897 s |
| 211-p0 | Candidate | 21 | 100% / 100% | 8,431.25 MiB | 518.510 W | 2.154 s |
| 211-p1 | Baseline | 21 | 100% / 100% | 8,431.25 MiB | 514.882 W | 2.499 s |
| 212-p0 | Baseline | 23 | 100% / 100% | 8,431.25 MiB | 518.271 W | 1.853 s |
| 212-p1 | Candidate | 23 | 100% / 100% | 8,431.25 MiB | 514.843 W | 1.725 s |

All raw intervals, telemetry, auxiliary snapshots, and submission receipts are preserved. See the [final report](REPORT.md) and [machine-readable summary](summary.json). Regenerate them with python3 experiments/rmsnorm-residual/vllm/summarize_input_load_cache.py.

The next hypothesis is tracked in [issue #60](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/60): revalidate the 4,104-wide two-row RMSNorm setting at widths 4,103, 4,104, 4,105, and 4,112 using fresh seeds.
