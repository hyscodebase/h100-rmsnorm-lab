# Input-load cache policy: measurement progress

## Paired throughput

Each period warmed up for at least three seconds, then recorded 512 bursts processing 2,147,483,648 normalized rows. Median burst throughput is the median of `4,194,304 / burst_wall_seconds`. All four periods passed post-period exactness checks; seed fingerprints match within both completed seeds.

| Seed | Eight-warp baseline | Eight-warp `.cg` | Candidate gain | Exact and matched |
| ---: | ---: | ---: | ---: | :---: |
| 210 | 91,547,406 rows/s | 91,548,119 rows/s | +0.000779% | Yes |
| 211 | 91,544,022 rows/s | 91,544,924 rows/s | +0.000985% | Yes |

The candidate is effectively tied with the baseline in both seeds and is far below the preregistered 5% adoption threshold. Seed 212 remains scheduled to complete the reversed comparison as an independent confirmation.

## H100 resource samples

All four measured periods had 100% median GPU and memory busy. The NVML observer associated 21–23 samples with each period; median VRAM use was 8,431.25 MiB in each. The largest sample gap was 2.499 seconds. Auxiliary snapshots and every sample are retained in the raw period files.

| Seed-period | Backend | Samples | Median power | Median SM clock | Largest gap |
| --- | --- | ---: | ---: | ---: | ---: |
| 210-p0 | Baseline | 22 | 514.95 W | 1,980 MHz | 1.789 s |
| 210-p1 | `.cg` | 23 | 515.07 W | 1,980 MHz | 1.897 s |
| 211-p0 | `.cg` | 21 | 518.51 W | 1,980 MHz | 2.154 s |
| 211-p1 | Baseline | 21 | 514.88 W | 1,980 MHz | 2.499 s |

The complete measurements and auxiliary snapshots are in [s210-p0.json](s210-p0.json), [s210-p1.json](s210-p1.json), [s211-p0.json](s211-p0.json), and [s211-p1.json](s211-p1.json). [attempts.json](attempts.json) retains all sanitized submission and recovery receipts.

## Next step

Run seed 212 in the preregistered baseline-then-candidate order. The final report will derive all three gains directly from the saved burst intervals and record whether the `.cg` candidate meets the adoption rule.
