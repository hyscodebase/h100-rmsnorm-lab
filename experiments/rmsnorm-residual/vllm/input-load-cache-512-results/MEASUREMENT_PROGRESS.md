# Input-load cache policy: measurement progress

Seed 210 completed both separately checkpointed backend periods. Each period processed 2,147,483,648 normalized rows across 512 measured bursts after at least three seconds of warm-up. Captured outputs remained bitwise exact, both inputs and weights stayed unchanged, and the fingerprints matched across periods.

| Backend | Median burst throughput | Period wall time | Median GPU / memory busy | Median power |
| --- | ---: | ---: | ---: | ---: |
| Eight-warp baseline | 91,547,406 normalized rows/s | 23.464 s | 100% / 100% | 514.95 W |
| Eight-warp `.cg` candidate | 91,548,119 normalized rows/s | 23.463 s | 100% / 100% | 515.07 W |

The candidate's median burst throughput was **0.000779% higher** in seed 210, below the preregistered 5% adoption threshold. The observer associated 22 device samples with the baseline period and 23 with the candidate period. The largest gap between samples was 1.789 seconds in the baseline period. Auxiliary snapshots and all raw burst intervals are retained in [s210-p0.json](s210-p0.json) and [s210-p1.json](s210-p1.json).

This seed does not support adopting `.cg`. Two more independent seeds remain in the registered order, with seed 211 measuring the candidate first. The [attempt history](attempts.json) includes both period receipts.
