# Input-load cache policy: measurement progress

Seed 210 period 0 completed using the eight-warp baseline. The period followed 3.024 seconds of warm-up and processed 2,147,483,648 normalized rows across 512 measured bursts in 23.464 seconds (**91.52 million normalized rows/second**). Captured outputs remained bitwise exact and all input and weight fingerprints were unchanged.

The telemetry observer recorded 30 device slots, with 22 samples associated with the measured baseline period. Sampled GPU and memory utilization medians were both 100%; VRAM use was 8,431.25 MiB, power was 514.95 W, and the SM clock was 1,980 MHz. The largest gap between samples was 1.789 seconds. Auxiliary snapshots before and after the period are retained in the raw checkpoint.

This is one baseline period, not a paired result. The next registered work unit is seed 210 period 1 using the `.cg` candidate. The complete raw measurements are in [s210-p0.json](s210-p0.json); the [attempt history](attempts.json) includes the submission receipt.
