# Try one vector per physical lane at 128 token rows

The split-warp implementation matches the released vLLM 0.10.0 operator at 128 × 4,096. Its eight physical warps each process two eight-element groups per lane. The implementation retains two independent warp reductions and adds the 16 warp totals in order. Its latency fell short of the registered 5% gain. [Completed 128-row experiment](SMALL_ROWS_REPORT.md)

Use 16 physical warps with one eight-element group per lane. Preserve the 16 logical warp totals and their ordered sum, the BF16 rounding steps, vector width, input buffers and vLLM extension. Verify the actual Triton launch, lane mapping and generated instructions before comparing it with the existing eight-warp candidate. The working hypothesis is that each lane's shorter load and square-sum path lowers latency at 128 rows.

Use a standalone numerical checkpoint at seed 133. Verify all eight fixtures and all 32 captured input pairs for both Triton implementations against released vLLM. Require exact BF16 bits for the stored residual and normalization, the required signed-zero and weight-rounding probes, and an unchanged weight vector.

If the numerical gate passes, measure one checkpoint at a time on fresh seeds 134–136. Restore both buffers before each event and use 21 randomized graph replays with 32 in-place calls per graph. Use the same input pairs for all three backends and recheck every timed input. Require at least 5% lower median latency than both vLLM and the eight-warp candidate in each seed. Archive compiler, numerical, timing and device-observer records for each submission.
