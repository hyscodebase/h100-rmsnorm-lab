# Throughput-scale RMSNorm neighborhood at 32,768 rows

This plan implements the registered hypothesis in [issue #62](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/62). The preceding 1,024-row neighborhood experiment produced only 0.243 seconds of aggregate CUDA event intervals across twelve checkpoints, so this follow-up holds the nearby widths and row-tile comparison constant while increasing the batch to 32,768 rows.

## Preregistered workload

- Rows: 32,768 BF16 values per row; widths 4,103, 4,104, 4,105, and 4,112.
- Fresh seeds: 47, 48, and 49.
- Candidate: Triton, four warps, two rows/program.
- References: Triton four warps/one row, Triton eight warps/one row, and shape-specialized `torch.compile` RMSNorm. The fastest one-row Triton median is chosen separately for each width and seed.
- Numerical gate: all four backends must pass eight fixtures against FP32 at `rtol=0.005, atol=1e-6`; all must also pass the previous width-sweep cross-check at `rtol=0.016, atol=1e-5`. Both outcomes are retained separately. No timing begins if either gate fails.
- Timing: 401 rounds in randomized backend order, with four calls to each backend captured in one CUDA graph. Each shape/seed checkpoint contains 1,604 raw event intervals (6,416 RMSNorm calls), with interval duration and backend/order index retained.
- Decision: each width qualifies separately only when the two-row candidate is at least 5% faster than both the fastest one-row Triton reference and compiled PyTorch in all three seeds. The existing 1,024-row/4,104 recommendation is outside this decision.

## Memory estimate and preflight

At the maximum width, one BF16 output tensor is `32768 × 4112 × 2 = 269,484,032` bytes (0.251 GiB). Conservatively assume every captured output remains live: four backends × four output allocations per graph × 269,484,032 bytes = 4,311,744,512 bytes (4.02 GiB). This is below the preregistered 8 GiB measured capture-pool ceiling. Nominal tensor traffic per checkpoint is about 3.45 TB (6,416 calls × one input read plus one output write at the maximum width).

Before graph capture, the runner records allocated, reserved, free, and total CUDA memory. It aborts before capture if free memory is below 16 GiB. It takes the same snapshot after each backend graph is captured. After every capture, it computes the increase in maximum reserved memory from the pre-capture baseline; it aborts before timing if this conservative capture-pool proxy exceeds 8 GiB. The runner records peak allocated/reserved memory and the per-capture snapshots in the checkpoint. The preflight estimate is recomputed from the frozen JSON plan at runtime and checked against both ceilings.

The estimate intentionally counts all four output allocations per graph even if PyTorch's graph allocator reuses some storage. The 16 GiB free-memory floor provides additional room for the input, FP32 reference tensors, compiled PyTorch workspaces, and allocator fragmentation. No GPU run is part of plan preparation or freeze review.

## Execution and evidence

The runner checks both .lab/throughput-neighborhood/paused and the existing .lab/vllm/paused marker before creating an attempt and again immediately before SSH submission. Creating either marker between checkpoint invocations prevents the next GPU unit from starting. If a user requests a stop after a remote checkpoint has begun, the current bounded unit may finish and save; this one-unit runner never launches a second checkpoint in the same invocation.

One invocation of `python3 experiments/throughput-neighborhood/step.py` runs or recovers exactly one width/seed checkpoint. A local source pin fixes the frozen commit and hashes before the first remote submission. Each attempt receives a receipt; remote results are cached atomically by frozen commit, width, and seed so a lost SSH response can recover without repeating the GPU work. The tracked SSH runner reads the configured host and setup path from local Git config and embeds no credentials. The parent `h100-rmsnorm-lab` telemetry wrapper samples device utilization around each checkpoint. Raw results and telemetry live in `results/throughput-neighborhood/raw/`.

Use `python3 experiments/throughput-neighborhood/step.py --status` to check the next unit without contacting the GPU, `--pause` to set this experiment's pause marker, and `--resume` to clear only that marker.

After all twelve checkpoints, `python3 experiments/throughput-neighborhood/summarize.py` verifies the plan, coverage, numerical gates, interval count/order, memory guards, source hashes at the frozen revision, telemetry hashes, and derives each width's decision. It writes the machine-readable summary and report. The original issue #62 remains open until a separate PR is reviewed and merged.
