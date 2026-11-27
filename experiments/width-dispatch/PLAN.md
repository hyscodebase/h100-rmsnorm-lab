# Width-aware 4-row RMSNorm dispatch screen

This is the frozen implementation plan for [issue #66](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/66), following the completed sustained study in [PR #65](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/65).

## Question and scope

At 32,768 BF16 rows, the four-row/four-warp kernel was about 12% faster than the strongest one-row Triton setting at widths 4,103 and 4,105, but about 1.1% slower at 4,104 and 4,112. This screen checks whether that split repeats among eight preregistered nearby widths. The results apply only to these shapes and this workload; they do not establish a general parity rule or a production dispatch policy.

## Frozen comparison

- Rows: 32,768; dtype: BF16; epsilon: 1e-6.
- Widths: 4,102, 4,106, 4,107, 4,108, 4,109, 4,111, 4,113 and 4,116. Four are odd and four even, with two widths in each modulo-4 class.
- Seeds: 53, 54 and 55.
- Candidate: Triton four warps / four rows per program.
- References: Triton four warps / one row, Triton eight warps / one row and shape-specialized torch.compile.
- The candidate qualifies at a width only when it is at least 5% faster than both the fastest one-row result and torch.compile in each of the three paired seeds.
- Report a bounded odd/even dispatch screen only if all four registered odd widths qualify and none of the four registered even widths qualify. Otherwise publish per-width decisions without inferring a parity rule. Three seeds are a robustness screen, not a statistical confidence interval.

## Measurement protocol

Each checkpoint runs eight correctness fixtures and its timed input. All outputs must pass the strict FP32 check (rtol=0.005, atol=1e-6) and cross-check (rtol=0.016, atol=1e-5), then pass again after latency and sustained replay. The fixtures are random, zero rows, constant, tiny, large, alternating, column ramp and outlier. Every fixture is checked to remain BF16 before kernel calls. Each captured latency and sustained graph is also tested after its output buffer is poisoned with NaNs, so a partial or missing graph write cannot pass by reusing warmup output.

Each of four backends owns one persistent BF16 output tensor. A captured graph contains four writes to that backend's output. Backends run serially on one CUDA stream. After warmup, 401 randomized rounds produce 1,604 retained CUDA-event intervals. Each backend then runs 30 adjacent event windows targeting 1.1 seconds each (about 33 seconds per phase); every window duration, replay count and inter-window event gap is retained. A final synchronization and output check close each phase.

One phase-bounded nvidia-smi process samples at 250 ms. It uses the first `CUDA_VISIBLE_DEVICES` selector when present and matches both the nvidia-smi UUID and GPU name to PyTorch's active CUDA device; when the variable is unset, mapping is accepted only if nvidia-smi exposes exactly one GPU and its UUID/name match. Unverifiable mapping makes telemetry ineligible while preserving compute measurements. The separate activity-eligibility rule requires at least 25 in-window records, at least 95% finite coverage for GPU busy, memory busy, VRAM, power and temperature, and no gap above 1.5 seconds including phase boundaries. Sparse telemetry is preserved and makes activity claims ineligible without discarding otherwise valid compute measurements. Logical input-plus-output GB/s is an effective throughput measure, not measured HBM bandwidth.

Before allocation, the saved memory snapshots must show at least 16 GiB free; capture reserved-memory growth may not exceed 8 GiB. A memory snapshot is timestamped before each measured phase. The benchmark process has a 300-second hard limit, and the entire remote lock/setup/run operation has a 350-second deadline. Each checkpoint has at most three submitted attempts; a pause before SSH submission does not consume one. A saved correctness, memory, duration or compute failure blocks later checkpoints. NaN/Inf diagnostics are stored as `finite: false` with a null maximum error, so numerical failures remain serializable. Receipts and checkpoint files bind exact source hashes, source revision, identity and SHA-256; resume reconciles interrupted receipts, reruns the evidence audit, and rejects mixed-source or unbound checkpoint files before skipping work. The final audit verifies the complete fixture/backend matrix and the fixed 5% selection gate against the pinned plan. Pause and resume occur only at checkpoint boundaries.

## Work and stop rules

The matrix has 24 checkpoints. Four sustained phases per checkpoint target 3,168 seconds (52.8 minutes) of CUDA-event work; compilation, correctness, randomized latency, telemetry and checkpoint handling add wall time. Record actual GPU-event and host wall time.

Freeze and review the runner, plan and audit before submission. Once a reviewable PR has a frozen source pin, one checkpoint may run while PR review is underway. If review finds a blocking correctness or protocol issue, finish and save the current checkpoint, then pause. Stop submissions immediately when the user requests a pause or says they are moving.

## Evidence

Raw checkpoint JSON and attempt receipts go under results/width-dispatch/raw/. After each saved checkpoint, a source/receipt/evidence audit is refreshed atomically in `partial-audit.json`; a failed audit blocks the next submission. `step.py --status` reports cumulative CUDA-event minutes and saved GPU utilization, VRAM, power and temperature samples. The final audit recomputes throughputs and per-width seed decisions from raw records. The report will include every checkpoint's timings, correctness, memory and telemetry eligibility, plus the bounded dispatch screen outcome.
