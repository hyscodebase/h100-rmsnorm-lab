# 4,104-width neighborhood revalidation

This is the preregistered follow-up for [issue #60](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/60). At 1,024 BF16 rows, test widths 4,103, 4,104, 4,105, and 4,112 with new seeds 44, 45, and 46. The hypothesis is that the prior 4,104-only win may depend on exact width and alignment.

The fixed candidate is two rows per Triton program with four warps. Each shape and seed compares it with both one-row Triton kernels (four and eight warps; the faster per seed is the one-row reference) and shape-specialized `torch.compile` RMSNorm. All implementations use the same input and weight tensors, BF16 I/O, FP32 reductions, epsilon `1e-6`, and cached addresses without explicit L2 flushing.

Each checkpoint performs eight numerical fixtures before timing. Triton kernels must pass the repository's stricter FP32-reference gate (`rtol=0.005`, `atol=1e-6`); all backends must also pass the prior width-sweep cross-check (`rtol=0.016`, `atol=1e-5`). These two decisions are recorded separately. The runner then captures a CUDA graph of 32 calls per backend and measures 21 graph replays in randomized backend order. Results retain each raw event interval, backend order, fixture comparison, source hash, device/software metadata, input/weight fingerprints, and host-side execution timing. The local step runner checkpoints one shape/seed at a time and journals failed attempts. GPU telemetry is sampled around each remote invocation and copied into the result archive.

Only the 4,104 recommendation is eligible to continue: the fixed candidate must be at least 5% faster than both the fastest one-row kernel and `torch.compile` in every fresh seed. Neighbor widths are reported individually; they cannot expand the recommendation. A result below the threshold rejects the recommendation for this set. No claim is made about end-to-end LLM prefill performance.

The full frozen plan is `plans/width-neighborhood.json`. `python3 experiments/width-neighborhood/summarize.py` verifies checkpoint identities, provenance, raw intervals, exact seed/shape coverage, and the decision threshold before regenerating `REPORT.md`.
