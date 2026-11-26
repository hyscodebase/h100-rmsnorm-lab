# H100 kernel optimization sprint

**Question.** At 1024 token rows and hidden width 8192, can grouping multiple rows in a Triton program improve BF16 RMSNorm latency over the current one-row implementation and shape-specialized PyTorch?

| Stage | Experiment | Evidence |
| --- | --- | --- |
| 1. Establish the baseline | Compare eager PyTorch, compiled PyTorch and one-row Triton using four and eight warps. | Two recorded H100 runs, 8 shapes, 16 numerical checks. |
| 2. Change the row tile | Implement a multi-row variant. Measure the two-row and four-row configurations on the target shape and smaller controls. Include a partial final tile and a non-power-of-two width in the numerical checks. | Source, correctness output, and per-replay times for the existing and new kernels. |
| 3. Decide what to keep | Repeat the comparison in two processes. Record the winning and losing shapes and the strongest baseline at each. | A short report that states the latency delta and a decision for the target shape. |

A configuration qualifies for adoption on the target shape if its median latency is at least 5% lower than **both** the best existing one-row variant and compiled PyTorch in each of two runs, with all numerical checks passing. This is an engineering decision threshold. The sprint can also conclude by documenting an unsuccessful configuration and retaining the baseline.

Use synthetic data and one H100. A prospective FlashInfer contribution needs the same improvement reproduced against its current implementation and tests. Upstream review and merge are separate milestones.
