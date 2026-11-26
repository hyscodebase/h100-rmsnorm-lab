# Row tiling across hidden widths

At 1,024 rows, two rows per program showed a small gain at width 4,096 and a loss at width 8,192 in the preceding experiment. This sweep keeps the number of rows fixed and changes hidden width across the 4,096-column block boundary. [Earlier experiment](../row-tiles/REPORT.md) · [Plan recorded before the sweep](../../plans/width-sweep.json)

For each width, evaluate the same candidate against the best one-row Triton configuration and shape-specialized PyTorch within each seed. Adoption requires at least 5% lower median latency than both references in both seeds. Widths are separate comparisons. Each measurement runs in its own process and uses 21 CUDA-graph replay samples of 32 operations with backend order shuffled each round. The numerical check runs before timing.

## Decision

At width **4,104**, two rows per program and four warps reduce median latency by **5.77% / 5.42%** against the fastest one-row variant, and by **10.42% / 10.86%** against compiled PyTorch. Use `rmsnorm(x, weight, num_warps=4, rows_per_program=2)` for this measured H100 case. All four other tested widths fall below the decision threshold. The broad pattern does not yield a monotonic crossover. The default API retains the one-row configuration.

Each pair lists seed 42 then seed 43. Positive reductions mean lower latency. The displayed candidate maximizes its smallest reduction across both references and both seeds.

| Width | Fixed candidate | Candidate µs | Best one-row µs | Compiled PyTorch µs | Reduction vs one-row | Reduction vs compiled | All candidates passing 5% |
|---:|---|---:|---:|---:|---:|---:|---|
| 4096 | triton_4w_2r | 4.797 / 4.810 | 4.918 / 4.908 | 5.017 / 5.020 | +2.46% / +2.00% | +4.39% / +4.18% | None |
| 4104 | triton_4w_2r | 5.011 / 4.991 | 5.318 / 5.277 | 5.594 / 5.599 | +5.77% / +5.42% | +10.42% / +10.86% | triton_4w_2r |
| 5120 | triton_4w_2r | 5.795 / 5.791 | 5.829 / 5.810 | 5.924 / 5.920 | +0.58% / +0.33% | +2.18% / +2.18% | None |
| 6144 | triton_8w_2r | 6.611 / 6.597 | 6.406 / 6.387 | 6.606 / 6.634 | -3.20% / -3.29% | -0.08% / +0.56% | None |
| 8192 | triton_8w_2r | 8.626 / 8.850 | 8.372 / 8.484 | 8.467 / 8.483 | -3.03% / -4.31% | -1.88% / -4.33% | None |

The input grows by eight BF16 elements per row between 4,096 and 4,104. The reduction block also grows from 4,096 to 8,192 lanes, and row alignment changes. The timing alone cannot distinguish their contributions. An alignment sweep at 4,103/4,104/4,105 and 4,112 with fresh seeds would test how local this result is. This microbenchmark covers cache-reused forward RMSNorm. It does not measure overall LLM prefill performance.

All 60 numerical checks passed. Ten independently launched shape/seed checkpoints each contain eight backends and 21 replay samples per backend, for 1,680 raw samples. The [checkpoint archive](raw) includes the local experiment manifest that fixed the plan before measurement. The first two timing checkpoints predate the public plan commit; the public tracking issue and PR were opened after measurement began. The three numerical source files have identical SHA-256 hashes across the recorded Git revisions. Run `python3 results/width-sweep/summarize.py` in a clone with those revisions to verify that identity and regenerate the table.

## Reproduce

To reproduce the sweep on a CUDA system with the repository dependencies installed, run the following from the checkout.

```bash
python3 lab.py init --plan plans/width-sweep.json --state-dir .lab/width-sweep
python3 lab.py step --state-dir .lab/width-sweep
```

Repeat the second command for the remaining checkpoints. The runner saves progress after each completed unit.
