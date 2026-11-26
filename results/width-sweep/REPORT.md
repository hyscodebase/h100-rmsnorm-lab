# Row tiling across hidden widths

At 1,024 rows, two rows per program showed a small gain at width 4,096 and a loss at width 8,192 in the preceding experiment. This sweep keeps the number of rows fixed and changes hidden width across the 4,096-column block boundary. [Earlier experiment](../row-tiles/REPORT.md) · [Plan recorded before the sweep](../../plans/width-sweep.json)

For each width, evaluate the same candidate against the best one-row Triton configuration and shape-specialized PyTorch within each seed. Adoption requires at least 5% lower median latency than both references in both seeds. Widths are separate comparisons. Each measurement runs in its own process and uses 21 CUDA-graph replay samples of 32 operations with backend order shuffled each round. The numerical check runs before timing.

To reproduce the sweep on a CUDA system with the repository dependencies installed, run the following from the checkout.

```bash
python3 lab.py init --plan plans/width-sweep.json --state-dir .lab/width-sweep
python3 lab.py step --state-dir .lab/width-sweep
```

Repeat the second command for the remaining checkpoints. The runner saves progress after each completed unit.
