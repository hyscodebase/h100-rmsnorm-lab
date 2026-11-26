# H100 RMSNorm Lab

How much does the thread layout of a BF16 RMSNorm kernel matter on an NVIDIA H100? This inference microbenchmark varies the number of token rows while keeping the math and inputs fixed. The initial experiment compares a single-row Triton kernel with four or eight warps against eager PyTorch and shape-specialized TorchInductor.

![Speed relative to compiled PyTorch at four row counts and two hidden widths](results/warp-comparison.png)

## Findings

At hidden width 4096 and one token row, the eight-warp kernel took **1.82–1.83 µs** across two runs. Compiled PyTorch took **2.11–2.12 µs**, giving the Triton kernel about **14% lower latency**. At 1,024 rows and width 8192, compiled PyTorch led both custom kernels in both runs. The difference between four and eight warps changes sign as the number of rows grows.

Each cell below contains the first run's median GPU time in microseconds. Each timing sample averages 32 consecutive invocations in a CUDA graph. The raw files retain all 21 samples per implementation and shape.

| Token rows | Hidden width | PyTorch eager | PyTorch compiled | Triton 4 warps | Triton 8 warps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 16.19 | 2.11 | 1.88 | **1.82** |
| 16 | 4096 | 24.57 | 2.20 | 1.94 | **1.89** |
| 128 | 4096 | 27.45 | 2.44 | 2.11 | **2.07** |
| 1024 | 4096 | 71.85 | 5.00 | **4.92** | 5.20 |
| 1 | 8192 | 16.75 | 2.55 | 2.19 | **2.13** |
| 16 | 8192 | 19.13 | 2.68 | 2.28 | **2.26** |
| 128 | 8192 | 26.58 | 3.04 | 2.60 | **2.58** |
| 1024 | 8192 | 144.21 | **8.36** | 8.49 | 8.96 |

[First run](results/h100.json) · [Repeat run](results/h100-repeat.json) · [Numerical checks](results/correctness.json)

## Reproduce on a GPU

The recorded runs used Linux, an H100 80GB HBM3, Python 3.12.16, PyTorch 2.7.1+cu128, Triton 3.3.1 and NVIDIA driver 550.144.03. A C compiler and a linkable CUDA driver library are required to build the Triton launchers. The compiled PyTorch baseline can also use a C++ toolchain.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
python check.py
BENCH_REVISION=$(git rev-parse HEAD) python benchmark.py --compiled --output run.json
```

For a single shape, add `--rows 1 --widths 4096`. To regenerate the figure, install Matplotlib in a local plotting environment and run `python plot_results.py`.

## Kernel and measurement

- The baseline Triton program loads one contiguous row and the shared per-column weight. It accumulates the sum of squares in FP32, applies reciprocal RMS with epsilon 1e-6, multiplies by the weight and stores BF16 output. Masked loads support widths up to 16384. [Kernel](kernels.py)
- The initial eight numerical scenarios cover random signed weights, zero rows, tiny and large inputs, and widths on either side of a power of two. Both warp counts passed against an FP32 reference with relative tolerance 0.005 and absolute tolerance 0.000001. Each timed backend also passed a reference check at each measured shape. [Numerical harness](check.py)
- All implementations use the same seeded synthetic inputs and return a new output. Validation, JIT compilation and graph capture finish before timing. Each round shuffles backend order. CUDA events bracket the replay, and the raw JSON records the median, quartiles and per-round averages. [Benchmark harness](benchmark.py)
- Inputs reuse the same addresses without an explicit cache flush. GPU clocks were unmanaged. These measurements cover this operator and graph-replay workload. Model serving latency, cold-cache performance and FlashInfer performance remain open questions.

## Row tiling and the continuing experiment

Grouping two or four token rows per program passed all 60 numerical checks. At the 1024×8192 target, the fastest grouped variant took 8.63 and 8.59 µs across two runs. The best one-row variant took 8.46 and 8.39 µs. The default remains one row. [Row-tiling report](results/row-tiles/REPORT.md)

At 1024×4096 the two-row variant showed a smaller gain. The next experiment sweeps hidden width around that transition. Its plan can run one unit at a time. Completed units persist across sessions. [Current goal and handoff](GOAL.md) · [Experiment plan](plans/width-sweep.json)

```bash
python3 lab.py init --plan plans/width-sweep.json
python3 lab.py step
python3 lab.py status
```

[PyTorch RMSNorm](https://docs.pytorch.org/docs/2.7/generated/torch.nn.RMSNorm.html) · [Triton reduction tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) · [FlashInfer RMSNorm](https://docs.flashinfer.ai/generated/flashinfer.norm.rmsnorm.html)
