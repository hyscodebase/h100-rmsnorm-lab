# Can a fused residual update reduce H100 inference operator latency?

The experiment targets the point where a transformer adds a layer's output to a residual stream and normalizes the result. Both the residual for a later layer and the normalized activations are needed. At 1,024 rows and width 4,096, each returned BF16 tensor is 8 MiB. One kernel can load both inputs and emit both results. A compiler can also fuse operations, so the comparison uses the fastest correct compiled implementation measured on the same inputs. [Registered plan](PLAN.md)

Width 4,096 occurs in the published Mistral-7B-v0.1 configuration. FlashInfer exposes a fused add-and-RMSNorm operator that updates its input and residual buffers in place. This experiment materializes two fresh outputs and keeps the inputs available. Its H100 latency results apply to the two-output contract. [Mistral configuration](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json) · [FlashInfer operator](https://docs.flashinfer.ai/generated/flashinfer.norm.fused_add_rmsnorm.html)

## What makes two implementations comparable?

A single BF16 rounding step can change the normalized answer. With an activation of 1, a repeating residual of 1/256 and 3/256, and nonconstant weights, rounding the sum before normalization produces 0.9921875 in the first column. Normalizing the unrounded sum produces 0.99609375 in that column. An ordinary tolerance comparison can accept both. A fixed bit-level regression distinguishes the two. The full numerical suite also checks that outputs are separate and that reusable inputs remain intact. [Independent reference and numerical checks](check.py) · [Regression tests](test_check.py)

Triton receives one fixed four-warp implementation. Each checkpoint tries four compiled PyTorch variants, including a native RMSNorm spelling, and preserves every exclusion. The timing harness fills the live outputs with NaNs before a verification replay so stale values cannot satisfy the oracle. Its timed graphs retain all 32 pairs of outputs. Raw samples retain execution order and include units. [Kernel](kernel.py) · [Benchmark](benchmark.py)

The adoption gate is a reduction of at least 5% against the fastest correct compiled comparator at the target in each of three fresh seeds. Row counts 1, 16 and 128 test whether the conclusion travels to smaller workloads. The experiment records operator latency in microseconds. [Tracking issue](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/7)
