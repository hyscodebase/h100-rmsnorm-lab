# Match the vLLM in-place residual RMSNorm contract on H100

A four-warp Triton kernel lowered out-of-place latency by 23.44–23.83% against the strongest correct compiled PyTorch reference. A serving runtime can keep both results in the original input buffers and apply the weight after rounding the scaled activation. Compare an in-place Triton implementation against the released vLLM CUDA operator under that contract. [Previous experiment](../materialized/REPORT.md) · [vLLM fused kernel](https://github.com/vllm-project/vllm/blob/v0.10.0/csrc/layernorm_kernels.cu#L41-L92) · [Packed scalar conversions](https://github.com/vllm-project/vllm/blob/v0.10.0/csrc/type_convert.cuh#L84-L133)

## Reference and accuracy

Use the existing Linux H100 with CUDA 12.8 and PyTorch 2.7.1. Pin the official vLLM 0.10.0 CUDA wheel for that PyTorch version. Load its original operator extension in an isolated experiment directory and record the wheel and extension SHA-256 hashes. [CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.10.0/requirements/cuda.txt#L5-L10) · [Pinned wheel and published checksum](https://pypi.org/project/vllm/0.10.0/#files)

For finite BF16 activations at 1,024 × 4,096, set the residual buffer to the BF16-rounded input sum, compute the RMS in FP32 from that value, and round the scaled value to BF16 before multiplying by the BF16 weight. Write the final BF16 value into the input buffer. Check the updated residual exactly and the normalized output at relative tolerance 0.005 and absolute tolerance 1e-6 against an independent FP64 oracle that performs both rounding steps.

Use one fixed four-warp row-per-program Triton candidate. Seed 109 is numerical preflight. Run the prior six distributions, and add signed zeros and an exact-bit probe for the round before weight multiplication. The first four columns of the weight probe repeat activation values 1, 1.25, 2, 0.75 with zero residual and weights 0.375, 1.5625, 0.7421875, 1.1328125. Column 1 must return 1.46875. One final cast after weighting would return 1.4609375. Verify that only the two designated buffers change and that they remain disjoint. Both implementations must pass preflight before collecting latency.

## Timing and decision

Use fresh seeds 110, 111 and 112, one checkpoint for each. Validate both implementations again. Capture 32 calls on 32 disjoint input and residual pairs from the same seed. Before every replay, restore each pair from its original contents and enqueue the start event after the restoration copies. Verify all 32 mutated output pairs in an untimed replay. Retain each pair and measure 21 shuffled CUDA-event rounds. Report the latency of the in-place operator and the independent restoration evidence. Adoption at this single input shape requires at least 5% lower median latency than the vLLM operator in each seed.

Freeze the prep, implementation and plan before the numerical preflight. Keep a saved checkpoint even if a reference is unavailable or fails the numerical gate. Retain the failed phase and the actual target runtime versions.
