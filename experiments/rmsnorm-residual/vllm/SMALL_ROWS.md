# Preserve the released vLLM reduction order at 128 rows

The fixed 256-thread Triton shuffle matched the eight seed-125 fixtures at 128 × 4,096. Its later CUDA Graph checks found 238 activation bit disagreements and 126 numerical rejections across seeds 126–128. The released vLLM source changes its thread block to 1,024 below 256 rows. At width 4,096 with its aligned width-eight specialization, 512 threads each process one eight-element group and the remaining threads contribute zero. The existing Triton kernel combines two groups in each lane before reducing across warps. The different FP32 addition tree is the next hypothesis for the observed mismatch. [Raw checkpoints](transfer-results) · [Released dispatch and vector path](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.0/csrc/layernorm_kernels.cu)

## Controlled change

Keep the Triton launch at 256 physical threads and preserve its memory layout, the released vLLM wheel, the two BF16 rounding steps and the updated residual. Reduce each half of the row separately within the physical warps. Store the resulting 16 FP32 logical warp sums and combine them in ascending logical warp order. All groups sum nonnegative squares, so the inactive vLLM warp totals contribute exact zero on the registered finite input suite. Verify the emitted instructions and keep the same shape and compiler settings for every checkpoint.

## Checkpoints and decision

1. Reproduce the archived capture disagreement counts from seeds 126–128, one seed per checkpoint. Check the original shuffle against the same official operator as before. Record whether the new ordering corrects every normalized output and keeps every stored residual bit equal.
2. Run the same eight fixtures and 32 disjoint captured input pairs with untouched seed 129. Require bitwise agreement for the candidate on both outputs, an unmodified weight vector and a passing bounded numerical check on every fixture and capture. Keep the signed-zero and weight-rounding probes.
3. After all four numerical checkpoints pass, use fresh timing seeds 130–132. Require the same numerical gate in each new process and a reduction of at least 5% in median kernel latency against released vLLM in each seed. Keep the untimed input restoration, 21 randomized CUDA Graph replay rounds and original event intervals. Archive any numerical exclusions before deciding where to go next.
