# Check throughput under sustained multi-stream H100 load

The 16-warp implementation matches both outputs of vLLM at 128 × 4,096 and meets the three-seed 5% latency threshold. Test whether it also processes at least 5% more normalized token rows per second under sustained eight-stream load. Compare released vLLM 0.10.0, the frozen eight-warp split kernel and the new 16-warp kernel on the same H100. [Single-operator result](SMALL_ROWS_16W_REPORT.md)

## Workload

Use eight CUDA streams. Give each stream 32 disjoint pairs of contiguous BF16 buffers at 128 × 4,096, with the same weight and input values across backends. Capture one graph per backend and stream. Each graph restores both buffers from immutable device inputs before executing 32 in-place calls. The timed throughput includes those device copies and the cost of scheduling the concurrent graphs. The work unit is a row normalized by this operator.

Use a standalone checkpoint at seed 137 to verify the captured outputs and buffer independence in all eight streams. Check both outputs bitwise against released vLLM, preserve the original weight bits and run the established signed-zero and weight-rounding fixtures. Freeze the environment, both candidate instruction hashes and the extension checksum. Run fresh seeds 138–140 only if qualification passes. Each seed is its own checkpoint, with a fresh process and a complete numerical gate before collecting throughput.

For each timing seed, run three balanced rounds of at least six seconds per backend after warm-up. Randomize the initial order, then rotate it so each backend occupies each position once. Issue the same fixed-size enqueue bursts across all eight streams and wait for a completing event from each stream. Stop at the first completed burst at or beyond the target time. Record attempted and completed work, wall time, coordinated CUDA-event intervals and exact warm-up and measurement boundaries. Use the median of the three measured block rates as the backend's seed-level throughput. The adoption threshold is at least 5% higher throughput than both baselines in every seed.

Each block starts with at least one second of warm-up, and each enqueue burst submits 32 graph replays per stream. The wall-time interval starts before enqueueing and ends after synchronization. Coordinated events cover the work on all streams. Recheck the two outputs after each measured block.

## Resource evidence

Sample GPU activity, memory activity, VRAM, power, temperature and clocks through NVML during the measured periods. Retain UTC time, observer time, missed samples, and the assignment of each sample to a qualified period. Use the existing device observer as supporting evidence. A high-load claim requires at least 12 valid GPU activity samples across each backend's three blocks, no observation gap above two seconds within each block including its boundaries, and median GPU activity of at least 80% in each seed. Report the throughput and observed coverage even if the high-load claim is unsupported.
