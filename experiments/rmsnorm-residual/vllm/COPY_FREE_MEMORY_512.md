# Measure H100 memory traffic in the 512-row copy-free graphs

At 512 × 4,096 the fused kernel led the exact two-kernel path by 19.74–19.76% in three independent seeds. The split graph rereads a 4 MiB BF16 residual after the addition. Test how much of that logical read reaches DRAM and how the two paths differ in L2 read sectors. [512-row measurements](COPY_FREE_ROWS_512_REPORT.md) · [NVIDIA memory counter definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#device-memory)

Keep the same seed-161 kernel binaries, H100 stack, eight CUDA streams, 32 independent pairs per stream and identical writable output addresses. Use released vLLM only for the bitwise oracle. Record the original activation, residual and weight hashes and check both outputs after the profiled ranges.

Use Nsight Compute application range replay with fixed work and explicit CUDA profiler markers. Each range encloses one coordinated burst of 32 graph replays per stream, or 4,194,304 normalized rows. Execute two identical warm-up bursts for the selected backend immediately before the range. Keep `--cache-control none` and `--clock-control none` for both backends and record the order and replay-pass count. Application range replay reruns the application to collect counter passes while preserving concurrent CUDA Graph execution. The original 512-row campaign supplies the unprofiled throughput. [Application range replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#application-range-replay) · [CLI controls](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

Collect the following counter values per range. The installed Nsight Compute CLI lists all four for GH100.

| Counter | Unit |
| --- | --- |
| `dram__bytes_read.sum` | byte |
| `dram__bytes_write.sum` | byte |
| `lts__t_sectors_op_read.sum` | sector |
| `gpu__time_duration.sum` | ns |

Seed 165 qualifies the exact outputs, original input hashes, marker ranges, counter collection and work counts in one GPU job. After qualification, use independent seeds 166–168. Give each seed four matched fused and split range pairs, with each backend first twice. Divide DRAM bytes by normalized elements. For each pair subtract the fused read bytes per element from the split value, then take the paired median per seed. The registered DRAM effect requires at least 0.5 fewer read bytes per normalized element in every timing seed. Report the L2 sectors, write bytes and duration alongside it. Save selected raw counter values and units, seed, device and tool versions, output checks and recovery.

The H100's experimental cache contains NVIDIA's Linux x86-64 Nsight Compute 2025.1.1.2 archive, whose CLI reports version 2025.1.1.0 build 35528883. The verified archive SHA-256 is `6f55f6dfb33abe15b25a5f0c656ae8d9b6a6f33e55f0337b9b6a017daf8ad610`. The CLI advertises `app-range` and all four requested GH100 counters. Seed 165 will validate collection on the allocated GPU. [NVIDIA redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json) · [Tracking issue](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/36)
