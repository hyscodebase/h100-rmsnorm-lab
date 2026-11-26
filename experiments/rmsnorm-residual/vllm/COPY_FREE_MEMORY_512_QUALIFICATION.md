# Direct memory-counter qualification at 512 rows

Nsight Compute could list the requested GH100 metrics. The H100 host rejected access to the hardware counters when the profiled application reached the CUDA range. The second attempt returned `ERR_NVGPUCTRPERM` after attaching to Python. The archive contains zero profiled ranges and zero exported Nsight reports. The registered DRAM-read hypothesis remains untested. [Protocol](COPY_FREE_MEMORY_512.md) · [NVIDIA counter permission reference](https://developer.nvidia.com/ERR_NVGPUCTRPERM)

| Seed | Source | Observed outcome |
| ---: | --- | --- |
| 165 | `972f023` | Nsight rejected the explicit `--profile-from-start` option before attaching to the target. |
| 165 | `321c7c5` | The CUDA-marker launch reached the profiler range, where the host returned `ERR_NVGPUCTRPERM`. |

The two attempt journals preserve wall times, outcomes and source revisions. The error extracts retain the private profiler-log SHA-256, attachment state, number of exported profiles and number of complete application traces. The whole-device observer retained 20 samples across both attempt windows, with peaks of 14% GPU activity and 125.19 W. [Archived attempts and device samples](memory-512-results)

The follow-up holds the 512-row kernels, buffers and launch counts fixed and changes the placement of residual addition and normalization in each graph. It measures the effect of producer-to-consumer distance on throughput. [Scheduling protocol](COPY_FREE_SPLIT_SCHEDULE_512.md) · [Tracking issue](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/38)

```bash
python3 experiments/rmsnorm-residual/vllm/memory_profile_failure_report.py
```
