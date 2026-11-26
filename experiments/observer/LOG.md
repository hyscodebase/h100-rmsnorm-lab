# Observer comparison log

All times below are UTC on November 26, 2026.

| Time | Event | Evidence |
| --- | --- | --- |
| Before collection | Registered the gate, 18-arm schedule and fixed input shape. Froze the three runtime sources at 5bd202d. | [Design](../prefill-gemm/NEXT.md) · [Source revision](https://github.com/hyscodebase/h100-rmsnorm-lab/commit/5bd202d) |
| 07:16 | Verified CLI and NVML access to the same physical H100 and obtained both readings before any measured compute. | [Preflight](results/preflight.json) |
| 07:17–07:21 | First complete A/B/C seed block | Standalone CLI returned 38 usable samples, maximum gap 7.156 s, and direct NVML returned 45, maximum gap 1.366 s. B/A wall throughput was 0.99706. The simultaneous arm had 38 matched slots and a median absolute activity difference of 0 percentage points. [CLI](results/s51-A.json) · [NVML](results/s51-B.json) · [Paired](results/s51-C.json) |
| 07:22–07:30 | Completed two more seed blocks | Direct NVML passed the standalone coverage criterion in all three completed seeds. CLI passed in one. Matched simultaneous activity counts for seeds 52 and 53 were 22 and 32. The decision remains open until all 18 runs complete. [Seed 52](results/s52-C.json) · [Seed 53](results/s53-C.json) |
| 07:43 | Completed 18 measured runs and evaluated the frozen gates | Direct NVML passed 6/6 standalone coverage checks, CLI 3/6. All six simultaneous arms had at least 20 usable pairs and the standalone median B/A throughput ratio was 0.99873. [Final report](REPORT.md) · [Next kernel PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/8) |

At preflight CLI reported 4 MiB used and the NVML call reported 567.69 MiB, while both activity samples were 0%. The samples can have different collection times and the two memory counters can have different accounting semantics. The registered agreement gate uses GPU activity. Retain memory readings by source and interpret that comparison separately.
