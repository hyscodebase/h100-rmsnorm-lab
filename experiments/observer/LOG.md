# Observer comparison log

All times below are UTC on November 26, 2026.

| Time | Event | Evidence |
| --- | --- | --- |
| Before collection | Registered the gate, 18-arm schedule and fixed input shape. Froze the three runtime sources at 5bd202d. | [Design](../prefill-gemm/NEXT.md) · [Source revision](https://github.com/hyscodebase/h100-rmsnorm-lab/commit/5bd202d) |
| 07:16 | Verified CLI and NVML access to the same physical H100 and obtained both readings before any measured compute. | [Preflight](results/preflight.json) |

At preflight CLI reported 4 MiB used and the NVML call reported 567.69 MiB, while both activity samples were 0%. The samples can have different collection times and the two memory counters can have different accounting semantics. The registered agreement gate uses GPU activity. Retain memory readings by source and interpret that comparison separately.
