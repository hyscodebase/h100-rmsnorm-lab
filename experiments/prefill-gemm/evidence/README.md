# Prefill experiment evidence

Start with [run-index.csv](run-index.csv) to find a shape and seed. The task ID maps to its [raw numerical checkpoint](../results), [supplementary samples](supplemental) and, for newer executions, its [controller attempt](attempts). The index links evidence using the Git revision, SHA-256 of the benchmark source, experiment hash and SHA-256 of the archived checkpoint.

| Question | Where to look | Interpretation |
| --- | --- | --- |
| What ran and when? | Index columns for M/K/N, seed, operand bytes, timed start and end, and checkpoint completion; separate [attempt JSONs](attempts) | Times are UTC. The timed end is the recorded start plus monotonic elapsed time. Controller work includes startup, warmup and artifact handling. |
| How fast was it? | Wall and CUDA event TFLOP/s, warmup duration, counts and per-batch JSON | P10/P50/P90 summarize each batch's event time divided by its GEMM count. They describe amortized batch timing. |
| Which outputs were checked? | Correctness section in each raw checkpoint and tolerances in the index | The full reduction dimension was checked at 16 sampled rows by 16 sampled columns. The raw JSON preserves the indices and maximum absolute error. |
| What did the GPU report during the timed window? | Raw checkpoint telemetry and index columns starting with `primary_` | Includes GPU and memory activity, SM clock, power, used memory, failed queries, sample counts and maximum gap. These readings determine activity eligibility under the [reviewed coverage rule](../README.md). |
| What did the separate sampler observe? | [Per-run CSVs](supplemental) and `supplemental_` index columns | Only samples with GPU-host timestamps in the half-open timed interval are included. Sample counts, time bins and largest gaps show how much was observed. Primary and supplementary power readings describe the same device and should be interpreted separately. |
| How much energy was observed? | Observed power duration (s) and estimated energy (J) for each collector | Trapezoidal integration uses consecutive valid readings at most 2.5 s apart. Missing readings break the interval. Values exclude unobserved edges and gaps. |
| How was the machine configured? | [Point-in-time environment snapshot](environment/20261126T064441Z.json) and per-run metadata | The separate snapshot records CUDA libraries, distro, driver, H100 hardware, clocks, power limits, ECC and MIG settings. It was captured on November 26, 2026 at 06:44:41 UTC. Every earlier run retains its own environment fields. |

Supplementary GPU activity, memory usage and power are available from seed 44 where samples overlap the timed window. Temperature, separate SM/memory clocks, the configured power limit and memory activity were added to the supplementary sampler during this campaign. The first archived in-window extended reading is **06:43:32.426 UTC on November 26, 2026**, for seed 46 with 8,192 rows. Empty numerical cells mean unobserved. Primary checkpoints also have SM clock and memory activity samples from earlier runs.

The controller began saving an attempt record with seed 46 at 8,192 rows. It records a portable command before launch and updates the attempt after checkpoint persistence or a handled error. A retry gets its own ID. If the controller abruptly stops, a local record can retain only the start. Earlier successful runs still have their original completion timestamps. The archive publishes terminal attempt records. The timed-window coverage decision uses the primary collector under the same protocol across all seeds.

## Collect supplementary samples on the GPU host

Start this command in a second terminal on the benchmark's Linux GPU host just before launching a checkpoint. Run from the repository root. The original allocation exposed one H100 at NVIDIA device index 0. With multiple devices, set `device` to the `nvidia-smi` index of the physical GPU running the GEMM. CUDA ordinals can be remapped.

```bash
device=0
logs=$(mktemp -d)
basic=timestamp,utilization.gpu,memory.used,memory.total,power.draw
extended=temperature.gpu,clocks.current.sm,clocks.current.memory,power.limit,utilization.memory
fields="$basic,$extended"
TZ=UTC timeout 90s stdbuf -oL nvidia-smi -i "$device" \
  --query-gpu="$fields" --format=csv,noheader,nounits -l 1 \
  | python3 experiments/prefill-gemm/smi_to_jsonl.py > "$logs/gemm-samples.jsonl"
```

The polling period is one second. The 90-second cap covers one typical 45-second measured checkpoint plus startup. The GPU host supplies the UTC timestamp and polling can miss intervals. If the extended query exits without a usable reading because of an unsupported field, rerun with `fields="$basic"`. This also reproduces the early four-metric collector. The converter leaves unavailable numeric fields null, skips malformed rows, and flushes each sample to disk. [CSV-to-JSONL converter](../smi_to_jsonl.py)

Pass the directory printed by `echo "$logs"` to the exporter as `--telemetry-dir` alongside the matching checkpoint's results directory. Use a fresh output directory for a new campaign. The exporter retains timestamps at or after each timed start and strictly before its end, so starting the collector during warmup is sufficient.

## Regenerate the index

From a checkout containing the recorded Git revisions:

```bash
python3 experiments/prefill-gemm/export_evidence.py \
  --results experiments/prefill-gemm/results \
  --output experiments/prefill-gemm/evidence
```

The exporter verifies the plan, recorded numerical source and benchmark evidence before writing the index. It preserves existing supplementary CSVs when the original monitor logs are unavailable. The optional `--telemetry-dir` argument accepts the timestamped GEMM monitor logs from the same allocated device and campaign. UTC time alignment associates readings with each run; the caller supplies the correct device logs. The export includes only listed numerical fields and UTC sample times. It merges matching timestamps and rejects conflicting readings.

To capture a new environment snapshot, run [capture_environment.py](../capture_environment.py) on the GPU host in the benchmark's Python environment. The collector for the published snapshot is at [commit 98bb5ca](https://github.com/hyscodebase/h100-rmsnorm-lab/commit/98bb5ca).
