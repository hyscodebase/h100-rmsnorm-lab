# Bound H100 energy measurement uncertainty over longer fixed-work windows

During the timestamped comparison, seed 202 had an opening query that took 1.851 s. Its reported sample left 7.938 ms of combined padding and covered the whole period. Two slow closing queries left at least 295 ms of padding and cost the seed one backend order. The follow-up tests whether longer fixed-work periods make the energy conclusion insensitive to the measured padding under an explicit power assumption. [Seed 202 raw checkpoint](timestamped-nvml-energy-512-results/s202.json) · [Timestamped-field protocol](COPY_FREE_TIMESTAMPED_NVML_ENERGY_512.md) · [Tracking issue](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/56)

Keep the 512 × 4,096 BF16 shape, eight streams, original input and output contract, and the pinned four- and eight-warp binaries. On seed 205, verify the eight standard numerical fixtures, both replay outputs, input and weight digests and availability of bracketing NVML timestamps. Use independent seeds 206–208 for timing if the qualification passes. Give each backend 4,096 bursts per period after at least three seconds of warm-up, for 17,179,869,184 normalized rows per period. Keep two pairs in opposite backend orders. Record each period atomically with its seed, backend, order and matching input digests. Resume from the earliest outstanding period and verify the same code, input and binary identities.

Record the API call clocks, field clock calibration and maximum padding for each period. A pair qualifies when both periods have advancing counters and sample intervals that bracket the completed work. If enough evidence is missing to qualify a pair, repeat the first invalid pair at most once in the same order. Keep all raw attempts and treat a missing order as an incomplete seed.

Each timed period starts in its own process and repeats the numerical checks before measuring. The saved seed's input and weight fingerprints must match across its processes. If a numerical check fails, save the failed checkpoint and skip the rest of that seed. Energy deltas and clock brackets are evaluated within each process.

On a configured H100 host, run the same command once per work unit. Supply the Git revision used for the first checkpoint on subsequent invocations. Create a file named `paused` inside the state directory to prevent the next submission. Remove it when ready to resume.

```bash
export BENCH_REVISION="$(git rev-parse HEAD)"
python3 experiments/rmsnorm-residual/vllm/long_window_energy_run.py \
  --asset-dir .lab/vllm-release --state-dir .lab/long-window-energy-512-local
```

Compute a sensitivity interval for work energy from the recorded board energy. Its upper endpoint is the full counter delta. Its lower endpoint subtracts the maximum padding time multiplied by an assumed mean power during that padding, with a floor of zero. Evaluate 700 W and 1,000 W and record the device's observed enforced power limit alongside them. These are explicit sensitivity assumptions for the padding. The field timestamp is assigned on the CPU and the physical energy counter's integration time is outside the experiment's observations. [NVML field timestamp](https://docs.nvidia.com/deploy/archive/R550/nvml-api/structnvmlFieldValue__t.html)

Across the two orders, take the medians of each backend's lower and upper energy bounds. For a complete seed, the lower bound on fractional savings is one minus the four-warp upper median divided by the eight-warp lower median, when that denominator is positive. Support the 1.5% energy criterion only when this bound at 1,000 W meets it and four-warp throughput stays within 0.5% of the eight-warp baseline in all three exact seeds. Archive both power scenarios and the paired gross counter estimates.
