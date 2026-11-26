"""Run one resumable H100 energy period or the numerical qualification."""
import argparse
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import energy_counter_boundaries_run as boundaries
import four_warp_energy_run as energy
import run
import timestamped_nvml_energy_run as field

ROWS, WIDTH = energy.ROWS, energy.WIDTH
NUMERIC_SEED, TIMING_SEEDS = 205, (206, 207, 208)
MEASURED_BURSTS = 4_096
COUNTER_FUNCTION = field.COUNTER_FUNCTION
ANCHOR = energy.ANCHOR
SOURCES = (*field.SOURCES, *(str(path.relative_to(run.ROOT)) for path in (
    Path(__file__).resolve(), run.HERE / "COPY_FREE_LONG_WINDOW_ENERGY_512.md",
    run.HERE / "timestamped-nvml-energy-512-results/s201.json",
    run.ROOT / "lab.py", run.ROOT / "residual_plan.py",
)))


@dataclass(frozen=True)
class WorkUnit:
    seed: int
    index: int | None = None
    order_index: int | None = None

    def __post_init__(self):
        if self.seed == NUMERIC_SEED and self.index is None and self.order_index is None:
            return
        if (self.seed not in TIMING_SEEDS or type(self.index) is not int or self.index not in range(6)
                or type(self.order_index) is not int or self.order_index not in (0, 1)
                or (self.index < 4 and self.order_index != self.index // 2)):
            raise ValueError("The seed, period or backend order is outside the registered experiment.")

    @property
    def name(self):
        return f"s{self.seed}" if self.index is None else f"s{self.seed}-p{self.index}"

    @property
    def period(self):
        if self.index is None:
            return None
        order = energy.base.balanced_order(self.seed, energy.BACKENDS)
        position = self.index % 2
        return {"index": self.index, "pair_index": self.index // 2, "order_index": self.order_index,
                "position": position, "backend": order[self.order_index][position]}

    def verify_identity(self, record):
        if (record["rows"], record["width"], record["seed"], record["period"]) != (
            ROWS, WIDTH, self.seed, self.period
        ):
            raise ValueError(f"Unexpected checkpoint identity for {self.name}.")


def read_records(directory):
    return {path.stem: json.loads(path.read_bytes()) for path in directory.glob("s[0-9]*.json")}


def period_is_exact(record):
    window = record["window"]
    return record["numerically_exact"] and record["seed_fingerprints_match"] is not False and (window is None or (
        record["post_input_fingerprints"] == record["numerical"]["source_fingerprints"]
        and energy.base.outputs_match(window["post_graph_checks"])
    ))


def pair_qualifies(pair):
    return len(pair) == 2 and all(
        record["window"] is not None and period_is_exact(record)
        and record["window"]["sample_bounds"]["status"] == "qualified" for record in pair
    )


def ordered_work(records):
    qualification = WorkUnit(NUMERIC_SEED)
    yield qualification
    first = records.get(qualification.name)
    if first is None or not first["gate_passed"]:
        return
    for seed in TIMING_SEEDS:
        original = [WorkUnit(seed, index, index // 2) for index in range(4)]
        for unit in original:
            yield unit
            if unit.name not in records:
                return
            if not period_is_exact(records[unit.name]):
                break
        else:
            retry_order = next((index for index in range(2) if not pair_qualifies(
                [records[unit.name] for unit in original[2 * index:2 * index + 2]]
            )), None)
            if retry_order is None:
                continue
            for index in (4, 5):
                unit = WorkUnit(seed, index, retry_order)
                yield unit
                if unit.name not in records:
                    return
                if not period_is_exact(records[unit.name]):
                    break


def next_work(records):
    keys = set(records)
    for unit in ordered_work(records):
        if unit.name not in records:
            if keys:
                raise ValueError("Saved checkpoints skip the earliest outstanding work.")
            return unit
        unit.verify_identity(records[unit.name])
        keys.remove(unit.name)
    if keys:
        raise ValueError("Saved checkpoints fall outside the registered sequence.")
    return None


def checkpoint(unit, asset_dir, qualification, revision, *, seed_reference=None):
    period = unit.period
    needs_seed_reference = period is not None and period["index"] > 0
    if needs_seed_reference != (seed_reference is not None):
        raise ValueError("Later periods require the first saved period of the same seed.")

    def validate(provenance):
        energy.validate_qualification(qualification, NUMERIC_SEED, provenance, COUNTER_FUNCTION, field.counter_passed)
        if seed_reference is not None:
            WorkUnit(unit.seed, 0, 0).verify_identity(seed_reference)
            if (any(seed_reference[key] != value for key, value in provenance.items())
                    or not period_is_exact(seed_reference)):
                raise ValueError("The seed reference does not match the frozen experiment's exact first period.")

    def collect(torch, verified, sensor, _cuda_uuid, exact):
        matched = None if seed_reference is None else (
            seed_reference["numerical"]["source_fingerprints"] == verified.numerical["source_fingerprints"]
        )
        payload = {"counter_function": COUNTER_FUNCTION, "period": period,
                   "counter_probe": None, "window": None, "post_input_fingerprints": None,
                   "seed_fingerprints_match": matched}
        if period is None:
            probe = None if sensor is None else field.probe_counter(
                sensor, verified.runner, verified.graphs[energy.BASELINE], max_padding_ns=None,
            )
            payload.update(counter_probe=probe, gate_passed=exact and field.counter_passed(probe))
        elif exact and matched is not False and sensor is not None:
            payload["window"] = boundaries.collect_period(
                torch, verified, sensor, time.perf_counter(), period["backend"],
                bursts=MEASURED_BURSTS, measure_window=partial(field.fixed_window, max_padding_ns=None),
            )
            payload["post_input_fingerprints"] = energy.base.input_fingerprints(verified.works, verified.weight)
        return payload

    return energy.checked_device_run(unit.seed, asset_dir, revision, source_paths=SOURCES, collect=collect,
                                     validate=validate if period is not None else None)


def run_next(directory, asset_dir, revision):
    # The standalone H100 runner uses the same checkpoint sequence as the SSH worker.
    sys.path.insert(0, str(run.ROOT))
    import lab
    from residual_plan import verify_revision

    directory.mkdir(parents=True, exist_ok=True)
    with lab.worker_lock(directory):
        if (directory / "paused").exists():
            raise ValueError("Paused at a saved checkpoint.")
        records = read_records(directory)
        unit = next_work(records)
        if unit is None:
            print("All registered work is recorded.")
            return
        source_hashes = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
        verify_revision(run.ROOT, revision, tuple(sorted(source_hashes.items())))
        if any(record["source_sha256"] != source_hashes or record["code_revision"] != revision
               for record in records.values()):
            raise ValueError("The state directory belongs to another frozen source revision.")
        if (directory / "paused").exists():
            raise ValueError("Paused at a saved checkpoint.")
        reference = records.get(f"s{unit.seed}-p0") if unit.index is not None and unit.index > 0 else None
        result = checkpoint(unit, asset_dir, records.get(f"s{NUMERIC_SEED}"), revision, seed_reference=reference)
        unit.verify_identity(result)
        lab.save_json(directory / f"{unit.name}.json", result)
        print(f"Saved {unit.name}.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--seed", type=int, choices=(NUMERIC_SEED, *TIMING_SEEDS))
    cli.add_argument("--period-index", type=int, choices=range(6))
    cli.add_argument("--order-index", type=int, choices=(0, 1))
    cli.add_argument("--asset-dir", type=Path, required=True)
    cli.add_argument("--qualified", type=Path)
    cli.add_argument("--seed-reference", type=Path)
    cli.add_argument("--state-dir", type=Path, help="Run and atomically save the next work unit here.")
    args = cli.parse_args()
    revision = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        cli.error("Provide a frozen Git revision.")
    if args.state_dir is not None:
        if any(value is not None for value in (
            args.seed, args.period_index, args.order_index, args.qualified, args.seed_reference,
        )):
            cli.error("A state directory chooses the next unit and its qualification.")
        run_next(args.state_dir, args.asset_dir, revision)
    else:
        try:
            unit = WorkUnit(args.seed, args.period_index, args.order_index)
        except ValueError as error:
            cli.error(str(error))
        qualification = json.loads(args.qualified.read_bytes()) if args.qualified else None
        reference = json.loads(args.seed_reference.read_bytes()) if args.seed_reference else None
        result = checkpoint(unit, args.asset_dir, qualification, revision, seed_reference=reference)
        print(json.dumps(result, separators=(",", ":"), allow_nan=False))
