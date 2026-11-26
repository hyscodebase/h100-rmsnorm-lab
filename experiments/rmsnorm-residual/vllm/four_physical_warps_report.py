"""Verify the 512-row four-physical-warp experiment and its adoption decision."""
import argparse
import hashlib
import json
from pathlib import Path

import copy_free_run as copy_free
from copy_free_report import GUARDS, verify_supplemental_samples
import four_physical_warps_run as protocol
import run
from split_schedule_report import verify_attempt_clocks
from sustained_report import admissible, verify_measurement, verify_numerical, verify_recovery
from transfer_report import verify_revision

RESULTS = run.HERE / "four-physical-warps-512-results"


def verify_record(record, seed, anchor, sources, revision, qualification):
    if ((record["seed"], record["rows"], record["width"]) != (seed, protocol.ROWS, protocol.WIDTH)
            or record["source_sha256"] != sources or record["code_revision"] != revision
            or record["environment"] != anchor["environment"] or record["asset"] != anchor["asset"]):
        raise ValueError("Unexpected hardware, source, seed or shape.")
    codegen = record["codegen"]
    protocol.verify_codegen(codegen, anchor)
    if protocol.identities(codegen) != protocol.identities(qualification["codegen"]):
        raise ValueError("A measured kernel differs from qualification.")
    expected = copy_free.graph_contract(protocol.ROWS, backends=protocol.VERIFICATION_BACKENDS)
    if record["graph_contract"] != expected:
        raise ValueError("The benchmark graph introduces an unexpected copy or kernel call.")
    numerical = record["numerical"]
    digests = verify_numerical(numerical, protocol.VERIFICATION_BACKENDS, rows=protocol.ROWS)
    if any(set(guards) != GUARDS for guards in numerical["alias_guards"].values()):
        raise ValueError("An alias guard is missing.")
    passed = copy_free.numeric_gate(numerical, protocol.VERIFICATION_BACKENDS)
    timing = record["timing"]
    if record["gate_passed"] != passed or (timing is not None) != (passed and seed in protocol.TIMING_SEEDS):
        raise ValueError("The numerical gate or timing admission differs.")
    if timing:
        verify_measurement(seed, timing, backends=protocol.BACKENDS, period_s=copy_free.PERIOD_S, rows=protocol.ROWS)
    return digests


def read_results(folder=RESULTS):
    records = {int(path.stem[1:]): json.loads(path.read_bytes()) for path in folder.glob("s[0-9]*.json")}
    first = records[protocol.NUMERIC_SEED]
    expected = protocol.SEEDS if first["gate_passed"] else (protocol.NUMERIC_SEED,)
    if records.keys() != set(expected):
        raise ValueError("The archive differs from the registered sequence.")
    sources, revision = first["source_sha256"], first["code_revision"]
    if sources.keys() != set(protocol.SOURCES):
        raise ValueError("The source manifest differs.")
    verify_revision(run.ROOT, revision, tuple(sorted(sources.items())))
    anchor_bytes = protocol.ANCHOR.read_bytes()
    if hashlib.sha256(anchor_bytes).hexdigest() != sources[str(protocol.ANCHOR.relative_to(run.ROOT))]:
        raise ValueError("The earlier isolated anchor changed.")
    anchor = json.loads(anchor_bytes)
    seen = set()
    for seed, record in sorted(records.items()):
        digests = verify_record(record, seed, anchor, sources, revision, first)
        if seen & digests:
            raise ValueError("Independent seeds reuse input digests.")
        seen.update(digests)
    verify_attempt_clocks(folder, records, earlier_clock=folder / "clock-probe-before.json")
    verify_supplemental_samples(folder)
    if any(record["timing"] for record in records.values()):
        verify_recovery(records, json.loads((folder / "checkpoint-recovery.json").read_bytes()), revision)
    return records


def adopted(records):
    timings = [records.get(seed, {}).get("timing") for seed in protocol.TIMING_SEEDS]
    return all(admissible(t) and t["comparisons"][protocol.BASELINE]["passes_5pct"] for t in timings)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, nargs="?", default=RESULTS)
    records = read_results(cli.parse_args().directory)
    for seed, record in sorted(records.items()):
        timing = record["timing"]
        print(f"{seed}: qualified={record['gate_passed']}")
        if admissible(timing):
            for name, rate in timing["median_rows_per_second"].items():
                print(f"  {name}: {rate/1e6:.5f} M rows/s")
            print(f"  four-warp change: {100*timing['comparisons'][protocol.BASELINE]['throughput_gain']:+.4f}%")
    print("Candidate adopted:", adopted(records))
