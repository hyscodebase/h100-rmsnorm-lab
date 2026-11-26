"""Verify the exact split scheduling archives and recompute acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import copy_free_run as copy_free
from copy_free_report import GUARDS, clock_bounds, verify_codegen, verify_supplemental_samples
import run
from sixteen_warp_report import verify_attempts
import split_schedule_run as protocol
from sustained_report import admissible, verify_graph_records, verify_measurement, verify_numerical, verify_recovery
from transfer_report import timestamp, verify_revision

RESULTS = run.HERE / "split-schedule-512-results"
EARLIER_CLOCK = run.HERE / "copy-free-512-results/clock-probe.json"


def verify_schedules(checks, identities, fingerprints):
    for record in checks.values():
        actual = record["compiled_kernels_by_stream"]
        if (set(record["alias_guards"]) != GUARDS or len(actual) != protocol.base.STREAMS
                or any(item != identities for item in actual)):
            raise ValueError("A schedule uses a different buffer contract or split binary.")
        verify_graph_records(record["streams"], rows=protocol.ROWS)
    return protocol.schedule_gate(checks, fingerprints)


def verify_record(record, seed, anchor, sources, revision):
    if ((record["seed"], record["rows"], record["width"]) != (seed, protocol.ROWS, protocol.WIDTH)
            or record["source_sha256"] != sources or record["code_revision"] != revision
            or record["environment"] != anchor["environment"] or record["asset"] != anchor["asset"]):
        raise ValueError("The stack, shape or source identity differs.")
    verify_codegen(record["codegen"], anchor["codegen"], anchor["codegen"], workload=protocol.WORKLOAD)
    ids = copy_free.binary_identity(anchor["codegen"], candidate=protocol.FUSED)
    expected = {key: ids[key] for key in ("split_add", "split_norm")}
    numerical = record["numerical"]
    digests = verify_numerical(numerical, protocol.WORKLOAD.backends, rows=protocol.ROWS)
    if any(set(guards) != GUARDS for guards in numerical["alias_guards"].values()):
        raise ValueError("An output alias check is missing.")
    eligible = (copy_free.numeric_gate(numerical, protocol.WORKLOAD.backends)
                and verify_schedules(record["schedule_checks"], expected, numerical["source_fingerprints"]))
    timing = record["timing"]
    if record["gate_passed"] != eligible or bool(timing) != (eligible and seed in protocol.TIMING_SEEDS):
        raise ValueError("Recorded numerical gate or timing eligibility differs.")
    if timing:
        verify_measurement(seed, timing, backends=protocol.BACKENDS, period_s=copy_free.PERIOD_S, rows=protocol.ROWS)
    return digests


def verify_attempt_clocks(folder, results):
    earlier = json.loads(EARLIER_CLOCK.read_bytes())
    later = json.loads((folder / "clock-probe.json").read_bytes())
    attempts = [json.loads(path.read_bytes()) for path in (folder / "attempts").glob("*.json")]
    for attempt in attempts:
        if not (timestamp(earlier["collected_utc"]) <= timestamp(attempt["started_utc"])
                <= timestamp(attempt["finished_utc"]) <= timestamp(later["collected_utc"])):
            raise ValueError("The two clock probes do not surround an attempt.")
        if (attempt["outcome"] == "checkpoint_saved"
                and not 0 < results[attempt["seed"]]["wall_seconds"] <= attempt["elapsed_s"]):
            raise ValueError("The GPU checkpoint did not fit in the local monotonic attempt.")
    # The earlier and later probes saw different offsets. Keep both in the audit.
    before, after = clock_bounds(earlier), clock_bounds(later)
    span = (min(before[0], after[0]), max(before[1], after[1]))
    verify_attempts(folder, results, rows=protocol.ROWS, remote_clock_bounds_s=span)


def read_results(folder=RESULTS):
    results = {int(path.stem[1:]): json.loads(path.read_bytes()) for path in folder.glob("s[0-9]*.json")}
    first = results[protocol.NUMERIC_SEED]
    expected = protocol.SEEDS if first["gate_passed"] else (protocol.NUMERIC_SEED,)
    if results.keys() != set(expected):
        raise ValueError("The archive differs from the registered seed set.")
    revision, sources = first["code_revision"], first["source_sha256"]
    if sources.keys() != set(protocol.SOURCES):
        raise ValueError("The schedule source manifest differs.")
    verify_revision(run.ROOT, revision, tuple(sorted(sources.items())))
    anchor = protocol.ANCHOR.read_bytes()
    if hashlib.sha256(anchor).hexdigest() != sources[str(protocol.ANCHOR.relative_to(run.ROOT))]:
        raise ValueError("The reference 512-row checkpoint changed.")
    anchor = json.loads(anchor)
    seen = set()
    for seed, record in sorted(results.items()):
        digests = verify_record(record, seed, anchor, sources, revision)
        if seen & digests:
            raise ValueError("Independent seeds share input digests.")
        seen.update(digests)
    verify_attempt_clocks(folder, results)
    verify_supplemental_samples(folder)
    if any(record["timing"] for record in results.values()):
        verify_recovery(results, json.loads((folder / "checkpoint-recovery.json").read_bytes()), revision)
    return results


def adopted(data):
    timings = [data.get(seed, {}).get("timing") for seed in protocol.TIMING_SEEDS]
    return all(admissible(timing) and timing["comparisons"][protocol.STAGED]["passes_5pct"] for timing in timings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=RESULTS)
    data = read_results(parser.parse_args().directory)
    for seed, record in sorted(data.items()):
        timing = record["timing"]
        if not timing:
            print(seed, "qualified" if record["gate_passed"] else "excluded")
            continue
        print(seed, "verified" if admissible(timing) else "invalidated")
        for backend in protocol.BACKENDS:
            load = timing["telemetry"]["by_backend"][backend]
            print(f"  {backend}: {timing['median_rows_per_second'][backend]/1e6:.5f} M rows/s, "
                  f"GPU median {load['metrics']['gpu_busy_pct']['median']}%, gap {load['maximum_gap_s']:.3f} s")
        print(f"  adjacent vs staged: {100*timing['comparisons'][protocol.STAGED]['throughput_gain']:+.4f}%")
    print("Registered scheduling effect reproduced:", adopted(data))
