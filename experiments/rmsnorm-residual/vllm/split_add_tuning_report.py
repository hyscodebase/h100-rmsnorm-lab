"""Audit the seed-173 tile selection and independent held-out results."""
import argparse
import hashlib
import json
from pathlib import Path

import copy_free_run as copy_free
from copy_free_report import GUARDS, verify_codegen, verify_supplemental_samples
import run
import split_add_tuning_run as protocol
from split_schedule_report import verify_attempt_clocks
from sustained_report import verify_measurement, verify_numerical, verify_recovery
from transfer_report import verify_revision

RESULTS = run.HERE / "split-add-tiles-512-results"


def verify_additions(additions, requested, anchor, qualification=None):
    expected = {name: tile for tile, name in protocol.TILE_BACKENDS.items() if name in requested}
    if set(additions) != set(expected):
        raise ValueError("The compiled set of residual-add tiles differs.")
    normalization = anchor["codegen"][protocol.ORIGINAL]["normalization"]
    for name, record in additions.items():
        add = record["residual_add"]
        if (add["tile_elements"] != expected[name] or add["num_warps"] != protocol.ADD_WARPS or add["fp_fusion"]
                or record["normalization_identity"] != [normalization[key] for key in copy_free.IDENTITY_FIELDS]):
            raise ValueError("The compiled split contract differs.")
        if name == protocol.ORIGINAL:
            previous = anchor["codegen"][protocol.ORIGINAL]["residual_add"]
            if any(add[key] != previous[key] for key in copy_free.IDENTITY_FIELDS):
                raise ValueError("The baseline addition changed.")
        if qualification is not None:
            selected = qualification["addition_codegen"][name]["residual_add"]
            if any(add[key] != selected[key] for key in copy_free.IDENTITY_FIELDS):
                raise ValueError("A held-out addition differs from its tuning binary.")


def verify_record(record, seed, anchor, revision, sources, qualification):
    if ((record["seed"], record["rows"], record["width"]) != (seed, protocol.ROWS, protocol.WIDTH)
            or record["code_revision"] != revision or record["source_sha256"] != sources
            or record["asset"] != anchor["asset"] or record["environment"] != anchor["environment"]):
        raise ValueError("The archive changed the seed, shape, source or hardware.")
    tuning = seed == protocol.NUMERIC_SEED
    tile = None if tuning else protocol.qualified_tile(
        qualification, sources, revision, anchor["environment"], anchor["asset"],
    )
    requested = tuple(protocol.TILE_BACKENDS.values()) if tuning else protocol.held_out_backends(tile)
    verify_codegen(record["codegen"], anchor["codegen"], anchor["codegen"], workload=protocol.WORKLOAD)
    additions = record["addition_codegen"]
    verify_additions(additions, requested, anchor, None if tuning else qualification)
    numerical = record["numerical"]
    digests = verify_numerical(numerical, (copy_free.REFERENCE, *requested), rows=protocol.ROWS)
    if set(numerical["alias_guards"]) != set(requested) or any(
        set(guards) != GUARDS for guards in numerical["alias_guards"].values()
    ):
        raise ValueError("An output alias check is missing.")
    eligible = protocol.eligible_backends(numerical, additions, requested)
    if record["eligibility"] != eligible:
        raise ValueError("The recorded per-path qualification differs.")
    timing = record["timing"]
    if bool(timing) != (eligible[protocol.ORIGINAL] and (tuning or all(eligible.values()))):
        raise ValueError("An ineligible path was measured or a qualifying run is missing.")
    if timing:
        measured = [name for name in requested if eligible[name]]
        verify_measurement(seed, timing, backends=measured, rows=protocol.ROWS,
                           period_s=protocol.TUNING_PERIOD_S if tuning else copy_free.PERIOD_S,
                           rounds=protocol.TUNING_ROUNDS if tuning else None)
    exact = protocol.exact_timing(timing)
    if tuning and exact:
        tile = protocol.choose_tile(timing["median_rows_per_second"])
    if record["gate_passed"] != exact or record["selected_tile"] != tile:
        raise ValueError("The recorded exactness or tile selection differs.")
    comparison = protocol.held_out_comparison(timing, tile) if exact and not tuning else None
    if record["held_out_comparison"] != comparison:
        raise ValueError("The held-out decision differs from the original timings.")
    return digests


def read_results(folder=RESULTS):
    data = {int(path.stem[1:]): json.loads(path.read_bytes()) for path in folder.glob("s[0-9]*.json")}
    qualification = data[protocol.NUMERIC_SEED]
    expected = protocol.SEEDS if qualification["gate_passed"] else (protocol.NUMERIC_SEED,)
    if data.keys() != set(expected):
        raise ValueError("The archive differs from the registered seeds.")
    revision, sources = qualification["code_revision"], qualification["source_sha256"]
    if sources.keys() != set(protocol.SOURCES):
        raise ValueError("The source manifest differs.")
    verify_revision(run.ROOT, revision, tuple(sorted(sources.items())))
    anchor = protocol.ANCHOR.read_bytes()
    if hashlib.sha256(anchor).hexdigest() != sources[str(protocol.ANCHOR.relative_to(run.ROOT))]:
        raise ValueError("The original 512-row anchor changed.")
    anchor = json.loads(anchor)
    seen = set()
    for seed, record in sorted(data.items()):
        digests = verify_record(record, seed, anchor, revision, sources, qualification)
        if seen & digests:
            raise ValueError("Independent seeds share input digests.")
        seen.update(digests)
    verify_attempt_clocks(folder, data, earlier_clock=folder / "clock-probe-before.json")
    verify_supplemental_samples(folder)
    if any(seed in protocol.TIMING_SEEDS and record["timing"] for seed, record in data.items()):
        verify_recovery(data, json.loads((folder / "checkpoint-recovery.json").read_bytes()), revision)
    return data


def adopted(data):
    comparisons = [data.get(seed, {}).get("held_out_comparison") for seed in protocol.TIMING_SEEDS]
    return all(item and item["selected_tile_passes_5pct"] for item in comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=RESULTS)
    data = read_results(parser.parse_args().directory)
    for seed, record in sorted(data.items()):
        print(f"{seed}: verified={record['gate_passed']} selected_tile={record['selected_tile']}")
        if record["timing"]:
            for backend, rate in record["timing"]["median_rows_per_second"].items():
                print(f"  {backend}: {rate/1e6:.5f} M rows/s")
        if record["held_out_comparison"]:
            item = record["held_out_comparison"]
            print(f"  selected vs original: {100*item['selected_tile_gain_over_original']:+.4f}%")
            print(f"  fused vs stronger split: {100*item['fused_gain_over_stronger_split']:+.4f}%")
    print("New addition tile adopted:", adopted(data))
