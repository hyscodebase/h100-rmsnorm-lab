"""Verify serialized counter boundaries, eligible pairs and energy effects."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import energy_counter_boundaries_run as protocol
import four_warp_energy_report as previous
import run
from split_schedule_report import verify_attempt_clocks
from sustained_report import verify_graph_records, verify_recovery, verify_window
from transfer_report import timestamp, verify_revision

RESULTS = run.HERE / "energy-counter-boundaries-512-results"


def verify_snapshot(sample):
    if sample.keys() != {"utc", "query_start_s", "query_end_s", "values", "errors"}:
        raise ValueError("Incomplete auxiliary device snapshot.")
    timestamp(sample["utc"])
    start, end = sample["query_start_s"], sample["query_end_s"]
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
        raise ValueError("Invalid auxiliary query timestamps.")
    values = sample["values"]
    if values.keys() != set(protocol.FIELDS) or not isinstance(sample["errors"], dict):
        raise ValueError("The auxiliary sensor fields differ.")
    for field, value in values.items():
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            or (field.endswith("_pct") and value > 100)
        ):
            raise ValueError("An auxiliary measurement is invalid.")


def verify_timing(seed, timing, *, verify_energy_window=previous.verify_energy_window,
                  qualifies_pair=protocol.pair_eligible):
    energy = protocol.energy
    order = energy.base.balanced_order(seed, protocol.BACKENDS)
    pairs = timing["pairs"]
    if timing["order"] != order or len(pairs) not in (2, 3) or timing["auxiliary_scope"] != protocol.AUXILIARY_SCOPE:
        raise ValueError("The backend order, pair cap or auxiliary scope differs.")
    timestamp(timing["origin_utc"])
    previous_query_end = 0.0
    for pair in pairs:
        index, blocks = pair["order_index"], pair["blocks"]
        if index not in (0, 1) or [b["backend"] for b in blocks] != order[index]:
            raise ValueError("An attempted pair differs from its assigned order.")
        for block in blocks:
            warmup = block["warmup"]
            before, after = block["auxiliary_before"], block["auxiliary_after"]
            verify_window(warmup, energy.WARMUP_S, rows=protocol.ROWS)
            verify_energy_window(block, energy.MEASURED_BURSTS)
            verify_snapshot(before)
            verify_snapshot(after)
            if (not previous_query_end <= before["query_start_s"] <= before["query_end_s"] <= warmup["start_s"]
                    < warmup["end_s"] <= block["energy_start"]["query_start_s"]
                    or block["energy_end"]["query_end_s"] > after["query_start_s"]):
                raise ValueError("An auxiliary query or warm-up overlaps a measured counter interval.")
            previous_query_end = after["query_end_s"]
            verify_graph_records(block["post_graph_checks"], rows=protocol.ROWS)
        if pair["eligible"] != qualifies_pair(blocks):
            raise ValueError("The recorded pair eligibility differs from the counter boundaries.")
    intended = [0, 1]
    rejected = next((index for index, pair in enumerate(pairs[:2]) if not pair["eligible"]), None)
    if rejected is not None:
        intended.append(rejected)
    if [pair["order_index"] for pair in pairs] != intended:
        raise ValueError("A retry must repeat the first rejected pair exactly once.")
    if any(timing[key] != value for key, value in protocol.accepted_summary(pairs).items()):
        raise ValueError("The accepted pairs or aggregate measurements differ from the raw windows.")
    exact = all(energy.base.outputs_match(block["post_graph_checks"]) for p in pairs for block in p["blocks"])
    if timing["outputs_match_after_timing"] != exact:
        raise ValueError("Post-period output exactness differs.")


def verify_record(record, seed, anchor, sources, revision):
    return previous.verify_record(record, seed, anchor, sources, revision, timing_seeds=protocol.TIMING_SEEDS,
                                  verify_timing_fn=verify_timing)


def read_results(folder=RESULTS, *, campaign=protocol, verify=verify_record):
    records = {int(path.stem[1:]): json.loads(path.read_bytes()) for path in folder.glob("s[0-9]*.json")}
    first = records[campaign.NUMERIC_SEED]
    expected = campaign.SEEDS if first["gate_passed"] else (campaign.NUMERIC_SEED,)
    if records.keys() != set(expected):
        raise ValueError("The archive differs from the registered seed sequence.")
    sources, revision = first["source_sha256"], first["code_revision"]
    if sources.keys() != set(campaign.SOURCES):
        raise ValueError("The frozen source manifest differs.")
    verify_revision(run.ROOT, revision, tuple(sorted(sources.items())))
    anchor_bytes = campaign.ANCHOR.read_bytes()
    if hashlib.sha256(anchor_bytes).hexdigest() != sources[str(campaign.ANCHOR.relative_to(run.ROOT))]:
        raise ValueError("The earlier verified kernel anchor differs.")
    anchor = json.loads(anchor_bytes)
    seen = set()
    for seed, record in sorted(records.items()):
        digests = verify(record, seed, anchor, sources, revision)
        if seen & digests:
            raise ValueError("Independent seeds share input digests.")
        seen.update(digests)
    verify_attempt_clocks(folder, records, earlier_clock=folder / "clock-probe-before.json")
    verify_recovery(records, json.loads((folder / "checkpoint-recovery.json").read_bytes()), revision,
                    require_timing=any(record["timing"] for record in records.values()))
    return records


def adopted(records, timing_seeds=protocol.TIMING_SEEDS):
    effects = [records.get(seed, {}).get("effect") for seed in timing_seeds]
    return all(effect and effect["passes_energy"] and effect["passes_throughput"] for effect in effects)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, nargs="?", default=RESULTS)
    records = read_results(cli.parse_args().directory)
    for seed, record in sorted(records.items()):
        print(f"{seed}: exact={record['numerically_exact']} counter_ready={protocol.counter_passed(record['counter_probe'])}")
        if timing := record["timing"]:
            maximum = max(protocol.energy.boundary_padding_seconds(b) for p in timing["pairs"] for b in p["blocks"])
            print(f"  accepted pairs: {len(timing['accepted_pair_indices'])}/2, attempted: {len(timing['pairs'])}")
            print(f"  maximum attempted padding: {1000*maximum:.3f} ms")
        if effect := record["effect"]:
            print(f"  energy saving: {100*effect['energy_saving']:+.4f}%")
            print(f"  throughput change: {100*effect['throughput_change']:+.4f}%")
    print("Registered energy criteria met:", adopted(records))
