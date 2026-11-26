"""Verify frozen input-preserving results and compare all four backends."""
import argparse
import json
from pathlib import Path

import out_of_place_run as protocol
import run
from sixteen_warp_report import verify_attempts
from sustained_report import admissible, verify_measurement, verify_numerical, verify_recovery
from transfer_report import verify_revision


def read_results(folder):
    files = {path.name: json.loads(path.read_bytes()) for path in folder.glob("s*.json")}
    first = files[f"s{protocol.NUMERIC_SEEDS[0]}.json"]
    expected = protocol.SEEDS if first["gate_passed"] else protocol.NUMERIC_SEEDS
    if files.keys() != {f"s{seed}.json" for seed in expected}:
        raise ValueError("The checkpoint set differs from the registered seeds.")
    anchor = json.loads(protocol.ANCHOR.read_bytes())
    sources, revision = first["source_sha256"], first["code_revision"]
    if sources.keys() != set(protocol.SOURCES):
        raise ValueError("The source manifest differs.")
    verify_revision(run.ROOT, revision, tuple(sorted(sources.items())))
    data, digests = {}, set()
    for seed in expected:
        result = files[f"s{seed}.json"]
        if (result["seed"], result["rows"], result["width"]) != (seed, protocol.ROWS, protocol.WIDTH):
            raise ValueError("The seed or shape differs.")
        if (result["code_revision"] != revision or result["source_sha256"] != sources
                or result["asset"] != anchor["asset"] or result["environment"] != anchor["environment"]):
            raise ValueError("Source or stack provenance differs.")
        if result["graph_contract"] != protocol.graph_contract():
            raise ValueError("The copy or operator contract differs.")
        codegen = result["codegen"]
        for name in protocol.base.BACKENDS[1:]:
            for field in ("ptx_instruction_sha256", "binary_sha256"):
                if codegen[name][field] != anchor["codegen"][name][field]:
                    raise ValueError("A frozen baseline changed.")
        candidate = codegen[protocol.CANDIDATE]
        for field in ("ptx_instruction_sha256", "binary_sha256"):
            if candidate[field] != first["codegen"][protocol.CANDIDATE][field]:
                raise ValueError("The candidate changed after preflight.")
        if (candidate["num_warps"], candidate["required_threads"], candidate["emitted_shared_loads"],
                candidate["ptx_opcodes"].get("bar.sync")) != (16, 512, 4, 1):
            raise ValueError("The compiled layout or collective differs.")
        numerical = result["numerical"]
        current = verify_numerical(numerical, protocol.BACKENDS)
        if (digests & current
                or set(numerical["alias_guards"]) != {"activation_output_aliases_input", "residual_output_aliases_input",
                                                     "outputs_alias_each_other"}):
            raise ValueError("Source independence or output alias evidence is invalid.")
        digests.update(current)
        passed = protocol.numeric_gate(numerical)
        if passed != result["gate_passed"] or bool(result["timing"]) != (passed and seed in protocol.TIMING_SEEDS):
            raise ValueError("The recorded eligibility differs.")
        if result["timing"]:
            verify_measurement(seed, result["timing"], backends=protocol.BACKENDS, period_s=protocol.PERIOD_S)
        data[seed] = result
    verify_attempts(folder, data)
    if any(result["timing"] for result in data.values()):
        verify_recovery(data, json.loads((folder / "checkpoint-recovery.json").read_bytes()), revision)
    return data


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, nargs="?", default=run.HERE / "out-of-place-results")
    args = cli.parse_args()
    for seed, result in sorted(read_results(args.directory).items()):
        timing = result["timing"]
        if not timing:
            print(seed, "qualified" if result["gate_passed"] else "excluded")
            continue
        print(seed, "verified" if admissible(timing) else "invalidated")
        for backend in protocol.BACKENDS:
            load = timing["telemetry"]["by_backend"][backend]
            gpu = load["metrics"]["gpu_busy_pct"]
            print(f"  {backend}: {timing['median_rows_per_second'][backend]/1e6:.3f} M rows/s"
                  f"   GPU median={gpu['median']}% samples={gpu['count']} gap={load['maximum_gap_s']:.3f} s")
        for name, values in timing["comparisons"].items():
            print(f"  vs {name}: {100 * values['throughput_gain']:+.3f}% pass={values['passes_5pct']}")
