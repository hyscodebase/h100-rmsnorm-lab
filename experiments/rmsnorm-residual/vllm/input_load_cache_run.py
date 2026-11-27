"""Qualify or measure one checkpoint for the fixed 512x4096 L1 cache-policy test."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time

from bounded_timing import bitwise_case, verify_fixtures
import grouped_run as grouped
import run
import sustained_run as base
from sustained_sensors import FIELDS, Nvml, Observer, associate, summarize_load
from wheel import MEMBER, sha

ROWS, WIDTH, STREAMS = 512, run.WIDTH, base.STREAMS
NUMERIC_SEED, TIMING_SEEDS = 209, (210, 211, 212)
BASELINE = "triton_8warps_out_of_place"
CANDIDATE = "triton_8warps_cg_inputs_out_of_place"
REFERENCE = run.BACKENDS[0]
BACKENDS = (REFERENCE, BASELINE, CANDIDATE)
ORDER = {
    210: (BASELINE, CANDIDATE),
    211: (CANDIDATE, BASELINE),
    212: (BASELINE, CANDIDATE),
}
BURSTS = 512
MINIMUM_WARMUP_S = 3.0
ANCHOR = run.HERE / "long-window-energy-512-results/s205.json"
PLAN = run.HERE / "input-load-cache-512-plan.json"
COMPILED_SOURCES = (
    ANCHOR, PLAN, Path(__file__).resolve(), run.HERE / "INPUT_LOAD_CACHE_512.md",
    run.HERE / "eight_warp_outputs.py",
    run.HERE / "input_cache_kernel.py", run.HERE / "grouped_run.py",
    run.HERE / "sustained_run.py", run.HERE / "sustained_sensors.py",
    run.HERE / "bounded_timing.py", run.HERE / "sixteen_warp_kernel.py",
    run.HERE / "grouped_kernel.py", run.HERE / "shuffle_kernel.py",
    run.HERE / "small_rows_kernel.py", run.HERE / "small_rows_run.py",
    run.HERE / "run.py", run.HERE / "wheel.py",
    run.HERE / "two_kernel.py",
    run.ROOT / "lab.py", run.ROOT / "gemm_plan.py",
    run.ROOT / "materialized_plan.py", run.ROOT / "residual_plan.py",
)
SOURCES = tuple(sorted(set(
    (*base.SOURCES, *(str(path.relative_to(run.ROOT)) for path in COMPILED_SOURCES))
)))


@dataclass(frozen=True)
class WorkUnit:
    seed: int
    index: int | None = None
    order_index: int | None = None

    def __post_init__(self):
        if self.seed == NUMERIC_SEED and self.index is None and self.order_index is None:
            return
        if (self.seed not in TIMING_SEEDS or type(self.index) is not int or self.index not in (0, 1)
                or type(self.order_index) is not int or self.order_index != self.index):
            raise ValueError("The checkpoint falls outside the registered sequence.")

    @property
    def name(self):
        return f"s{self.seed}" if self.index is None else f"s{self.seed}-p{self.index}"

    @property
    def period(self):
        if self.index is None:
            return None
        order = ORDER[self.seed]
        return {
            "index": self.index,
            "order_index": self.order_index,
            "position": self.index,
            "backend": order[self.order_index],
            "order": list(order),
        }

    def verify_identity(self, record):
        if (record.get("rows"), record.get("width"), record.get("seed"), record.get("period")) != (
            ROWS, WIDTH, self.seed, self.period
        ):
            raise ValueError(f"Unexpected checkpoint identity for {self.name}.")


def read_records(directory):
    return {
        path.stem: json.loads(path.read_bytes())
        for path in directory.glob("s[0-9]*.json")
        if path.is_file()
    }


def ordered_work(records):
    qualification = WorkUnit(NUMERIC_SEED)
    yield qualification
    qualified = records.get(qualification.name)
    if qualified is None or not qualified.get("gate_passed"):
        return
    for seed in TIMING_SEEDS:
        for index in (0, 1):
            unit = WorkUnit(seed, index, index)
            yield unit
            if unit.name not in records:
                return
            if not records[unit.name].get("gate_passed"):
                return


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


def source_hashes():
    return {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def inplace_fixture(fn):
    def call(x, residual, weight):
        fn(x.clone(), residual.clone(), weight, x, residual)

    return call


def capture_candidate(torch, fn, work, weight):
    x, residual = work.source
    x_out, residual_out = work.buffer

    def apply(index):
        fn(x[index], residual[index], weight, x_out[index], residual_out[index])

    def record():
        for index in range(run.OPS):
            apply(index)

    return base.capture_graph(torch, work.stream, lambda: apply(0), record)


def compile_kernel(torch, implementation):
    x = torch.zeros((ROWS, WIDTH), dtype=torch.bfloat16, device="cuda:0")
    residual, weight = torch.zeros_like(x), torch.ones_like(x[0])
    x_out, residual_out = torch.empty_like(x), torch.empty_like(x)
    compiled = implementation(x, residual, weight, x_out, residual_out)
    code = grouped.launch_metadata(compiled, ttgir=compiled.src.src)
    code.update(
        binary_sha256=hashlib.sha256(compiled.asm["cubin"]).hexdigest(),
        spills=compiled.n_spills,
    )
    code["ptx_load_cache_policy"] = [
        line.strip() for line in compiled.asm["ptx"].splitlines()
        if re.match(r"\s*(?:@!?%[a-z0-9]+\s+)?ld\.global(?:\.|\s)", line)
    ]
    return code


def ptx_load_counts(code):
    return {
        "global_load_instructions": sum(count for opcode, count in code["ptx_opcodes"].items()
                                        if opcode.startswith("ld.global")),
        "cg_load_instructions": sum(count for opcode, count in code["ptx_opcodes"].items()
                                    if re.match(r"ld\.global\.cg(?:\.|$)", opcode)),
        "ordinary_global_load_instructions": sum(count for opcode, count in code["ptx_opcodes"].items()
                                                 if opcode.startswith("ld.global")
                                                 and not re.match(r"ld\.global\.cg(?:\.|$)", opcode)),
    }


def verify_cache_codegen(baseline, candidate, anchor):
    expected = anchor["codegen"][BASELINE]
    if any(baseline.get(key) != expected.get(key)
           for key in ("num_warps", "ptx_instruction_sha256", "binary_sha256")):
        raise ValueError("The eight-warp baseline differs from its frozen H100 binary.")
    before, after = ptx_load_counts(baseline), ptx_load_counts(candidate)
    if baseline["num_warps"] != candidate["num_warps"] or candidate["num_warps"] != 8:
        raise ValueError("The cache-policy comparison changed the launch geometry.")
    if before["cg_load_instructions"] != 0 or before["global_load_instructions"] != 6:
        raise ValueError("The baseline global-load instruction set differs from seed 205.")
    if (after["cg_load_instructions"] != 4 or after["ordinary_global_load_instructions"] != 2
            or after["global_load_instructions"] != before["global_load_instructions"]):
        raise ValueError("PTX does not show exactly four input loads bypassing L1 and two ordinary weight loads.")
    if candidate["ptx_instruction_sha256"] == baseline["ptx_instruction_sha256"]:
        raise ValueError("The candidate PTX does not differ from the baseline.")
    return {
        BASELINE: before,
        CANDIDATE: after,
        "distinguishes_input_policy": True,
        "candidate_and_baseline_binary_differ": candidate["binary_sha256"] != baseline["binary_sha256"],
    }


def numeric_gate(fixtures, checks, fingerprints, inputs_unchanged):
    return (
        len(fixtures) == 8 and all(bitwise_case(case) for case in fixtures)
        and set(checks) == set(BACKENDS)
        and all(check["weight_unchanged"] and check["finite_activations"]
                and base.outputs_match(check["streams"])
                and check.get("source_fingerprints") == fingerprints
                for check in checks.values())
        and inputs_unchanged
    )


def auxiliary_snapshot(sensor, origin):
    utc = run.original.utc_now()
    start = time.perf_counter() - origin
    try:
        values, errors = sensor.read()
    except (OSError, AttributeError, RuntimeError, ValueError) as error:
        values, errors = dict.fromkeys(FIELDS), {"read": type(error).__name__}
    return {
        "utc": utc,
        "query_start_s": start,
        "query_end_s": time.perf_counter() - origin,
        "values": values,
        "errors": errors,
    }


def measure_period(torch, verified, backend, cuda_uuid, order_index):
    origin_utc = run.original.utc_now()
    origin = time.perf_counter()
    sensor = Nvml(cuda_uuid)
    observer = Observer(cuda_uuid, origin)
    graphs = verified["graphs"][backend]
    try:
        before = auxiliary_snapshot(sensor, origin)
        warmup = base.collect_window(verified["runner"], graphs, MINIMUM_WARMUP_S, origin, rows=ROWS)
        window = base.collect_window(
            verified["runner"], graphs, 0.0, origin, rows=ROWS,
            fixed_bursts=BURSTS, retain_clock=True,
        )
        after = auxiliary_snapshot(sensor, origin)
        post_checks = base.compare(torch, verified["works"], verified["reference"])
        inputs_unchanged = base.input_fingerprints(verified["works"], verified["weight"]) == verified["fingerprints"]
    finally:
        telemetry = observer.finish()
        sensor.close()
    block = {
        **window,
        "id": 0,
        "backend": backend,
        "order_index": order_index,
        "warmup": warmup,
        "auxiliary_before": before,
        "auxiliary_after": after,
        "post_graph_checks": post_checks,
    }
    associate(telemetry["slots"], [block])
    telemetry["by_backend"] = summarize_load([block], telemetry["slots"], (backend,))
    return {
        "origin_utc": origin_utc,
        "backend": backend,
        "order_index": order_index,
        "warmup": warmup,
        "window": window,
        "auxiliary_before": before,
        "auxiliary_after": after,
        "telemetry": telemetry,
        "post_graph_checks": post_checks,
        "inputs_unchanged_after_timing": inputs_unchanged,
        "outputs_match_after_timing": base.outputs_match(post_checks),
    }


def checkpoint(unit, asset_dir, revision, qualification=None, seed_reference=None):
    import torch
    import triton
    from eight_warp_outputs import fused_out_of_place as baseline_kernel
    from input_cache_kernel import fused_out_of_place as candidate_kernel

    anchor = json.loads(ANCHOR.read_bytes())
    library = asset_dir / Path(MEMBER).name
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not anchor["gate_passed"]:
        raise ValueError("Use the registered single H100 and passing seed-205 anchor.")
    environment = base.validate_stack(torch, triton, library, anchor)
    hashes = source_hashes()
    if unit.seed == NUMERIC_SEED:
        if qualification is not None or seed_reference is not None:
            raise ValueError("Seed 209 is the independent numerical qualification.")
    else:
        if (qualification is None or not qualification.get("gate_passed")
                or (qualification.get("seed"), qualification.get("rows"), qualification.get("width"))
                != (NUMERIC_SEED, ROWS, WIDTH)
                or qualification.get("source_sha256") != hashes
                or qualification.get("code_revision") != revision
                or qualification.get("environment") != environment
                or qualification.get("asset") != anchor["asset"]):
            raise ValueError("A passing seed-209 qualification from this revision is required.")
        if unit.index == 0 and seed_reference is not None:
            raise ValueError("The first timing period must establish the seed fingerprints.")
        if unit.index == 1:
            if seed_reference is None:
                raise ValueError("The second timing period needs the first period of the same seed.")
            WorkUnit(unit.seed, 0, 0).verify_identity(seed_reference)
            if (not seed_reference.get("gate_passed")
                    or any(seed_reference.get(key) != value for key, value in {
                        "code_revision": revision, "source_sha256": hashes,
                        "environment": environment, "asset": anchor["asset"],
                    }.items())):
                raise ValueError("The seed reference differs from the frozen experiment.")

    started, wall_start = run.original.utc_now(), time.perf_counter()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode():
        baseline_code = compile_kernel(torch, baseline_kernel)
        candidate_code = compile_kernel(torch, candidate_kernel)
        cache_evidence = verify_cache_codegen(baseline_code, candidate_code, anchor)
        codegen = {BASELINE: baseline_code, CANDIDATE: candidate_code}
        if unit.seed != NUMERIC_SEED:
            expected_qualification = qualification["codegen"][CANDIDATE]
            if any(candidate_code.get(key) != expected_qualification.get(key)
                   for key in ("ptx_instruction_sha256", "binary_sha256")):
                raise ValueError("The candidate binary changed after seed 209 qualification.")

        torch.ops.load_library(str(library.resolve()))
        op = torch.ops._C.fused_add_rms_norm
        vllm = lambda x, residual, weight: op(x, residual, weight, 1e-6)
        baseline = inplace_fixture(baseline_kernel)
        candidate = inplace_fixture(candidate_kernel)
        methods = {REFERENCE: vllm, BASELINE: baseline, CANDIDATE: candidate}
        fixtures = verify_fixtures(torch, methods, unit.seed, torch.device("cuda:0"), ROWS)
        works, weight, buffers = base.make_work(torch, unit.seed, torch.device("cuda:0"), rows=ROWS)
        initial_weight = weight.view(torch.int16).clone()
        fingerprints = base.input_fingerprints(works, weight)
        graphs = {
            REFERENCE: [base.capture(torch, vllm, work, weight) for work in works],
            BASELINE: [capture_candidate(torch, baseline_kernel, work, weight) for work in works],
            CANDIDATE: [capture_candidate(torch, candidate_kernel, work, weight) for work in works],
        }
        runner = base.Replay(torch, works)
        reference, graph_checks = base.verify_graphs(
            torch, runner, graphs, works, weight, initial_weight,
            backends=BACKENDS, check_inputs=True,
        )
        inputs_unchanged = base.input_fingerprints(works, weight) == fingerprints
        eligible = numeric_gate(fixtures, graph_checks, fingerprints, inputs_unchanged)
        if not eligible:
            raise ValueError("The numerical fixtures, captured outputs or input fingerprints failed.")
        seed_fingerprints_match = None
        if seed_reference is not None:
            seed_fingerprints_match = seed_reference["source_fingerprints"] == fingerprints
            if not seed_fingerprints_match:
                raise ValueError("The seed input and weight fingerprints changed between periods.")

        numerical = {
            "fixtures": fixtures,
            "graph_checks": graph_checks,
            "buffers": buffers,
            "source_fingerprints": fingerprints,
            "inputs_unchanged": inputs_unchanged,
            "seed_fingerprints_match": seed_fingerprints_match,
        }
        timing = None
        if unit.period is not None:
            backend = unit.period["backend"]
            cuda_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
            timing = measure_period(torch, {
                "graphs": graphs, "runner": runner, "works": works, "weight": weight,
                "reference": reference, "fingerprints": fingerprints,
            }, backend, cuda_uuid, unit.order_index)
            if not timing["inputs_unchanged_after_timing"] or not timing["outputs_match_after_timing"]:
                raise ValueError("The timed period changed inputs or failed its post-period output checks.")

    return {
        "seed": unit.seed,
        "rows": ROWS,
        "width": WIDTH,
        "code_revision": revision,
        "source_sha256": hashes,
        "environment": environment,
        "asset": anchor["asset"],
        "codegen": codegen,
        "ptx_load_cache_policy": cache_evidence,
        "numerical": numerical,
        "source_fingerprints": fingerprints,
        "seed_fingerprints_match": seed_fingerprints_match,
        "gate_passed": eligible,
        "period": unit.period,
        "timing": timing,
        "graph_contract": {
            "rows": ROWS,
            "width": WIDTH,
            "streams": STREAMS,
            "pairs_per_stream": run.OPS,
            "bursts": BURSTS if timing is not None else 0,
            "normalized_rows_per_period": timing["window"]["normalized_rows"] if timing is not None else 0,
        },
        "pytorch_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "pytorch_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "started_utc": started,
        "finished_utc": run.original.utc_now(),
        "wall_seconds": time.perf_counter() - wall_start,
    }


def parse_args():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--seed", type=int, choices=(NUMERIC_SEED, *TIMING_SEEDS), required=True)
    cli.add_argument("--period-index", type=int, choices=(0, 1))
    cli.add_argument("--order-index", type=int, choices=(0, 1))
    cli.add_argument("--asset-dir", required=True, type=Path)
    cli.add_argument("--qualified", type=Path)
    cli.add_argument("--seed-reference", type=Path)
    return cli


if __name__ == "__main__":
    cli = parse_args()
    args = cli.parse_args()
    revision = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        cli.error("Provide a frozen full Git revision.")
    try:
        unit = WorkUnit(args.seed, args.period_index, args.order_index)
    except ValueError as error:
        cli.error(str(error))
    if unit.period is None:
        if args.qualified is not None or args.seed_reference is not None:
            cli.error("Seed 209 does not take prior checkpoints.")
    elif args.qualified is None or (unit.index == 1) != (args.seed_reference is not None):
        cli.error("Timing periods need seed 209 qualification; period 1 also needs period 0.")
    qualification = json.loads(args.qualified.read_bytes()) if args.qualified else None
    reference = json.loads(args.seed_reference.read_bytes()) if args.seed_reference else None
    result = checkpoint(unit, args.asset_dir, revision, qualification, reference)
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
