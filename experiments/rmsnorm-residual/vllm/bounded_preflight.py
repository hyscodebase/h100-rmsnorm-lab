"""Check two in-place kernels on fresh inputs with a bounded FP32 oracle."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import run

SEED = 113
ADDITIONS = (run.HERE / "PRECISION_PLAN.md", run.HERE / "float32_gate.py", Path(__file__).resolve())
SOURCES = (*run.SOURCES, *(str(p.relative_to(run.ROOT)) for p in ADDITIONS))


def verify_case(torch, outputs, inputs, case):
    from float32_gate import assess, bf16_bits

    expected, normalized = run.reference(*inputs)
    activations = []
    for output, updated in outputs:
        if (output.shape != expected.shape or output.dtype != torch.bfloat16
                or not bool(torch.isfinite(output).all()) or not torch.equal(bf16_bits(updated), bf16_bits(expected))):
            raise AssertionError("output_or_residual_bits")
        activations.append(output)

    strict, rejected = assess(expected, inputs[2], normalized, *activations)
    evidence = {"fixture": case, "elements": expected.numel(), "strict_outside_tolerance": strict.sum().item(),
                "bounded_rejected": rejected.sum().item(),
                "pairs_differ_bitwise": (bf16_bits(activations[0]) != bf16_bits(activations[1])).sum().item(),
                "accepted_coords": (strict & ~rejected).nonzero().tolist(),
                "rejected_coords": rejected.nonzero()[:16].tolist(),
                "max_abs_error": [float((a.double() - normalized).abs().max().item()) for a in activations],
                "passed": not bool(rejected.any())}
    if case == "weight_rounding":
        probes = [a[0, 1].item() for a in activations]
        evidence.update(weight_probe=probes, passed=evidence["passed"] and probes == [1.46875, 1.46875])
    if case == "signed_zero":
        observed = [bf16_bits(a[:2, 0].contiguous()).tolist() for a in activations]
        evidence.update(normalized_zero_bits=observed, passed=evidence["passed"] and observed == [[32768, 0], [32768, 0]])
    return evidence


def preflight(asset_dir, original):
    import torch
    import check
    import triton
    from kernel import fused_inplace
    from wheel import MEMBER, sha

    started = run.original.utc_now()
    env = original["environment"]
    lib = asset_dir / Path(MEMBER).name
    if (original["seed"] != 109 or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or sha(lib) != original["asset"]["extension_sha256"]):
        raise ValueError("Prepared operator or original preflight differs.")
    props = torch.cuda.get_device_properties(0)
    if (props.name, props.total_memory, props.multi_processor_count, str(torch.__version__), torch.version.cuda,
            triton.__version__, run.original.driver_version()) != (
                env["gpu"], env["total_memory_bytes"], env["sm_count"], env["torch"], env["cuda"], env["triton"], env["driver"]
            ):
        raise ValueError("CUDA environment differs from preflight.")
    sources = {path: hashlib.sha256((run.ROOT / path).read_bytes()).hexdigest() for path in SOURCES}
    if any(sources[name] != value for name, value in original["source_sha256"].items()):
        raise ValueError("Frozen in-place source differs from the original run.")
    torch.ops.load_library(str(lib.resolve()))
    op = torch.ops._C.fused_add_rms_norm
    methods = (lambda x, r, w: op(x, r, w, 1e-6), fused_inplace)
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(SEED)
    cases = []
    with torch.inference_mode():
        for case in (*check.FIXTURES, "weight_rounding", "signed_zero"):
            inputs = run.fixture(torch, check, case, generator, device)
            outputs = []
            for fn in methods:
                x, residual, weight = (v.clone() for v in inputs)
                fn(x, residual, weight)
                if not torch.equal(weight.view(torch.int16), inputs[2].view(torch.int16)):
                    raise AssertionError("mutated_weight")
                outputs.append((x, residual))
            try:
                cases.append(verify_case(torch, outputs, inputs, case))
            except AssertionError as exc:
                cases.append({"fixture": case, "passed": False, "reason": str(exc)})
    return {"seed": SEED, "rows": run.ROWS, "width": run.WIDTH, "code_revision": os.environ["BENCH_REVISION"],
            "source_sha256": sources, "environment": env, "asset": original["asset"],
            "gate_passed": all(case["passed"] for case in cases), "fixtures": cases,
            "started_utc": started, "finished_utc": run.original.utc_now()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", os.environ.get("BENCH_REVISION", "")):
        parser.error("Provide a frozen Git revision.")
    print(json.dumps(preflight(args.asset_dir, json.loads(args.original.read_text())), indent=2, allow_nan=False))
