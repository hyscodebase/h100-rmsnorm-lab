"""Locate cancellation-case disagreements without timing either implementation."""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import run


def midpoint_distance(value):
    if value == 0:
        return None
    magnitude = abs(value)
    step = math.ldexp(1.0, math.frexp(magnitude)[1] - 8)
    midpoint = (math.floor(magnitude / step) + 0.5) * step
    return {"midpoint": math.copysign(midpoint, value), "distance_in_bf16_steps": abs(magnitude - midpoint) / step}


def diagnose(directory, preflight):
    import torch
    import check
    from kernel import fused_inplace
    from wheel import MEMBER, sha

    lib = directory / Path(MEMBER).name
    if sha(lib) != preflight["asset"]["extension_sha256"]:
        raise ValueError("Operator binary differs from the original preflight.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("This diagnostic requires the allocated H100.")
    props = torch.cuda.get_device_properties(0)
    env = preflight["environment"]
    if (props.name, str(torch.__version__), torch.version.cuda, run.original.driver_version()) != (
        env["gpu"], env["torch"], env["cuda"], env["driver"]
    ):
        raise ValueError("Runtime differs from preflight.")
    if any(hashlib.sha256((run.ROOT / path).read_bytes()).hexdigest() != digest
           for path, digest in preflight["source_sha256"].items()):
        raise ValueError("Numerical source differs from preflight.")
    torch.ops.load_library(str(lib.resolve()))
    op = torch.ops._C.fused_add_rms_norm
    methods = (lambda x, r, w: op(x, r, w, 1e-6), fused_inplace)
    device = torch.device("cuda:0")
    gen = torch.Generator(device=device).manual_seed(preflight["seed"])
    with torch.inference_mode():
        for name in check.FIXTURES:
            inputs = run.fixture(torch, check, name, gen, device)
            if name == "cancellation":
                break
        x, residual, weight = inputs
        updated, reference = run.reference(*inputs)
        z64 = updated.double()
        scaled_64 = z64 * (z64.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        z32 = updated.float()
        scaled_32 = z32 * (z32.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        float32_norm = (scaled_32.bfloat16().float() * weight.float()).bfloat16()
        output = {}
        summary, union = {}, torch.zeros_like(x, dtype=torch.bool)
        allowance = check.ATOL + check.RTOL * reference.abs()
        for name, fn in zip(run.BACKENDS, methods):
            activation, remainder = x.clone(), residual.clone()
            fn(activation, remainder, weight)
            bad = (activation.double() - reference).abs() > allowance
            union |= bad
            output[name] = activation
            summary[name] = {
                "updated_matches_fp64_bits": torch.equal(remainder.view(torch.int16), updated.view(torch.int16)),
                "outside_tolerance": bad.sum().item(),
                "differs_from_rounded_fp32_path": (activation.view(torch.int16) != float32_norm.view(torch.int16)).sum().item(),
                "max_abs_error": (activation.double() - reference).abs().max().item(),
            }
        a, b = (output[n].view(torch.int16) for n in run.BACKENDS)
        witness = []
        for row, col in union.nonzero()[:16].tolist():
            truth = scaled_64[row, col].item()
            point = {"row": row, "column": col, "updated": updated[row,col].item(), "weight": weight[col].item(),
                     "scaled_fp64": truth, "scaled_fp32": scaled_32[row,col].item(), "nearby_bf16_midpoint": midpoint_distance(truth),
                     "scaled_bf16_from_fp64": scaled_64[row,col].bfloat16().item(),
                     "scaled_bf16_from_fp32": scaled_32[row,col].bfloat16().item(),
                     "normalized_fp64": reference[row,col].item(),
                     "normalized_tolerance": allowance[row,col].item(),
                     "normalized_bf16_from_fp32": float32_norm[row,col].item()}
            for name, values in output.items():
                point[name] = values[row,col].item()
            witness.append(point)
    return {"seed": preflight["seed"], "rows": run.ROWS, "width": run.WIDTH, "fixture": "cancellation",
            "elements": x.numel(), "backends_differ_bitwise_at": (a != b).sum().item(),
            "num_union_failures": union.sum().item(), "backends": summary, "witnesses": witness,
            "preflight_revision": preflight["code_revision"], "code_revision": os.environ.get("BENCH_REVISION"),
            "diagnostic_sha256": sha(Path(__file__)), "extension_sha256": preflight["asset"]["extension_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    rev = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", rev):
        parser.error("Provide a frozen Git revision.")
    print(json.dumps(diagnose(args.asset_dir, json.loads(args.preflight.read_text())), indent=2, allow_nan=False))
