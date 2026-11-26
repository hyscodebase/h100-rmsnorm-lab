"""Inspect whether seed 114 disagreements fit the FP32 interval for each backend."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import run


def diagnose(asset_dir, checkpoint):
    import torch
    import check
    from float32_gate import assess, bf16_bits, endpoints
    from kernel import fused_inplace
    from wheel import MEMBER, sha

    lib = asset_dir / Path(MEMBER).name
    common = checkpoint["source_sha256"]
    if checkpoint["seed"] != 114 or sha(lib) != checkpoint["asset"]["extension_sha256"] or any(
        hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() != digest for name, digest in common.items()
    ):
        raise ValueError("This diagnostic requires the recorded seed 114 sources and vLLM binary.")
    env = checkpoint["environment"]
    if (torch.cuda.get_device_name(0), str(torch.__version__), torch.version.cuda, run.original.driver_version()) != (
        env["gpu"], env["torch"], env["cuda"], env["driver"]
    ):
        raise ValueError("The H100 stack differs from seed 114.")
    torch.ops.load_library(str(lib.resolve()))
    op = torch.ops._C.fused_add_rms_norm
    methods = (lambda x, r, w: op(x, r, w, 1e-6), fused_inplace)
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(checkpoint["seed"])
    records = []
    with torch.inference_mode():
        for case in (*check.FIXTURES, "weight_rounding", "signed_zero"):
            inputs = run.fixture(torch, check, case, generator, device)
            values = []
            for fn in methods:
                output, updated, weight = (t.clone() for t in inputs)
                fn(output, updated, weight)
                values.append(output)
            updated, expected = run.reference(*inputs)
            old, rejected = assess(updated, inputs[2], expected, *values)
            row, col = rejected.nonzero(as_tuple=True)
            untouched = old.sum().item() - row.numel()
            if row.numel():
                z = updated.double()
                mean = z[row].square().mean(-1)
                adj, low, high = endpoints(z[row, col], inputs[2][col].double(), mean)
                bits = [bf16_bits(out[row, col]) for out in values]
                accepted_independently = [(v == bf16_bits(low)) | (v == bf16_bits(high)) for v in bits]
                both_endpoints = adj & accepted_independently[0] & accepted_independently[1]
                samples = [{"row": r, "column": c, "endpoint_values": [lo, hi], "outputs": [a, b]}
                           for r,c,lo,hi,a,b in zip(row[:8].tolist(), col[:8].tolist(),
                                                   low[:8].tolist(), high[:8].tolist(),
                                                   values[0][row[:8],col[:8]].tolist(),
                                                   values[1][row[:8],col[:8]].tolist())]
            else:
                both_endpoints = torch.zeros(0, dtype=torch.bool)
                samples = []
            records.append({"fixture": case, "accepted_boundary_agreements": untouched, "rejected": row.numel(),
                            "rejected_despite_each_matching_an_endpoint": both_endpoints.sum().item(), "witnesses": samples})
    return {"seed": 114, "original_code_revision": checkpoint["code_revision"], "code_revision": os.environ["BENCH_REVISION"],
            "source_sha256": sha(Path(__file__)), "asset_sha256": checkpoint["asset"]["extension_sha256"],
            "results": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    options = parser.parse_args()
    print(json.dumps(diagnose(options.asset_dir, json.loads(options.checkpoint.read_text())), indent=2, allow_nan=False))
