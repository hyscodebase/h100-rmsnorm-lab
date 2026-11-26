"""Run numerical checks on CUDA and write the results as JSON to stdout."""

import json
from itertools import product

import torch

from kernels import rmsnorm


EPS = 1e-6
RTOL = 5e-3
ATOL = 1e-6
SEED = 42
CASES = [
    (1, 1, "random"),
    (1, 4096, "random"),
    (8, 3073, "random"),
    (32, 8192, "random"),
    (8, 4096, "mixed_zero_rows"),
    (1, 4095, "constant"),
    (32, 1025, "tiny"),
    (8, 2048, "large"),
    (7, 3073, "random"),
    (1024, 8192, "random"),
]


@torch.inference_mode()
def main():
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    checks = []
    for rows, width, case in CASES:
        x = torch.randn(rows, width, generator=generator, device="cuda")
        if case == "mixed_zero_rows":
            x[::2].zero_()
        elif case == "constant":
            x.fill_(-0.625)
        elif case == "tiny":
            x.mul_(1e-5)
        elif case == "large":
            x.mul_(1024)
        x = x.to(torch.bfloat16)
        weight = torch.randn(width, generator=generator, device="cuda").to(torch.bfloat16)
        xf = x.float()
        reference = xf * torch.rsqrt(xf.square().mean(dim=1, keepdim=True) + EPS) * weight.float()
        for rows_per_program, warps in product((1, 2, 4), (4, 8)):
            actual = rmsnorm(
                x, weight, eps=EPS, num_warps=warps, rows_per_program=rows_per_program
            )
            torch.cuda.synchronize()
            error = (actual.float() - reference).abs()
            close = torch.isclose(actual.float(), reference, rtol=RTOL, atol=ATOL)
            checks.append(
                {
                    "rows": rows,
                    "width": width,
                    "input": case,
                    "rows_per_program": rows_per_program,
                    "num_warps": warps,
                    "passed": bool(close.all().item()),
                    "max_abs_error": error.max().item(),
                }
            )
    passed = all(case["passed"] for case in checks)
    print(json.dumps({
        "passed": passed, "reference": "fp32", "dtype": "bfloat16",
        "eps": EPS, "rtol": RTOL, "atol": ATOL, "seed": SEED, "checks": checks,
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
