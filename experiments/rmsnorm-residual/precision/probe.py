"""One paired correctness checkpoint for Inductor's BF16 intermediate cast setting."""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS, WIDTH, SEED = 1024, 4096, 104
FLAG = "TORCHINDUCTOR_EMULATE_PRECISION_CASTS"
CASES = ("random", "bf16_ties", "bf16_denominator_tie")
COORDS = ((0, 0), (0, 3), (ROWS - 1, WIDTH - 4), (ROWS - 1, WIDTH - 1))


class DiagnosticError(ValueError):
    """A public diagnostic explaining a controlled failure."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child_env(flag, cache):
    return {**os.environ, FLAG: str(int(flag)), "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"), "CUDA_CACHE_PATH": str(cache / "cuda"),
            "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "0", "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "0"}


def effective_config(requested):
    from torch._inductor import config
    actual = config.emulate_precision_casts
    if actual is not requested or os.environ.get(FLAG) != str(int(requested)):
        raise DiagnosticError("The effective Inductor setting disagrees with the requested process environment.")
    return actual, sha(config.__file__)


def selected(tensor, coords):
    r, c = zip(*coords)
    return tensor[list(r), list(c)]


def snapshot(torch, values):
    bf16 = values.to(torch.bfloat16)
    return {"values": bf16.float().tolist(), "bits_hex": [f"0x{int(v) & 0xffff:04x}" for v in bf16.view(torch.int16).tolist()]}


def fingerprint(torch, tensors):
    digester = hashlib.sha256()
    for tensor in tensors:
        digester.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digester.hexdigest()


def inspect_fixture(torch, check, name, inputs, functions, coords=COORDS):
    expected = check.oracle(*inputs)
    original = tuple(x.clone() for x in inputs)
    result = {"input_sha256": fingerprint(torch, inputs),
              "oracle": {"updated": snapshot(torch, selected(expected[0], coords)),
                         "normalized": snapshot(torch, selected(expected[1], coords)),
                         "normalized_fp64": selected(expected[1], coords).tolist()}, "backends": {}}
    retained = {}
    for label, fn in functions.items():
        # Keep both returned tensors alive and consume them independently for every method.
        retained[label] = outputs = fn(*inputs)
        if any(not torch.equal(x.view(torch.int16), orig.view(torch.int16)) for x, orig in zip(inputs, original)):
            raise DiagnosticError("A diagnostic backend mutated its input.")
        if len(outputs) != 2 or any(t.dtype != torch.bfloat16 or t.shape != inputs[0].shape for t in outputs):
            raise DiagnosticError("A diagnostic backend returned an incompatible output pair.")
        details = {"updated": snapshot(torch, selected(outputs[0], coords)),
                   "normalized": snapshot(torch, selected(outputs[1], coords)),
                   "updated_sum": outputs[0].sum(dtype=torch.float64).item(),
                   "normalized_sum": outputs[1].sum(dtype=torch.float64).item()}
        try:
            check.verify_outputs(outputs, expected, inputs)
            if name in check.ROUNDING_TARGETS:
                check.check_rounding_entries(outputs[1], expected[1], check.ROUNDING_TARGETS[name])
        except AssertionError as exc:
            details.update(passed=False, reason=str(exc))
        else:
            details.update(passed=True, reason=None)
        result["backends"][label] = details
    return result


def worker(requested):
    # The coordinator sets the process environment before this interpreter starts.
    import torch
    import triton
    sys.path.insert(0, str(HERE.parent))
    import check
    import kernel

    actual, config_hash = effective_config(requested)
    if str(torch.__version__).split("+", 1)[0] != "2.7.1" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise DiagnosticError("The registered checkpoint requires CUDA PyTorch 2.7.1 and one visible GPU.")
    props = torch.cuda.get_device_properties(0)
    if "H100" not in props.name or (props.major, props.minor) != (9, 0):
        raise DiagnosticError("The registered device is an H100.")
    drivers = set(subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                          text=True, timeout=10).splitlines())
    if len(drivers) != 1 or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", next(iter(drivers))):
        raise DiagnosticError("Could not establish a single public driver version.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    env = {"torch": str(torch.__version__), "torch_git_revision": torch.version.git_version,
           "cuda": torch.version.cuda, "triton": triton.__version__, "python": platform.python_version(),
           "driver": drivers.pop(), "gpu": props.name, "compute_capability": [props.major, props.minor],
           "total_memory_bytes": props.total_memory, "sm_count": props.multi_processor_count}
    sources = {"precision/PLAN.md": sha(HERE / "PLAN.md"), "precision/probe.py": sha(__file__),
               "check.py": sha(check.__file__), "kernel.py": sha(kernel.__file__),
               "torch/_inductor/config.py": config_hash}
    with torch.inference_mode():
        generator = torch.Generator(device=device).manual_seed(SEED)
        functions = {"eager": check.eager_reference, "fused": kernel.fused_residual_rmsnorm,
                     "compiled": torch.compile(check.eager_reference, mode="default", fullgraph=True, dynamic=False)}
        cases = {name: inspect_fixture(torch, check, name, check.make_inputs(name, ROWS, generator, device), functions)
                 for name in CASES}
    return {"requested": requested, "effective": actual, "environment": env, "source_sha256": sources, "cases": cases}


def decide(off, on):
    for run, expected in ((off, False), (on, True)):
        if run["effective"] is not expected or run["requested"] is not expected:
            raise DiagnosticError("Paired configuration mismatch.")
    if off["environment"] != on["environment"] or off["source_sha256"] != on["source_sha256"]:
        raise DiagnosticError("Software, hardware or source changed between diagnostic children.")
    for case in CASES:
        a, b = off["cases"][case], on["cases"][case]
        if a["input_sha256"] != b["input_sha256"] or a["oracle"] != b["oracle"]:
            raise DiagnosticError("Inputs or FP64 oracle changed between diagnostic children.")
    def passed(run, case, method):
        return run["cases"][case]["backends"][method]["passed"] is True
    if not all(passed(run, case, method) for run in (off, on) for case in CASES for method in ("eager", "fused")):
        return "control_failed"
    if not all(passed(on, case, "compiled") for case in CASES):
        return "on_failed"
    if any(not passed(off, case, "compiled") for case in CASES if case != "random"):
        return "restored_on_these_probes"
    return "both_passed" if passed(off, "random", "compiled") else "off_random_only_failure"


def paired():
    revision = os.getenv("BENCH_REVISION")
    if revision is not None and not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision):
        raise DiagnosticError("BENCH_REVISION must be a full, clean commit ID.")
    arms = []
    started = datetime.now(timezone.utc).isoformat()
    for flag in (False, True):
        with tempfile.TemporaryDirectory(prefix="rmsnorm-cast-") as cache:
            proc = subprocess.run([sys.executable, str(HERE / "probe.py"), "--worker", str(int(flag))],
                                  env=child_env(flag, Path(cache)), text=True, stdout=subprocess.PIPE, check=True)
            arms.append(json.loads(proc.stdout))
    return {"rows": ROWS, "width": WIDTH, "seed": SEED, "code_revision": revision, "flag": FLAG,
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "coordinates": COORDS, "decision": decide(*arms), "off": arms[0], "on": arms[1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("0", "1"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        result = worker(args.worker == "1") if args.worker is not None else paired()
    data = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if not args.output:
        print(data, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(data)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, args.output)
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except DiagnosticError as exc:
        sys.exit(f"Precision diagnostic failed: {exc}")
    except (KeyError, ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(f"Precision diagnostic failed ({type(exc).__name__}).")
