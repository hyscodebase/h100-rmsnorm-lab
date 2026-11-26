"""Print a public, point-in-time environment snapshot for the prefill experiment."""

import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import warnings
from datetime import datetime, timezone
from importlib import metadata
from uuid import UUID


# Each value in the second column is the exact nvidia-smi query name.
SMI = {
    "driver_version": "driver_version",
    "max_sm_clock_mhz": "clocks.max.sm", "max_memory_clock_mhz": "clocks.max.memory",
    "power_limit_w": "power.limit", "enforced_power_limit_w": "enforced.power.limit",
    "default_power_limit_w": "power.default_limit", "min_power_limit_w": "power.min_limit",
    "max_power_limit_w": "power.max_limit", "ecc_mode": "ecc.mode.current",
    "mig_mode": "mig.mode.current", "compute_mode": "compute_mode",
}


def smi_settings(device):
    """Scope to PyTorch's device, or accept exactly one visible row. Never export its identifier."""
    settings = dict.fromkeys(SMI)
    if not shutil.which("nvidia-smi"):
        return settings
    for key, field in SMI.items():
        args = ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"]
        if device:
            args.extend(["-i", device])
        try:
            proc = subprocess.run(args, text=True, capture_output=True, timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            break
        values = proc.stdout.strip().splitlines()
        if proc.returncode or len(values) != 1:
            continue
        value = values[0].strip()
        if key.endswith(("_w", "_mhz")) and re.fullmatch(r"\d+(\.\d+)?", value):
            settings[key] = float(value)
        elif key == "driver_version" and re.fullmatch(r"\d+(\.\d+)+", value):
            settings[key] = value
        elif key.endswith("_mode") and value in {
            "Enabled", "Disabled", "Default", "Exclusive_Process", "Exclusive_Thread", "Prohibited",
        }:
            settings[key] = value
    return settings


def library_versions(major):
    """Read version properties of the CUDA libraries available to this process."""
    versions = {}
    for library, symbol, fields in [
        ("cudart", "cudaRuntimeGetVersion", [None]),
        ("cublas", "cublasGetProperty", [0, 1, 2]),  # MAJOR, MINOR, PATCH
    ]:
        versions[library] = None
        try:
            # CUDA 11's runtime SONAME includes the minor version.
            soname = "11.0" if library == "cudart" and major == "11" else major
            name = f"lib{library}.so" + (f".{soname}" if soname else "")
            query = getattr(ctypes.CDLL(name), symbol)
            parts = []
            for field in fields:
                value = ctypes.c_int()
                params = [ctypes.byref(value)] if field is None else [field, ctypes.byref(value)]
                if query(*params) != 0:
                    break
                parts.append(value.value)
            if len(parts) == len(fields):
                versions[library] = (f"{parts[0] // 1000}.{parts[0] % 1000 // 10}" if library == "cudart"
                                     else ".".join(map(str, parts)))
        except (AttributeError, OSError):
            pass
    return versions


def capture():
    try:
        distro = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        distro = {}
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = None
    host_os = {"system": platform.system(), "distribution_id": distro.get("ID"),
               "distribution_version": distro.get("VERSION_ID"), "kernel_release": platform.release(),
               "architecture": platform.machine(), "cpu_count_in_affinity_mask": cpus}
    gpu = {"name": None, "compute_capability": None, "sm_count": None, "total_memory_bytes": None}
    device = build = cudnn = None
    try:
        import torch
        build = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu.update(name=props.name, compute_capability=[props.major, props.minor],
                       sm_count=props.multi_processor_count, total_memory_bytes=props.total_memory)
            if getattr(props, "uuid", None):
                device = "GPU-" + str(UUID(str(props.uuid).removeprefix("GPU-")))
            cudnn = torch.backends.cudnn.version()
    except (ImportError, RuntimeError, OSError, AttributeError, ValueError):
        pass
    major = build.split(".")[0] if build else None
    packages = {}
    names = ["torch", "triton"]
    if major:
        names.extend(f"nvidia-{library}-cu{major}" for library in ("cublas", "cudnn", "cuda-runtime"))
    for name in names:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    libraries = library_versions(major)
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Point-in-time snapshot. Historical measurements carry their own per-run environment.",
        "os": host_os, "python": platform.python_version(), "packages": packages,
        "torch_cuda_build": build, "cuda_runtime_version": libraries["cudart"],
        "cublas_runtime_version": libraries["cublas"], "cudnn_runtime_version_code": cudnn,
        "gpu": {**gpu, **smi_settings(device)},
    }


if __name__ == "__main__":
    try:
        # Optional driver probes can emit warnings containing local installation paths.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manifest = capture()
        print(json.dumps(manifest, indent=2, allow_nan=False))
    except Exception:
        raise SystemExit("Environment snapshot failed.") from None
