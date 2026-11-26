"""Independent, bounded H100 sensor workers on the same monotonic clock as GEMM."""

import ctypes as C
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import threading
import time
from uuid import UUID


SLOTS = 45
PERIOD = 1.0
QUERY_TIMEOUT = 5.0
STARTUP_TIMEOUT = 5.0
BINDING = "nvml-ctypes-v1"
FIELDS = {
    "utilization.gpu": "gpu_busy_pct", "utilization.memory": "memory_busy_pct",
    "memory.used": "device_memory_used_mib", "power.draw": "device_power_w",
    "clocks.current.sm": "sm_clock_mhz",
}


def empty_metrics():
    return dict.fromkeys(FIELDS.values())


def public_version(raw):
    return raw if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,4}", raw) else None


class Utilization(C.Structure):
    _fields_ = [("gpu", C.c_uint), ("memory", C.c_uint)]


class Memory(C.Structure):
    # nvmlMemory_t v1 for nvmlDeviceGetMemoryInfo, in bytes.
    _fields_ = [("total", C.c_ulonglong), ("free", C.c_ulonglong), ("used", C.c_ulonglong)]


class Nvml:
    def __init__(self, device):
        self.lib = C.CDLL("libnvidia-ml.so.1")
        init = self.function("nvmlInit_v2")
        self.shutdown = self.function("nvmlShutdown")
        if init() != 0:
            raise RuntimeError("NVML initialization failed")
        self.handle = C.c_void_p()
        get_handle = self.function("nvmlDeviceGetHandleByUUID", C.c_char_p, C.POINTER(C.c_void_p))
        if get_handle(device.encode("ascii"), C.byref(self.handle)) != 0:
            self.shutdown()
            raise RuntimeError("Physical device selection failed")

    def function(self, name, *args):
        fn = getattr(self.lib, name)
        fn.argtypes = args
        fn.restype = C.c_int
        return fn

    def versions(self):
        result = {"binding_version": BINDING}
        for key, name in (("nvml_version", "nvmlSystemGetNVMLVersion"), ("driver_version", "nvmlSystemGetDriverVersion")):
            buf = C.create_string_buffer(128)
            fn = self.function(name, C.POINTER(C.c_char), C.c_uint)
            if fn(buf, len(buf)) == 0:
                result[key] = public_version(buf.value.decode("ascii", errors="ignore"))
        return result

    def get(self, name, value, *enums):
        try:
            fn = self.function(name, C.c_void_p, *([C.c_uint] * len(enums)), C.POINTER(type(value)))
        except AttributeError:
            return None
        return value if fn(self.handle, *enums, C.byref(value)) == 0 else None

    def read(self):
        values = empty_metrics()
        util = self.get("nvmlDeviceGetUtilizationRates", Utilization())
        if util is not None:
            values["gpu_busy_pct"], values["memory_busy_pct"] = float(util.gpu), float(util.memory)
        memory = self.get("nvmlDeviceGetMemoryInfo", Memory())
        power = self.get("nvmlDeviceGetPowerUsage", C.c_uint())
        clock = self.get("nvmlDeviceGetClockInfo", C.c_uint(), 1)  # NVML_CLOCK_SM
        if memory is not None:
            values["device_memory_used_mib"] = memory.used / 2**20
        if power is not None:
            values["device_power_w"] = power.value / 1000
        if clock is not None:
            values["sm_clock_mhz"] = float(clock.value)
        return values

    def close(self):
        self.shutdown()


class Cli:
    def __init__(self, device):
        self.device = device

    def versions(self):
        raw = subprocess.check_output(["nvidia-smi", "--version"], text=True, timeout=2, stderr=subprocess.DEVNULL)
        result = {}
        for label, version in re.findall(r"(?im)^\s*(NVIDIA-SMI|NVML|DRIVER)\s+version\s*:\s*([\d.]+)\s*$", raw):
            key = {"NVIDIA-SMI": "nvidia_smi_version", "NVML": "nvml_version", "DRIVER": "driver_version"}[label.upper()]
            result[key] = public_version(version)
        return result

    def read(self):
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=" + ",".join(FIELDS), "--format=csv,noheader,nounits", "-i", self.device],
            stderr=subprocess.DEVNULL, text=True, timeout=QUERY_TIMEOUT,
        ).strip()
        parts = raw.split(",")
        if len(parts) != len(FIELDS) or "\n" in raw:
            raise ValueError("Unexpected field count")
        values = empty_metrics()
        for key, value in zip(FIELDS.values(), parts):
            try:
                values[key] = float(value)
            except ValueError:  # Unsupported optional counters stay missing.
                pass
        return values

    def close(self):
        pass


def worker(name):
    request = json.loads(sys.stdin.readline())
    if sys.platform.startswith("linux"):
        # Each reader has its own session for independent timeouts. Link it to
        # its supervisor as well so a hard benchmark deadline leaves no reader.
        death_signal = signal.SIGKILL
        if name == "cli":
            death_signal = signal.SIGTERM
            signal.signal(death_signal, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
        if C.CDLL(None).prctl(1, death_signal, 0, 0, 0) != 0 or os.getppid() != request["parent"]:
            return
    sensor = None
    versions = {}
    try:
        sensor = (Nvml if name == "nvml" else Cli)(request["device"])
        if request["versions"]:
            versions = sensor.versions()
    except (OSError, ValueError, RuntimeError, AttributeError, subprocess.SubprocessError):
        pass
    print(json.dumps(versions), flush=True)
    try:
        for command in sys.stdin:
            if command.strip() != "read":
                break
            start = time.perf_counter()
            status, values = "error", empty_metrics()
            try:
                if sensor is not None:
                    values = sensor.read()
                    for key, value in values.items():
                        if value is not None and (not math.isfinite(value) or value < 0
                                                   or key.endswith("_pct") and value > 100):
                            values[key] = None
                    status = "valid" if values["gpu_busy_pct"] is not None else "error"
            except subprocess.TimeoutExpired:
                status = "timeout"
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                pass
            print(json.dumps([status, start, time.perf_counter(), values], allow_nan=False), flush=True)
    finally:
        if sensor is not None:
            sensor.close()


class Reader:
    """A replaceable process isolates stalled NVML calls or nvidia-smi children."""

    def __init__(self, name, device):
        self.name, self.device = name, device
        self.proc = None

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def receive(self, deadline):
        if not select.select([self.proc.stdout], [], [], max(0, deadline - time.perf_counter()))[0]:
            raise TimeoutError
        line = self.proc.stdout.readline()
        if not line:
            raise BrokenPipeError
        return json.loads(line)

    def launch(self, versions):
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), self.name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, start_new_session=True,
        )
        self.send(json.dumps({"device": self.device, "versions": versions, "parent": os.getpid()}))

    def close(self, graceful=False):
        proc = self.proc
        if proc is None:
            return
        try:
            if graceful and proc.poll() is None:
                self.send("quit")
                proc.wait(timeout=0.2)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            # Kill the process group, including any outstanding CLI subprocess.
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.wait(timeout=0.5)
            proc.stdin.close()
            proc.stdout.close()
            self.proc = None


class Observers:
    def __init__(self, arm, uuid):
        names = {"A": ("cli",), "B": ("nvml",), "C": ("cli", "nvml")}
        if arm not in names:
            raise ValueError("Unknown observer arm")
        try:
            device = "GPU-" + str(UUID(str(uuid).removeprefix("GPU-")))
        except ValueError:
            raise ValueError("A physical CUDA GPU UUID is required") from None
        self.readers = {name: Reader(name, device) for name in names[arm]}
        self.records = {
            name: [{"slot": slot, "status": "skipped", "query_start_s": None, "query_end_s": None,
                    "values": empty_metrics()} for slot in range(SLOTS)] for name in self.readers
        }
        self.threads = []
        self.done = threading.Event()
        self.origin = None

    def restart(self, reader):
        try:
            reader.launch(versions=False)
            reader.receive(time.perf_counter() + STARTUP_TIMEOUT)
        except (OSError, ValueError):
            reader.close()

    def prepare(self):
        versions = dict.fromkeys(("binding_version", "nvml_version", "driver_version", "nvidia_smi_version"))
        if "nvml" in self.readers:
            versions["binding_version"] = BINDING
        deadline = time.perf_counter() + STARTUP_TIMEOUT
        for reader in self.readers.values():
            reader.launch(versions=True)
        for reader in self.readers.values():
            try:
                for key, value in reader.receive(deadline).items():
                    if key in versions and value is not None:
                        versions[key] = value
            except (OSError, ValueError):
                reader.close()
        return versions

    def observe(self, reader, origin):
        busy_until = origin
        for slot, record in enumerate(self.records[reader.name]):
            deadline = origin + slot * PERIOD
            if deadline < busy_until or time.perf_counter() >= deadline + PERIOD:
                continue
            if self.done.wait(max(0, deadline - time.perf_counter())):
                break
            if reader.proc is None:
                self.restart(reader)
            if reader.proc is None or self.done.is_set() or time.perf_counter() >= deadline + PERIOD:
                continue
            before = time.perf_counter()
            try:
                reader.send("read")
                status, first, last, values = reader.receive(before + QUERY_TIMEOUT)
                # A delayed supervisor may find a reply already queued after
                # the deadline. Judge it by the worker's monotonic timestamps.
                if last > before + QUERY_TIMEOUT:
                    status, values = "timeout", empty_metrics()
            except (OSError, ValueError) as exc:
                first, last, values = before, time.perf_counter(), empty_metrics()
                status = "timeout" if isinstance(exc, TimeoutError) else "error"
                reader.close()
                if not self.done.is_set() and slot + 1 < SLOTS:
                    self.restart(reader)
            record.update(status=status, query_start_s=first - origin, query_end_s=last - origin, values=values)
            busy_until = time.perf_counter()

    def start(self, start):
        self.origin = start
        for reader in self.readers.values():
            thread = threading.Thread(target=self.observe, args=(reader, start), daemon=True)
            self.threads.append(thread)
            thread.start()

    def finish(self, end):
        self.done.set()
        for thread in self.threads:
            thread.join(QUERY_TIMEOUT + STARTUP_TIMEOUT + 1)
            if thread.is_alive():
                raise RuntimeError("Observer supervisor failed to stop within its deadline")
        for slots in self.records.values():
            for record in slots:
                if record["query_end_s"] is not None and record["query_end_s"] > end - self.origin:
                    record["status"] = "crossing"
        return self.records

    def close(self):
        self.done.set()
        for thread in self.threads:
            thread.join(QUERY_TIMEOUT + STARTUP_TIMEOUT + 1)
        for reader in self.readers.values():
            reader.close(graceful=True)


if __name__ == "__main__":
    worker(sys.argv[1])
