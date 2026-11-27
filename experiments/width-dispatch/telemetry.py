"""One long-lived, phase-bounded nvidia-smi poller for sustained GPU windows."""
import csv
import io
import math
import subprocess
import threading
import time
from datetime import datetime, timezone


FIELDS = "timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def number(value):
    value = value.strip()
    if value.lower() in {"n/a", "[n/a]", "not supported", "[not supported]"}:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def query_time(value):
    value = value.strip()
    parsed = None
    for pattern in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(value, pattern)
            break
        except ValueError:
            continue
    if parsed is None:
        raise ValueError(f"Unrecognized nvidia-smi timestamp: {value}")
    utc = parsed.astimezone(timezone.utc)
    return utc.isoformat(), int(utc.timestamp() * 1_000_000_000)


class InWindowSampler:
    """Start one persistent nvidia-smi loop only for one active replay phase."""

    def __init__(self, interval_seconds=0.25, device_id=None):
        self.interval_seconds = float(interval_seconds)
        self.device_id = device_id
        self.samples = []
        self.error = None
        self.process = None
        self.thread = None
        self._lock = threading.Lock()
        milliseconds = round(self.interval_seconds * 1000)
        self.cadence_arg = (f"--loop={milliseconds // 1000}"
                            if milliseconds >= 1000 and milliseconds % 1000 == 0
                            else f"--loop-ms={milliseconds}")
        self.process_description = (
            "configured phase-bounded nvidia-smi query; "
            f"{self.cadence_arg}"
        )

    def start(self):
        if self.device_id is None:
            raise RuntimeError("No verified CUDA-visible nvidia-smi device selector.")
        command = ["nvidia-smi", "--query-gpu=" + FIELDS,
                   "--format=csv,noheader,nounits", f"--id={self.device_id}", self.cadence_arg]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, bufsize=1)
        if self.process.poll() is not None:
            raise RuntimeError("nvidia-smi loop exited before telemetry reader started.")
        self.thread = threading.Thread(target=self._read_loop,
                                       name="gpu-in-window-telemetry", daemon=True)
        self.thread.start()

    def _read_loop(self):
        try:
            for line in self.process.stdout:
                arrival_utc = utc_now()
                arrival_ns = time.perf_counter_ns()
                row = next(csv.reader(io.StringIO(line)), [])
                if len(row) != 6:
                    sample = {"arrival_utc": arrival_utc,
                              "arrival_monotonic_ns": arrival_ns,
                              "error": f"Expected six fields, received {len(row)}.",
                              "raw_line": line.rstrip("\r\n")}
                else:
                    try:
                        gpu, memory, used, power, temperature = map(number, row[1:])
                        query_utc, query_unix_ns = query_time(row[0])
                        sample = {"arrival_utc": arrival_utc,
                                  "arrival_monotonic_ns": arrival_ns,
                                  "nvidia_smi_timestamp": row[0].strip(),
                                  "query_utc": query_utc,
                                  "query_unix_ns": query_unix_ns,
                                  "gpu_busy_percent": gpu,
                                  "memory_busy_percent": memory,
                                  "memory_used_mib": used,
                                  "power_w": power,
                                  "temperature_c": temperature}
                    except ValueError as exc:
                        sample = {"arrival_utc": arrival_utc,
                                  "arrival_monotonic_ns": arrival_ns,
                                  "error": str(exc), "raw_line": line.rstrip("\r\n")}
                with self._lock:
                    sample["sample_index"] = len(self.samples)
                    self.samples.append(sample)
        except Exception as exc:
            self.error = f"Telemetry reader failed: {type(exc).__name__}: {exc}"

    def stop(self):
        if self.process is None:
            return []
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        if self.thread is not None:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                self.error = "Telemetry reader did not stop promptly."
        with self._lock:
            return list(self.samples)
