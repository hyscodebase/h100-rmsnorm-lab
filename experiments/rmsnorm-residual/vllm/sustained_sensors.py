"""Sample the measured CUDA device directly with NVML on a shared monotonic clock."""
import ctypes as C
import math
import statistics
import threading
import time
from datetime import datetime, timezone

PERIOD_S = 1.0
FIELDS = ("gpu_busy_pct", "memory_busy_pct", "vram_used_mib", "vram_total_mib",
          "power_w", "power_limit_w", "temperature_c", "sm_clock_mhz", "memory_clock_mhz")


class Utilization(C.Structure):
    _fields_ = [("gpu", C.c_uint), ("memory", C.c_uint)]


class Memory(C.Structure):
    _fields_ = [("total", C.c_ulonglong), ("free", C.c_ulonglong), ("used", C.c_ulonglong)]


class Nvml:
    def __init__(self, cuda_uuid):
        self.lib = C.CDLL("libnvidia-ml.so.1")
        init = self.function("nvmlInit_v2")
        if init():
            raise RuntimeError("NVML initialization failed.")
        self.handle = C.c_void_p()
        if cuda_uuid:
            uuid = "GPU-" + str(cuda_uuid).removeprefix("GPU-")
            by_uuid = self.function("nvmlDeviceGetHandleByUUID", C.c_char_p, C.POINTER(C.c_void_p))
            status = by_uuid(uuid.encode("ascii"), C.byref(self.handle))
        else:
            count = C.c_uint()
            get_count = self.function("nvmlDeviceGetCount_v2", C.POINTER(C.c_uint))
            by_index = self.function("nvmlDeviceGetHandleByIndex_v2", C.c_uint, C.POINTER(C.c_void_p))
            status = by_index(0, C.byref(self.handle)) if get_count(C.byref(count)) == 0 and count.value == 1 else 1
        if status:
            self.close()
            raise RuntimeError("CUDA device could not be matched to NVML.")

    def function(self, name, *arguments):
        fn = getattr(self.lib, name)
        fn.argtypes = arguments
        fn.restype = C.c_int
        return fn

    def read(self):
        values, errors = dict.fromkeys(FIELDS), {}

        def query(name, output, *enums):
            key = name if not enums else f"{name}:{enums[0]}"
            try:
                fn = self.function(name, C.c_void_p, *([C.c_uint] * len(enums)), C.POINTER(type(output)))
            except AttributeError:
                errors[key] = "unavailable"
                return None
            status = fn(self.handle, *enums, C.byref(output))
            if status:
                errors[key] = int(status)
                return None
            return output

        rates = query("nvmlDeviceGetUtilizationRates", Utilization())
        if rates is not None:
            values["gpu_busy_pct"], values["memory_busy_pct"] = rates.gpu, rates.memory
        memory = query("nvmlDeviceGetMemoryInfo", Memory())
        if memory is not None:
            values["vram_used_mib"], values["vram_total_mib"] = memory.used / 2**20, memory.total / 2**20
        for field, name, unit, enums in (
            ("power_w", "nvmlDeviceGetPowerUsage", 1000, ()),
            ("power_limit_w", "nvmlDeviceGetEnforcedPowerLimit", 1000, ()),
            ("temperature_c", "nvmlDeviceGetTemperature", 1, (0,)),
            ("sm_clock_mhz", "nvmlDeviceGetClockInfo", 1, (1,)),
            ("memory_clock_mhz", "nvmlDeviceGetClockInfo", 1, (2,)),
        ):
            reading = query(name, C.c_uint(), *enums)
            if reading is not None:
                values[field] = reading.value / unit
        return values, errors

    def close(self):
        self.function("nvmlShutdown")()


class Observer:
    def __init__(self, cuda_uuid, origin):
        self.origin = origin
        self.stop = threading.Event()
        self.records = []
        self.error = None
        try:
            self.sensor = Nvml(cuda_uuid)
        except (OSError, AttributeError, RuntimeError, ValueError) as error:
            self.error = type(error).__name__
            self.thread = None
        else:
            self.thread = threading.Thread(target=self.sample, daemon=True)
            self.thread.start()

    def sample(self):
        slot = 0
        while not self.stop.wait(max(0, self.origin + slot * PERIOD_S - time.perf_counter())):
            before = time.perf_counter() - self.origin
            for missed in range(slot, math.floor(before / PERIOD_S)):
                self.records.append({"slot": missed, "scheduled_s": missed * PERIOD_S, "status": "missed"})
                slot = missed + 1
            utc = datetime.now(timezone.utc).isoformat()
            try:
                values, errors = self.sensor.read()
            except Exception as error:
                values, errors = dict.fromkeys(FIELDS), {"read": type(error).__name__}
            self.records.append({"slot": slot, "scheduled_s": slot * PERIOD_S,
                                 "query_start_s": before, "query_end_s": time.perf_counter() - self.origin,
                                 "utc": utc, "status": "sampled", "values": values, "errors": errors})
            slot += 1

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.error = "query_incomplete_at_shutdown"
            else:
                self.sensor.close()
        return {"source": "nvml-ctypes through the local NVIDIA driver", "period_s": PERIOD_S,
                "observer_error": self.error, "slots": self.records.copy()}


def associate(slots, blocks):
    for slot in slots:
        before = slot.get("query_start_s", slot["scheduled_s"])
        after = slot.get("query_end_s", before)
        slot["block"] = next((block["id"] for block in blocks if block["start_s"] <= before <= after <= block["end_s"]),
                             None)


def summarize_load(blocks, slots, backends, *, group_by="backend"):
    result = {}
    for name in backends:
        values = []
        gaps = []
        for block in (block for block in blocks if block[group_by] == name):
            matched = [s for s in slots if s["block"] == block["id"] and s["status"] == "sampled"
                       and s["values"]["gpu_busy_pct"] is not None]
            values.extend(s["values"] for s in matched)
            timestamps = [block["start_s"], *((s["query_start_s"] + s["query_end_s"]) / 2 for s in matched),
                          block["end_s"]]
            gaps.extend(after - before for before, after in zip(timestamps, timestamps[1:]))
        fields = {}
        for field in FIELDS:
            samples = [row[field] for row in values if row[field] is not None]
            fields[field] = {"count": len(samples), "median": statistics.median(samples) if samples else None,
                             "peak": max(samples) if samples else None}
        gap = max(gaps, default=None)
        activity = fields["gpu_busy_pct"]
        result[name] = {"metrics": fields, "maximum_gap_s": gap,
                        "high_load": activity["count"] >= 12 and gap is not None and gap <= 2 and activity["median"] >= 80}
    return result
