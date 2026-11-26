"""Convert the documented five- or ten-column UTC nvidia-smi stream to GEMM monitor JSONL."""

import csv
import json
import math
import sys
from datetime import datetime, timezone


METRICS = ("util_pct", "used_mib", "total_mib", "power_w", "temperature_c",
           "sm_clock_mhz", "memory_clock_mhz", "power_limit_w", "memory_busy_pct")


def convert(fields):
    if len(fields) not in (5, 10):
        return None
    try:
        stamp = datetime.fromisoformat(fields[0].strip().replace("/", "-")).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    values = dict.fromkeys(METRICS)
    for name, raw in zip(METRICS, fields[1:]):
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0:
            values[name] = value
    if all(value is None for value in values.values()):
        return None
    if values["util_pct"] is not None and values["util_pct"] > 100:
        return None
    if values["memory_busy_pct"] is not None and values["memory_busy_pct"] > 100:
        values["memory_busy_pct"] = None
    return {"timestamp_utc": stamp.isoformat(), **values}


if __name__ == "__main__":
    for fields in csv.reader(sys.stdin):
        sample = convert(fields)
        if sample is not None:
            print(json.dumps(sample, allow_nan=False), flush=True)
