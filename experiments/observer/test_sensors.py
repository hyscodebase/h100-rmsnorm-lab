"""CPU-only sensor tests with a mock NVML ABI and a temporary nvidia-smi executable."""

import ctypes as C
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import sensors as s


DEVICE = "GPU-00000000-0000-4000-8000-000000000001"


class NvmlTest(unittest.TestCase):
    def test_abi_device_selection_units_partial_fields_and_public_versions(self):
        lib = MagicMock()
        lib.nvmlInit_v2.return_value = 0
        lib.nvmlDeviceGetHandleByUUID.return_value = 0

        def utilization(_, pointer):
            pointer._obj.gpu, pointer._obj.memory = 83, 27
            return 0

        def memory(_, pointer):
            pointer._obj.used = 3 * 2**20 + 2**19
            return 0

        def power(_, pointer):
            pointer._obj.value = 123450
            return 0

        def version(buf, length):
            buf.value = b"12.550.144.03"
            return 0

        lib.nvmlDeviceGetUtilizationRates.side_effect = utilization
        lib.nvmlDeviceGetMemoryInfo.side_effect = memory
        lib.nvmlDeviceGetPowerUsage.side_effect = power
        lib.nvmlDeviceGetClockInfo.return_value = 3  # not supported
        lib.nvmlSystemGetNVMLVersion.side_effect = version
        lib.nvmlSystemGetDriverVersion.return_value = 3
        with patch.object(C, "CDLL", return_value=lib):
            sensor = s.Nvml(DEVICE)
            values = sensor.read()
            self.assertEqual(lib.nvmlDeviceGetHandleByUUID.call_args.args[0], DEVICE.encode())
            self.assertEqual(lib.nvmlDeviceGetClockInfo.call_args.args[1], 1)
            self.assertEqual(values, {"gpu_busy_pct": 83, "memory_busy_pct": 27, "device_memory_used_mib": 3.5,
                                      "device_power_w": 123.45, "sm_clock_mhz": None})
            self.assertEqual(sensor.versions(), {"binding_version": s.BINDING, "nvml_version": "12.550.144.03"})
            sensor.close()
            lib.nvmlShutdown.assert_called_once()

    def test_rejects_mig_and_nonpublic_version(self):
        with self.assertRaisesRegex(ValueError, "physical"):
            s.Observers("C", "MIG-" + DEVICE[4:])
        self.assertIsNone(s.public_version("550.1 test-private"))


class ScriptedReader:
    """Same scheduling interface as a child process; no graphics library or binary."""
    def __init__(self, name, device):
        self.name = name
        self.proc = None
        self.reading = False
        self.calls = 0
        self.launches = 0

    def launch(self, versions):
        self.proc = object()
        self.reading = False
        self.launches += 1

    def send(self, command):
        self.reading = True

    def receive(self, deadline):
        if not self.reading:
            return {"binding_version": s.BINDING} if self.name == "nvml" else {}
        self.calls += 1
        start = time.perf_counter()
        if self.name == "nvml" and self.calls == 1:
            time.sleep(max(0, deadline - start))
            raise TimeoutError
        time.sleep(0.01)
        return "valid", start, time.perf_counter(), {**s.empty_metrics(), "gpu_busy_pct": 98}

    def close(self, graceful=False):
        self.proc = None


class SchedulerTest(unittest.TestCase):
    def test_independent_deadlines_skip_busy_slots_and_replace_a_timed_out_worker(self):
        with (patch.object(s, "Reader", ScriptedReader), patch.object(s, "SLOTS", 7),
              patch.object(s, "PERIOD", 0.1), patch.object(s, "QUERY_TIMEOUT", 0.23)):
            monitors = s.Observers("C", DEVICE)
            monitors.prepare()
            start = time.perf_counter()
            monitors.start(start)
            try:
                for thread in monitors.threads:
                    thread.join(timeout=2)
                result = monitors.finish(time.perf_counter())
                self.assertEqual([row["slot"] for row in result["nvml"]], list(range(7)))
                self.assertEqual([r["status"] for r in result["nvml"][:3]], ["timeout", "skipped", "skipped"])
                self.assertIn("valid", [r["status"] for r in result["nvml"][3:]])
                self.assertEqual(monitors.readers["nvml"].launches, 2)
                self.assertGreater(sum(r["status"] == "valid" for r in result["cli"]), 4)
                self.assertNotIn(DEVICE, json.dumps(result))
            finally:
                monitors.close()

    def test_preflight_finish_marks_crossing_and_skips_future_slots(self):
        with patch.object(s, "Reader", ScriptedReader), patch.object(s, "SLOTS", 45):
            monitors = s.Observers("A", DEVICE)
            monitors.prepare()
            start = time.perf_counter()
            monitors.start(start)
            try:
                deadline = start + 1
                while not monitors.readers["cli"].calls and time.perf_counter() < deadline:
                    time.sleep(0.001)
                result = monitors.finish(start + 0.001)["cli"]
                self.assertEqual(len(result), 45)
                self.assertEqual(result[0]["status"], "crossing")
                self.assertTrue(all(r["query_start_s"] is None and r["status"] == "skipped" for r in result[1:]))
            finally:
                monitors.close()

    def test_real_worker_ipc_with_fake_cli_and_unsupported_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "nvidia-smi"
            fake.write_text(f"#!{sys.executable}\nimport sys\n"
                            "if '--version' in sys.argv:\n"
                            "    print('NVIDIA-SMI version  : 550.144.03')\n"
                            "else:\n"
                            f"    assert sys.argv[-2:] == ['-i', '{DEVICE}']\n"
                            "    print('98, 30, 1000, N/A, 1380')\n")
            fake.chmod(0o755)
            with patch.dict(os.environ, PATH=directory + os.pathsep + os.environ["PATH"]), patch.object(s, "SLOTS", 1):
                monitors = s.Observers("A", DEVICE)
                try:
                    metadata = monitors.prepare()
                    self.assertEqual(metadata["nvidia_smi_version"], "550.144.03")
                    start = time.perf_counter()
                    monitors.start(start)
                    monitors.threads[0].join(timeout=2)
                    record = monitors.finish(time.perf_counter())["cli"][0]
                    self.assertEqual((record["status"], record["values"]["gpu_busy_pct"]), ("valid", 98))
                    self.assertIsNone(record["values"]["device_power_w"])
                    self.assertNotIn(DEVICE, json.dumps([metadata, record]))
                finally:
                    monitors.close()
                self.assertIsNone(monitors.readers["cli"].proc)

    def test_stalled_reader_process_is_reaped_by_group_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "nvidia-smi"
            fake.write_text(f"#!{sys.executable}\nimport sys, time\n"
                            "if '--version' not in sys.argv: time.sleep(20)\n")
            fake.chmod(0o755)
            with (patch.dict(os.environ, PATH=directory + os.pathsep + os.environ["PATH"]),
                  patch.object(s, "SLOTS", 1), patch.object(s, "QUERY_TIMEOUT", 0.25)):
                monitors = s.Observers("A", DEVICE)
                try:
                    monitors.prepare()
                    worker = monitors.readers["cli"].proc
                    start = time.perf_counter()
                    monitors.start(start)
                    monitors.threads[0].join(timeout=2)
                    result = monitors.finish(time.perf_counter())["cli"][0]
                    self.assertEqual(result["status"], "timeout")
                    self.assertGreaterEqual(result["query_end_s"] - result["query_start_s"], 0.24)
                    self.assertIsNotNone(worker.poll())
                    self.assertEqual(result["values"], s.empty_metrics())
                finally:
                    monitors.close()


if __name__ == "__main__":
    unittest.main()
