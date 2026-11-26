"""CPU-only checks of observer workload routing, accounting and preflight output."""
import contextlib
import importlib.util
import io
import itertools
import json
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

HERE = Path(__file__).resolve().parent


def mock_module():
    torch = MagicMock()
    torch.inference_mode.return_value = lambda f: f
    spec = importlib.util.spec_from_file_location("observer_benchmark", HERE / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch, "sensors": MagicMock()}):
        spec.loader.exec_module(module)
    return module


class ObserverBenchmark(unittest.TestCase):
    def setUp(self):
        self.bench = mock_module()

    def test_cuda_uuid_mapping_requires_exact_device(self):
        u = uuid.UUID(int=17)
        props = SimpleNamespace(name="NVIDIA H100", major=9, minor=0, uuid=u)
        self.bench.torch.cuda.get_device_properties.return_value = props
        with patch.object(self.bench.baseline, "smi", return_value=f"GPU-{uuid.UUID(int=2)}\nGPU-{u}"):
            self.assertEqual(self.bench.device_identity(), (props, f"GPU-{u}"))
        with patch.object(self.bench.baseline, "smi", return_value=f"GPU-{uuid.UUID(int=2)}"):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                self.bench.device_identity()
        props.uuid = None
        with patch.object(self.bench, "cuda_driver_uuid", return_value=str(u)), \
             patch.object(self.bench.baseline, "smi", return_value=f"GPU-{u}"):
            self.assertEqual(self.bench.device_identity(), (props, f"GPU-{u}"))

    def test_preflight_only_emits_public_sample_values(self):
        missing = {"status": "timeout", "query_start_s": 0, "query_end_s": 1}
        valid = {"status": "valid", "query_start_s": .1, "query_end_s": .2,
                 "values": {"gpu_busy_pct": 0, "extra_identifier": "private"}}
        unit = MagicMock()
        unit.finish.return_value = {"cli": [valid], "nvml": [missing]}
        info = {"gpu": "H100", "observer_versions": {"driver_version": "550.1"}}
        uid = str(uuid.UUID(int=19))
        with patch.object(self.bench, "device_identity", return_value=(object(), uid)), \
             patch.object(self.bench, "Observers", return_value=unit), \
             patch.object(self.bench, "environment", return_value=info), \
             patch.object(self.bench, "source_revision", return_value="1" * 40), \
             patch.object(self.bench.time, "sleep"), \
             patch.object(self.bench.time, "perf_counter", side_effect=[100.0, 102.5]), \
             patch.object(sys, "argv", ["benchmark.py", "--preflight"]), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.bench.main(), 1)
        data = json.loads(output.getvalue())
        self.assertTrue(data["physical_uuid_matched"])
        self.assertTrue(data["checks"]["cli"]["passed"])
        self.assertFalse(data["checks"]["nvml"]["passed"])
        self.assertEqual(data["environment"], info)
        self.assertEqual(data["code_revision"], "1" * 40)
        self.assertNotIn("private", output.getvalue())
        self.assertNotIn(uid, output.getvalue())
        self.bench.torch.mm.assert_not_called()
        unit.close.assert_called_once()

    def test_sustained_work_uses_shared_correctness_and_full_wall_denominator(self):
        module = self.bench
        module.torch.cuda.mem_get_info.return_value = (80 * 1024**3, 80 * 1024**3)
        module.torch.cuda.Event.return_value.elapsed_time.return_value = 50.0
        unit = MagicMock()
        unit.finish.return_value = {"nvml": []}
        with patch.object(module.time, "perf_counter", side_effect=itertools.count()), \
             patch.object(module.baseline, "check_output", return_value={"passed": True}) as check:
            data = module.measure(55, "B", unit)
        check.assert_called_once()
        self.assertEqual(data["observers"], {"nvml": []})
        self.assertEqual(data["seed"], 55)
        self.assertEqual((data["workload"]["m"], data["workload"]["k"], data["workload"]["n"]), (8192, 4096, 11008))
        tm = data["timing"]
        self.assertGreaterEqual(tm["wall_elapsed_s"], 45)
        self.assertEqual(tm["wall_elapsed_s"], unit.finish.call_args.args[0] - unit.start.call_args.args[0])
        self.assertEqual(tm["iterations"], sum(b["iterations"] for b in tm["batches"]))
        self.assertAlmostEqual(tm["wall_tflops"], 2 * 8192 * 4096 * 11008 * tm["iterations"] / tm["wall_elapsed_s"] / 1e12)


if __name__ == "__main__":
    unittest.main()
