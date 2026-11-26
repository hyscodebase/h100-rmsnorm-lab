"""CPU-only regression tests for residual checkpoints, including honest exclusions."""
import copy
import hashlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import lab
import residual_plan as residual

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("residual_benchmark", ROOT / residual.SOURCE)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
REVISION = "1" * 40


def result_for(plan, sources, seed=101, rows=1024, reject_candidate=False):
    diag = {"max_abs_error": 0.01, "updated_sum": 25.5, "normalized_sum": -16.25}
    method = {"rounds": 21, "ops_per_graph": 32, "warmup_calls": 5,
              "timing": "CUDA events around one graph replay; microseconds per two-output call",
              "cache": "reused input addresses without an explicit L2 flush",
              "outputs": "all 32 updated and normalized tensor pairs remain live during graph timing",
              "timing_started_utc": "2026-11-26T08:01:00Z", "timing_phase_wall_s": 1.0}
    results = {}
    for i, name in enumerate(residual.BACKENDS):
        checks = {"rows": rows, "width": 4096, "seed": seed, "eps": 1e-6, "rtol": 0.005, "atol": 1e-6,
                  "fixtures": [{"fixture": f, "passed": True, **diag} for f in residual.FIXTURES],
                  "passed": True, "failure": None}
        if name == residual.CANDIDATE and reject_candidate:
            failure = {"fixture": "bf16_ties", "reason": "rounding_sensitive_bits"}
            checks.update(passed=False, failure=failure)
            checks["fixtures"][residual.FIXTURES.index(failure["fixture"])] = {**failure, "passed": False}
            results[name] = {"status": "excluded", "excluded_at": "numerical_fixtures", "correctness": checks}
        elif name == residual.COMPILED[-1]:
            results[name] = {"status": "excluded", "excluded_at": "compile",
                             "error": {"exception_types": ["RuntimeError"], "detail": "Unavailable mode"}}
        else:
            results[name] = {"status": "measured", "correctness": checks, "replay_checks": [diag.copy() for _ in range(32)],
                             "timing": benchmark.latency([0.064 + i * 0.01] * 21)}
    method["round_order"] = benchmark.schedule([n for n in results if results[n]["status"] == "measured"], seed)
    return {"seed": seed, "rows": rows, "width": 4096, "dtype": "bfloat16", "eps": 1e-6, "method": method,
            "started_utc": "2026-11-26T08:00:00Z", "finished_utc": "2026-11-26T08:01:01Z", "wall_elapsed_s": 61.0,
            "code_revision": REVISION, "source_sha256": {Path(k).name: v for k, v in sources.items()},
            "backends": results, "comparison": benchmark.comparison(results),
            "environment": {"gpu": "NVIDIA H100", "compute_capability": [9, 0], "driver": "550",
                            "cuda_runtime": "12.8", "torch": "2.7.1", "triton": "3.3.1", "python": "3.12",
                            "sm_count": 132, "total_memory_bytes": 80 * 1024**3}}


class ResidualCheckpoint(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.state = self.root / "state"
        root_patch = patch.object(lab, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        for name in residual.SOURCES:
            dest = self.root / name
            dest.parent.mkdir(exist_ok=True, parents=True)
            dest.write_bytes((ROOT / name).read_bytes())
        self.plan_path = ROOT / "plans/residual-rmsnorm.json"
        self.plan = lab.read_json(self.plan_path)
        with redirect_stdout(io.StringIO()):
            lab.initialize(self.state, self.plan_path)
        self.sources = lab.read_json(self.state / "manifest.json")["sources"]

    def test_target_first_and_pinned_inputs(self):
        tasks = list(lab.tasks(self.plan))
        self.assertEqual(len({name for name, _ in tasks}), 12)
        self.assertEqual([shape for _, shape in tasks[:3]], [(101, 1024), (102, 1024), (103, 1024)])
        self.assertEqual(set(self.sources), set(residual.SOURCES))
        self.assertEqual(lab.command(Path("/fake"), tasks[0][1], self.plan),
                         ["/fake", "residual", "--rows", "1024", "--seed", "101"])
        self.assertEqual(lab.command(None, tasks[0][1], self.plan)[1], str(self.root / residual.SOURCE))
        with (self.root / residual.SOURCES[0]).open("a") as f:
            f.write("\nchanged plan")
        with self.assertRaisesRegex(ValueError, "source or saved plan changed"):
            lab.load_state(self.state)

    def test_cross_checks_evidence_and_reported_winner(self):
        original = result_for(self.plan, self.sources)
        residual.validate_result(original, (101, 1024), self.plan)
        for key in ("backends", "round_order", "summary", "replay", "fixtures", "missing_denominator", "comparison", "wall"):
            with self.subTest(key=key):
                bad = copy.deepcopy(original)
                candidate = bad["backends"][residual.CANDIDATE]
                if key == "backends":
                    bad["backends"].pop(residual.COMPILED[-1])
                elif key == "round_order":
                    bad["method"]["round_order"][0].reverse()
                elif key == "summary":
                    candidate["timing"]["median_us"] += 1
                elif key == "replay":
                    candidate["replay_checks"].pop()
                elif key == "fixtures":
                    candidate["correctness"]["fixtures"][0]["updated_sum"] = float("nan")
                elif key == "missing_denominator":
                    candidate["correctness"]["fixtures"].pop()
                elif key == "comparison":
                    bad["comparison"]["fastest_correct_compiled"] = residual.EAGER
                else:
                    bad["method"]["timing_phase_wall_s"] = 0.00001
                with self.assertRaises(ValueError):
                    residual.validate_result(bad, (101, 1024), self.plan)

    def test_ineligible_candidate_is_saved_and_hardware_mismatch_retries_next_cell(self):
        calls = []
        wrong_device = True
        def runner(cmd, **kwargs):
            if cmd[0] == "git":
                # Simulate the user's frozen commit and blobs without launching Git or a GPU.
                value = b"commit\n" if "-t" in cmd else (self.root / cmd[-1].split(":", 1)[1]).read_bytes()
                return subprocess.CompletedProcess(cmd, 0, stdout=value)
            seed = int(cmd[cmd.index("--seed") + 1])
            calls.append(seed)
            result = result_for(self.plan, self.sources, seed=seed, reject_candidate=True)
            if seed == 102 and wrong_device:
                result["environment"]["sm_count"] -= 1
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(result))

        with patch.object(subprocess, "run", side_effect=runner), redirect_stdout(io.StringIO()):
            lab.execute(self.state, Path("/fake"), 1)
            with self.assertRaisesRegex(ValueError, "GPU environment changed"):
                lab.execute(self.state, Path("/fake"), 1)
            wrong_device = False
            lab.execute(self.state, Path("/fake"), 1)
        self.assertEqual(calls, [101, 102, 102])
        self.assertEqual(len(list((self.state / "tasks").glob("*.json"))), 2)
        self.assertEqual(len(list((self.state / "attempts").glob("*.json"))), 3)
        first = lab.read_json(self.state / "tasks/s101-r1024-w4096.json")["result"]
        self.assertFalse(first["comparison"]["eligible"])
        self.assertEqual(first["backends"][residual.CANDIDATE]["correctness"]["failure"]["reason"], "rounding_sensitive_bits")

    def test_forged_source_and_wrong_historical_blob_are_rejected(self):
        sample = result_for(self.plan, self.sources)
        sample["source_sha256"]["check.py"] = hashlib.sha256(b"unrelated").hexdigest()
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            residual.verify_sources(sample, self.root, self.sources)
        sample = result_for(self.plan, self.sources)
        def wrong_blob(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=b"commit\n" if "-t" in cmd else b"wrong bytes")
        with patch.object(subprocess, "run", side_effect=wrong_blob), self.assertRaisesRegex(ValueError, "Git commit"):
            residual.verify_sources(sample, self.root, self.sources)


if __name__ == "__main__":
    unittest.main()
