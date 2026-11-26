"""CPU-only contract checks for the precision probe; the registered run requires an H100."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe


class PrecisionProbe(unittest.TestCase):
    def test_fresh_processes_really_load_the_requested_inductor_config(self):
        code = ("import json,sys; import probe; "
                "print(json.dumps(probe.effective_config(sys.argv[1]=='1')))")
        digests = []
        for flag in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                env = probe.child_env(flag, Path(temp))
                self.assertEqual(env[probe.FLAG], str(int(flag)))
                raw = subprocess.check_output([sys.executable, "-B", "-c", code, str(int(flag))],
                                              env=env, cwd=HERE, text=True)
                actual, digest = json.loads(raw)
                self.assertIs(actual, flag)
                self.assertEqual(len(digest), 64)
                digests.append(digest)
        self.assertEqual(*digests)

    def test_exact_bits_control_and_decisions(self):
        import torch
        sys.path.insert(0, str(HERE.parent))
        import check
        def wrong_denominator(x, residual, weight):
            raw = x.float() + residual.float()
            updated = raw.bfloat16()
            return updated, (updated.float() * torch.rsqrt(raw.square().mean(-1, keepdim=True) + check.EPS)
                             * weight.float()).bfloat16()

        def cpu_arm(flag):
            kwargs = {"requested": flag, "effective": flag, "environment": {"device": "cpu"}, "source_sha256": {"test": "same"}}
            gen = torch.Generator().manual_seed(probe.SEED)
            kwargs["cases"] = {
                name: probe.inspect_fixture(torch, check, name, check.make_inputs(name, 2, gen, "cpu"),
                                            {"eager": check.eager_reference, "fused": check.eager_reference,
                                             "compiled": check.eager_reference if flag else wrong_denominator},
                                            ((0, 0), (0, 3), (1, 4092), (1, 4095))) for name in probe.CASES
            }
            return kwargs
        off, on = cpu_arm(False), cpu_arm(True)
        cell = off["cases"]["bf16_denominator_tie"]
        self.assertEqual(cell["backends"]["compiled"]["normalized"]["values"][:2], [0.99609375, 1.9921875])
        self.assertEqual(cell["backends"]["compiled"]["normalized"]["bits_hex"][:2], ["0x3f7f", "0x3fff"])
        self.assertEqual(cell["oracle"]["normalized"]["bits_hex"][:2], ["0x3f80", "0x4000"])
        self.assertEqual(probe.decide(off, on), "restored_on_these_probes")
        invalid = copy.deepcopy(on)
        invalid["cases"]["random"]["input_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Inputs or FP64"):
            probe.decide(off, invalid)
        failed = copy.deepcopy(on)
        failed["cases"]["bf16_ties"]["backends"]["fused"]["passed"] = False
        self.assertEqual(probe.decide(off, failed), "control_failed")
        on["cases"]["bf16_denominator_tie"]["backends"]["compiled"]["passed"] = False
        self.assertEqual(probe.decide(off, on), "on_failed")


if __name__ == "__main__":
    unittest.main()
