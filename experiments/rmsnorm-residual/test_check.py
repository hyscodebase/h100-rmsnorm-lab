"""CPU checks for the residual-rounding and two-output verification contract."""
import unittest

import torch

import check


def normalize_unrounded(x, residual, weight):
    z = x.float() + residual.float()
    normalized = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + check.EPS) * weight.float()
    return z.to(torch.bfloat16), normalized.to(torch.bfloat16)


class ResidualChecks(unittest.TestCase):
    def test_explicit_reference_passes_all_fixtures(self):
        evidence = check.validate_backend(check.eager_reference, 2, 101, "cpu")
        self.assertTrue(evidence["passed"], evidence["failure"])
        self.assertEqual([f["fixture"] for f in evidence["fixtures"]], list(check.FIXTURES))
        for fixture in evidence["fixtures"]:
            self.assertIn("updated_sum", fixture)
            self.assertIn("normalized_sum", fixture)

    def test_frozen_ties_reject_unrounded_sum_beyond_generic_tolerance_check(self):
        inputs = check.rounding_fixture(2, "cpu")
        expected = check.oracle(*inputs)
        wrong = normalize_unrounded(*inputs)
        # The global tolerance even accepts this specific wrong answer.
        check.verify_outputs(wrong, expected, inputs)
        exact = expected[1][0, [0, 3]]
        rounded = exact.bfloat16()
        self.assertEqual(rounded.tolist(), [0.9921875, 1.765625])
        self.assertEqual(wrong[1][0, [0, 3]].tolist(), [0.99609375, 1.7578125])
        bits = rounded.view(torch.int16)
        for offset in (-1, 1):
            adjacent = (bits + offset).view(torch.bfloat16).double()
            distance = (rounded.double() - adjacent).abs()
            midpoint = (rounded.double() + adjacent) / 2
            self.assertTrue(((exact - midpoint).abs() / distance > 0.22).all().item())
        evidence = check.validate_backend(normalize_unrounded, 2, 101, "cpu")
        ties = next(f for f in evidence["fixtures"] if f["fixture"] == "bf16_ties")
        self.assertFalse(evidence["passed"])
        self.assertEqual(ties["reason"], "rounding_sensitive_bits")

    def test_requires_both_outputs_and_rejects_input_mutation(self):
        def overwrites_input(x, residual, weight):
            outputs = check.eager_reference(x, residual, weight)
            x.fill_(7)
            return outputs
        evidence = check.validate_backend(overwrites_input, 1, 101, "cpu")
        self.assertEqual(evidence["failure"]["reason"], "mutated_inputs")
        inputs = check.rounding_fixture(1, "cpu")
        expected = check.oracle(*inputs)
        updated, y = check.eager_reference(*inputs)
        with self.assertRaisesRegex(AssertionError, "expected_two_outputs"):
            check.verify_outputs((updated,), expected, inputs)
        with self.assertRaisesRegex(AssertionError, "overlapping_output_storage"):
            check.verify_outputs((updated, updated), expected, inputs)
        # Distinct slices from a pooled output allocation remain legal.
        pool = torch.stack((updated, y))
        check.verify_outputs(tuple(pool.unbind(0)), expected, inputs)


if __name__ == "__main__":
    unittest.main()
