"""CPU-only checks for raw timing units, live outputs and compiler selection."""

import unittest

import benchmark


class BenchmarkTest(unittest.TestCase):
    def test_event_units_and_shuffled_round_contract(self):
        names = [benchmark.EAGER, benchmark.CANDIDATE, *benchmark.COMPILED]
        order = benchmark.schedule(names, 101)
        self.assertEqual(order, benchmark.schedule(names, 101))
        self.assertEqual(len(order), 21)
        self.assertTrue(all(sorted(round_) == sorted(names) for round_ in order))
        self.assertGreater(len(set(map(tuple, order))), 1)
        timing = benchmark.latency([ms / 1000 * 32 for ms in range(1, 22)])
        self.assertEqual((timing["q1_us"], timing["median_us"], timing["q3_us"]), (6, 11, 16))
        with self.assertRaises(ValueError):
            benchmark.latency([0]*21)

    def test_fastest_correct_comparator_and_required_eager_candidate(self):
        result = {name: {"status": "excluded"} for name in (benchmark.EAGER, benchmark.CANDIDATE, *benchmark.COMPILED)}
        def measured(us):
            return {"status": "measured", "timing": {"median_us": us}}
        explicit, _, native, native_tuned = benchmark.COMPILED
        for name, us in ((benchmark.EAGER, 10), (benchmark.CANDIDATE, 3), (native, 4), (native_tuned, 3.5)):
            result[name] = measured(us)
        choice = benchmark.comparison(result)
        self.assertTrue(choice["eligible"] and choice["native_only_compiled_reference"])
        self.assertEqual(choice["fastest_correct_compiled"], native_tuned)
        result[explicit] = measured(3.25)
        choice = benchmark.comparison(result)
        self.assertEqual(choice["fastest_correct_compiled"], explicit)
        self.assertFalse(choice["native_only_compiled_reference"])
        result[benchmark.EAGER] = {"status": "excluded"}
        self.assertFalse(benchmark.comparison(result)["eligible"])
        self.assertIsNone(benchmark.comparison(result)["latency_reduction_pct"])
        result[benchmark.EAGER] = measured(10)
        result[benchmark.CANDIDATE] = {"status": "excluded"}
        self.assertFalse(benchmark.comparison(result)["eligible"])

    def test_graph_spans_cover_all_invocations(self):
        spans = [[(10, 20), (20, 30)], [(30, 40), (40, 50)]]
        benchmark.check_disjoint(spans, [(0, 10)], lambda x: x)
        spans[1][0] = (19, 25)
        with self.assertRaises(AssertionError):
            benchmark.check_disjoint(spans, [(0, 10)], lambda x: x)


if __name__ == "__main__":
    unittest.main()
