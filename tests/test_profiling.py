import unittest

from nast.profiling import StepMeasurement, aggregate_measurements, memory_reduction


class ProfilingTests(unittest.TestCase):
    def test_phase_times_and_peak_memory_are_aggregated(self):
        measurements = [
            StepMeasurement(1.0, 100, 10, 20, 0.1, 0.2, 0.6, 0.1),
            StepMeasurement(2.0, 200, 30, 40, 0.2, 0.4, 1.2, 0.2),
        ]
        summary = aggregate_measurements(measurements)
        self.assertEqual(summary["processed_tokens"], 300)
        self.assertEqual(summary["peak_allocated_bytes"], 30)
        self.assertAlmostEqual(summary["mean_scoring_seconds"], 0.15)
        self.assertAlmostEqual(summary["tokens_per_second"], 100.0)

    def test_memory_reduction(self):
        self.assertEqual(memory_reduction(100, 40), 0.6)
        self.assertIsNone(memory_reduction(0, 40))


if __name__ == "__main__":
    unittest.main()
