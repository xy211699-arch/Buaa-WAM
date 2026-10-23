"""A proximity hit, a backward crossing, or hidden violation must not pass."""
import importlib.util
import unittest


class MetricTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("air_hockey_eval"), "benchmark missing")
        from air_hockey_eval.metrics import EpisodeScorer, FrameState
        from air_hockey_eval.schema import BenchmarkConfig
        return EpisodeScorer, FrameState, BenchmarkConfig

    def test_crossing_without_physical_contact_is_failure(self):
        Scorer, Frame, Config = self.api()
        s = Scorer(Config(), y_limit_m=0.5)
        s.observe(Frame(0.0, (0.4, 0.0), (1.0, 0.0)))
        s.observe(Frame(0.2, (0.6, 0.0), (1.0, 0.0)))
        self.assertFalse(s.result()["success"])

    def test_contact_then_forward_crossing_interpolates_time_and_y(self):
        Scorer, Frame, Config = self.api()
        s = Scorer(Config(), y_limit_m=0.5)
        s.observe(Frame(0.0, (-0.6, 0.0), (-1.0, 0.0)))
        s.observe(Frame(0.1, (-0.5, 0.0), (1.0, 0.0), contact=True))
        s.observe(Frame(1.0, (0.5, 0.0), (1.0, 0.0)))
        s.observe(Frame(1.2, (0.6, 0.2), (1.0, 1.0)))
        r = s.result()
        self.assertTrue(r["success"])
        self.assertAlmostEqual(r["return_time_s"], 1.1)
        self.assertAlmostEqual(r["return_y_m"], 0.1)

    def test_violation_and_policy_error_override_a_return(self):
        Scorer, Frame, Config = self.api()
        s = Scorer(Config(), y_limit_m=0.5)
        s.observe(Frame(0.0, (0.5, 0), (1, 0), contact=True))
        s.observe(Frame(0.1, (0.6, 0), (1, 0), violations=("joint_velocity",)))
        self.assertTrue(s.result()["returned"])
        self.assertFalse(s.result()["success"])
        s.fail("policy_error")
        self.assertEqual(s.result()["failure_reason"], "policy_error")

    def test_wrong_direction_and_outside_open_playing_width_do_not_return(self):
        Scorer, Frame, Config = self.api()
        for first, second in ((Frame(0, (0.6, 0), (-1, 0), contact=True),
                               Frame(0.1, (0.5, 0), (-1, 0))),
                              (Frame(0, (0.5, 0.6), (1, 0), contact=True),
                               Frame(0.1, (0.6, 0.6), (1, 0)))):
            s = Scorer(Config(), y_limit_m=0.5)
            s.observe(first); s.observe(second)
            self.assertFalse(s.result()["returned"])

    def test_time_must_increase_and_nonfinite_truth_is_an_error(self):
        Scorer, Frame, Config = self.api()
        s = Scorer(Config(), y_limit_m=0.5)
        s.observe(Frame(0, (0, 0), (0, 0)))
        with self.assertRaises(ValueError):
            s.observe(Frame(0, (0, 0), (0, 0)))
        with self.assertRaises(ValueError):
            s.observe(Frame(0.1, (float("nan"), 0), (0, 0)))


if __name__ == "__main__":
    unittest.main()
