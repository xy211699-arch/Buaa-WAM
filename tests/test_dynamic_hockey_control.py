import importlib.util
import threading
import time
import unittest

import numpy as np


class ControlTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("air_hockey_eval.control"), "scheduler missing")
        from air_hockey_eval.control import TimedActionQueue, validate_action_chunk
        return TimedActionQueue, validate_action_chunk

    def test_late_prediction_drops_elapsed_prefix_not_retimes_it(self):
        Queue, _ = self.api()
        q = Queue(action_hz=10, execution_steps=4, initial_command=np.zeros(6))
        report = q.accept(np.repeat(np.arange(4)[:, None], 6, axis=1), observed_s=0,
                          ready_s=0.15, now_s=0.15)
        self.assertEqual(report["expired_steps"], 2)
        self.assertEqual(report["accepted_steps"], 2)
        np.testing.assert_array_equal(q.command_at(0.19)[0], 0)
        np.testing.assert_array_equal(q.command_at(0.20)[0], 2)
        np.testing.assert_array_equal(q.command_at(0.31)[0], 3)
        self.assertFalse(q.has_coverage(0.41))

    def test_wholly_stale_chunk_does_not_erase_a_valid_buffer(self):
        Queue, _ = self.api()
        q = Queue(action_hz=10, execution_steps=4, initial_command=np.zeros(6))
        q.accept(np.ones((4, 6)), observed_s=0.2, ready_s=0.2, now_s=0.2)
        q.accept(np.zeros((4, 6)), observed_s=0, ready_s=0.5, now_s=0.5)
        np.testing.assert_array_equal(q.command_at(0.5)[0], 1)

    def test_new_chunk_replaces_old_future_without_growing_queue(self):
        Queue, _ = self.api()
        q = Queue(action_hz=10, execution_steps=3, initial_command=np.zeros(6))
        for i in range(10):
            q.accept(np.full((20, 6), i), observed_s=i / 10, ready_s=i / 10, now_s=i / 10)
            self.assertLessEqual(q.pending_steps, 3)
        np.testing.assert_array_equal(q.command_at(1.0)[0], 9)

    def test_invalid_command_shapes_and_nan_cannot_reach_robot(self):
        _, validate = self.api()
        for actions in ([], np.zeros((2, 5)), np.zeros((1, 2, 6)), np.full((2, 6), np.nan)):
            with self.subTest(shape=np.shape(actions)), self.assertRaises(ValueError):
                validate(actions)


class RunnerTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("air_hockey_eval.runner"), "runner missing")
        from air_hockey_eval.runner import run_episode
        from air_hockey_eval.schema import BenchmarkConfig
        from air_hockey_eval.metrics import FrameState
        return run_episode, BenchmarkConfig, FrameState

    def backend(self, Frame, started, finished):
        class Backend:
            y_limit_m = 0.5
            def reset(self, case):
                self.t = 0.; self.x = -0.6; self.v = 0.; self.command = np.zeros(6)
                self.stepped_during_inference = 0; self.applied = []; self.perturbations = []
            def initial_command(self): return np.zeros(6)
            def frame(self): return Frame(self.t, (self.x, 0), (self.v, 0), contact=self.v > 0)
            def observe(self):
                return {"vision": {"cam_head": {"color": np.zeros((2, 2, 3), np.uint8)}},
                        "state": {"left_arm_joint_state": np.zeros(6)}, "timestamp_s": self.t}
            def apply(self, command):
                self.command = command.copy(); self.applied.append(self.t)
                if command[0] > 0.5: self.v = 20.
                return ()
            def perturb(self, delta): self.perturbations.append((self.t, tuple(delta)))
            def step(self):
                if started.is_set() and not finished.is_set(): self.stepped_during_inference += 1
                self.t = round(self.t + 0.01, 8); self.x += self.v * 0.01
        return Backend()

    def test_simulator_moves_while_prediction_is_running_and_applies_no_future_action(self):
        run, Config, Frame = self.api()
        started, finished = threading.Event(), threading.Event()
        backend = self.backend(Frame, started, finished)
        class Policy:
            def reset(self): pass
            def predict(self, packet):
                # No ground truth, case seed, reference trajectory or future labels.
                self.packet = packet
                started.set(); time.sleep(0.035); finished.set()
                return np.ones((30, 6))
        policy = Policy()
        cfg = Config.from_dict({"physics_dt_s": 0.01, "observation_hz": 100, "action_hz": 100,
                                "ttc_s": [0.1], "episode_timeout_s": 0.4,
                                "execution_steps": 20, "replan_interval_s": 0.05})
        case = {"case_id": "timing", "perturb_time_s": None, "delta_velocity_m_s": [0, 0]}
        result = run(backend, policy, case, cfg, mode="latency", trace=lambda _: None)
        self.assertGreater(backend.stepped_during_inference, 0)
        self.assertGreaterEqual(min(backend.applied), 0.03)
        self.assertTrue(result["success"])
        self.assertNotIn("case_id", policy.packet)
        self.assertNotIn("puck_gt", str(policy.packet.keys()))
        self.assertEqual(len(policy.packet["frames"]), 3)

    def test_policy_error_counts_as_failure_and_perturbation_occurs_exactly_once(self):
        run, Config, Frame = self.api()
        backend = self.backend(Frame, threading.Event(), threading.Event())
        class Policy:
            def reset(self): pass
            def predict(self, packet): raise ValueError("bad action")
        cfg = Config.from_dict({"physics_dt_s": 0.01, "ttc_s": [0.1], "episode_timeout_s": 0.2})
        case = {"case_id": "bad", "perturb_time_s": None, "delta_velocity_m_s": [0, 0]}
        result = run(backend, Policy(), case, cfg, mode="zero_latency")
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_reason"], "policy_error")
        class HoldPolicy:
            def reset(self): pass
            def predict(self, packet): return np.zeros((10, 6))
        case.update(perturb_time_s=0.035, delta_velocity_m_s=[0.1, 0.2])
        result = run(backend, HoldPolicy(), case, cfg, mode="zero_latency")
        self.assertEqual(len(backend.perturbations), 1)
        self.assertAlmostEqual(backend.perturbations[0][0], 0.04)
        self.assertTrue(result["perturbation_applied"])

    def test_return_before_planned_perturbation_is_not_a_perturbation_success(self):
        run, Config, Frame = self.api()
        backend = self.backend(Frame, threading.Event(), threading.Event())
        class EarlyReturn:
            def reset(self): pass
            def predict(self, packet): return np.ones((10, 6))
        cfg = Config.from_dict({"physics_dt_s": .01, "observation_hz": 100,
                                "action_hz": 100, "ttc_s": [.1], "episode_timeout_s": .2})
        case = {"case_id": "early", "perturb_time_s": .15, "delta_velocity_m_s": [.1, .2]}
        result = run(backend, EarlyReturn(), case, cfg, mode="zero_latency")
        self.assertFalse(result["success"])
        self.assertFalse(result["perturbation_applied"])
        self.assertEqual(result["failure_reason"], "contact_before_perturbation")


if __name__ == "__main__":
    unittest.main()
