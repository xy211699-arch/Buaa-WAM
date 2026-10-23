import copy
import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.test_dynamic_hockey_suite import ASSETS


class PolicyBoundaryTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("air_hockey_eval.policies"), "policy boundary missing")
        from air_hockey_eval.policies import XPolicyPolicy, action_rows_to_chunk
        return XPolicyPolicy, action_rows_to_chunk

    def test_extracts_configured_arm_without_treating_fingers_as_revolute_joints(self):
        _, convert = self.api()
        rows = [{"left_arm_joint_state": np.arange(6), "left_ee_joint_state": [0.04, -0.04],
                 "right_arm_joint_state": np.arange(6) + 10, "right_ee_joint_state": [0, 0]}]
        np.testing.assert_array_equal(convert(rows, "right"), np.array([[10, 11, 12, 13, 14, 15]]))
        with self.assertRaises(ValueError):
            convert([{"left_arm_joint_state": [1, 2, 3]}], "left")

    def test_existing_model_api_receives_rgb_and_each_history_frame_only_once(self):
        Wrapper, _ = self.api()
        class Model:
            def __init__(self): self.timestamps = []; self.resets = 0; self.red = []
            def reset(self): self.resets += 1
            def update_obs(self, obs):
                self.timestamps.append(obs["timestamp_s"])
                self.red.append(obs["vision"]["cam_head"]["color"][0, 0].tolist())
            def get_action(self): return [{"left_arm_joint_state": [0] * 6}]
        model = Model()
        policy = Wrapper(model, controlled_side="left")
        def frame(t):
            return {"timestamp_s": t, "vision": {"cam_head": {"color": np.array([[[255, 0, 0]]], np.uint8)}}}
        policy.reset()
        policy.predict({"frames": [frame(0), frame(0), frame(0.1)]})
        policy.predict({"frames": [frame(0.1), frame(0.2)]})
        self.assertEqual(model.timestamps, [0, 0.1, 0.2])
        self.assertEqual(model.red, [[255, 0, 0]] * 3)
        self.assertEqual(model.resets, 1)

    def test_policy_packet_filters_privileged_truth_and_only_explicitly_aliases_images(self):
        self.api()
        from air_hockey_eval.runner import policy_packet
        from air_hockey_eval.schema import BenchmarkConfig
        raw = {"timestamp_s": 0, "seed": 123, "puck_gt": [1, 2],
               "vision": {"cam_head": {"color": np.zeros((3, 4, 3), np.uint8)}},
               "state": {"left_arm_joint_state": [0] * 6, "puck_position": [1, 2]}}
        packet = policy_packet([raw], BenchmarkConfig())
        self.assertEqual(set(packet["frames"][0]["vision"]), {"cam_head"})
        self.assertEqual(set(packet["frames"][0]["state"]), {"left_arm_joint_state"})
        self.assertNotIn("seed", packet["frames"][0])
        self.assertNotIn("puck_gt", packet["frames"][0])

    def test_act_adapter_uses_new_full_native_chunk_for_every_observation(self):
        from air_hockey_eval.act_chunk import ACTChunkModel
        class Tensor:
            def __init__(self, array): self.array = array
            def __getitem__(self, key): return Tensor(self.array[key])
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.array.copy()
        class NativeACT:
            temporal_agg = False
            num_queries = 8
            def __init__(self): self.t = 7; self.calls = 0
            def get_action(self):
                if self.t == 0:
                    self.calls += 1
                    self.all_actions = Tensor(np.full((1, 8, 16), self.calls, dtype=float))
                self.t += 1
                return np.zeros((1, 16))  # legacy one-step interface
            def post_process(self, rows): return rows * 2 + 1
        class Legacy:
            def __init__(self): self.model = NativeACT()
            def reset(self): self.model.t = 0
            def update_obs(self, frame): pass
        legacy = Legacy()
        adapter = ACTChunkModel(legacy, unpack=lambda rows: [{"left_arm_joint_state": r[:6]} for r in rows])
        Wrapper, _ = self.api()
        policy = Wrapper(adapter, controlled_side="left", require_act_chunk=True)
        policy.reset()
        first = policy.predict({"frames": [{"timestamp_s": 0}]})
        second = policy.predict({"frames": [{"timestamp_s": .1}]})
        self.assertEqual(first.shape, (8, 6))
        np.testing.assert_array_equal(first, 3)
        np.testing.assert_array_equal(second, 5)
        from air_hockey_eval.control import TimedActionQueue
        queue = TimedActionQueue(action_hz=30, execution_steps=6, initial_command=np.zeros(6))
        accepted = queue.accept(first, observed_s=0, ready_s=.005, now_s=.01)
        self.assertEqual(accepted["accepted_steps"], 5)
        np.testing.assert_array_equal(queue.command_at(.04)[0], 3)

    def test_act_legacy_single_step_server_is_rejected_before_scoring(self):
        Wrapper, _ = self.api()
        class OldServer:
            def reset(self): pass
        policy = Wrapper(OldServer(), controlled_side="left", require_act_chunk=True)
        with self.assertRaisesRegex(ValueError, "ACT|chunk"):
            policy.reset()

    def test_act_loader_sets_actual_ur5_dimension_for_config_and_native_builder(self):
        from air_hockey_eval.policies import load_xpolicy_model
        captured = {}
        def construct(config):
            captured.update(config)
            captured["native_env_dimension"] = os.environ.get("ACT_ACTION_DIM")
            return SimpleNamespace(model=SimpleNamespace(temporal_agg=False))
        with patch.dict(os.environ, {"ACT_ACTION_DIM": "14"}), patch(
                "air_hockey_eval.policies.importlib.import_module", return_value=SimpleNamespace(Model=construct)):
            load_xpolicy_model("ACT", {"env_cfg_type": "air_hockey_eval_ur5", "action_type": "joint",
                                       "temporal_agg": False})
            self.assertEqual(os.environ["ACT_ACTION_DIM"], "14")
        self.assertEqual(captured["action_dim"], 16)
        self.assertEqual(captured["native_env_dimension"], "16")

    def test_wrong_robot_dimensions_are_rejected_before_model_import(self):
        from air_hockey_eval.policies import load_xpolicy_model
        with patch("air_hockey_eval.policies.importlib.import_module", side_effect=AssertionError("model import forbidden")):
            with self.assertRaisesRegex(ValueError, "dimension|six|UR5"):
                load_xpolicy_model("ACT", {"env_cfg_type": "air_hockey_eval_ur5", "action_type": "joint",
                                           "action_dim": 14, "temporal_agg": False})


class ReferenceTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("air_hockey_eval.reference"), "preflight missing")
        from air_hockey_eval.reference import prepare_manifest
        from air_hockey_eval.schema import BenchmarkConfig
        from air_hockey_eval.suite import generate_manifest, validate_manifest
        return prepare_manifest, BenchmarkConfig, generate_manifest, validate_manifest

    def test_reference_calibration_preserves_pairs_and_prepares_every_case(self):
        prepare, Config, generate, validate = self.api()
        m = generate(Config.from_dict({"speeds_m_s": [1.0], "ttc_s": [1.0], "repeats_per_cell": 1}), ASSETS)
        def reference(case, *, perturbed=False):
            # Explicit deterministic fixture: scene-dependent timing, no model.
            t = (case["position_m"][0] + 0.6) / abs(case["velocity_m_s"][0])
            return {"valid": True, "backend": "sapien-physx", "ttc_s": t,
                    "side_bounces": 1 if case["condition"] == "bank" else 0,
                    "intercept_y_m": 0.0, "intercept_speed_m_s": 1.0,
                    "perturbation_applied": bool(perturbed)}
        prepared = prepare(m, reference_provider=reference)
        validate(prepared, require_prepared=True)
        self.assertTrue(prepared["prepared"])
        self.assertFalse(m["prepared"])
        self.assertTrue(all(c["reference"]["valid"] for c in prepared["cases"]))

    def test_invalid_reference_is_rejected_not_filtered_into_an_easy_test_set(self):
        prepare, Config, generate, _ = self.api()
        m = generate(Config.from_dict({"speeds_m_s": [1.0], "ttc_s": [1.0], "repeats_per_cell": 1}), ASSETS)
        def reference(case, *, perturbed=False):
            return {"valid": False, "backend": "sapien-physx", "reason": "stopped"}
        with self.assertRaisesRegex(ValueError, "reference|preflight"):
            prepare(m, reference_provider=reference)
        self.assertEqual(len(m["cases"]), 5)


if __name__ == "__main__":
    unittest.main()
