import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests.test_dynamic_hockey_suite import ASSETS


class ExecutionTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("air_hockey_eval.execution"), "execution missing")
        from air_hockey_eval.execution import execute_suite
        from air_hockey_eval.schema import BenchmarkConfig
        from air_hockey_eval.suite import generate_manifest, seal_manifest
        cfg = BenchmarkConfig.from_dict({"speeds_m_s": [1.0], "ttc_s": [1.0],
                                         "repeats_per_cell": 1, "physics_dt_s": .01})
        m = generate_manifest(cfg, ASSETS)
        m["prepared"], m["reference_backend"] = True, "sapien-physx"
        for case in m["cases"]:
            case["reference"] = {"valid": True, "backend": "sapien-physx", "ttc_s": 1.0,
                                 "side_bounces": int(case["condition"] == "bank"),
                                 "intercept_y_m": 0., "intercept_speed_m_s": .8}
            if case["condition"] == "perturbed":
                case["perturbed_reference"] = dict(case["reference"], perturbation_applied=True)
        m = seal_manifest(m)
        policy_config = {"name": "test-policy", "backend": "python", "factory": "not_imported:factory",
                         "training_data_id": "fixture-data", "artifact_id": "fixture-artifact",
                         "training_seed": 0, "action_semantics": "joint_drive_target"}
        return execute_suite, m, policy_config

    def test_unprepared_suite_is_rejected_before_any_policy_is_loaded(self):
        execute, m, config = self.api()
        from air_hockey_eval.suite import seal_manifest
        m["prepared"] = False
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            execute(seal_manifest(m), config, d, hardware_label="fixture",
                    policy_factory=lambda *_: self.fail("a real policy was loaded"))

    def test_full_journal_and_resume_do_not_load_policy_or_run_cases_twice(self):
        execute, m, config = self.api()
        class Backend:
            y_limit_m = 0.5
            def reset(self, case): self.t = 0.
            def initial_command(self): return np.zeros(6)
            def frame(self):
                from air_hockey_eval.metrics import FrameState
                # All cases fail immediately for the same explicit safety reason.
                return FrameState(self.t, (0, 0), (-1, 0), violations=("fixture_violation",))
            def close(self): pass
        class Policy:
            def reset(self): pass
            def predict(self, packet): raise AssertionError("not needed for this fixture")
        with tempfile.TemporaryDirectory() as d:
            execute(m, config, d, hardware_label="fixture", mode="zero_latency",
                    backend_factory=lambda *_: Backend(), policy_factory=lambda *_: Policy())
            from air_hockey_eval.storage import read_run
            results = read_run(d)["results"]
            self.assertEqual(len(results), 5)
            self.assertFalse(any(r["success"] for r in results))
            execute(m, config, d, hardware_label="fixture", mode="zero_latency", resume=True,
                    backend_factory=lambda *_: self.fail("backend reloaded on a complete run"),
                    policy_factory=lambda *_: self.fail("policy reloaded on a complete run"))
            self.assertEqual(len(read_run(d)["results"]), 5)
            self.assertTrue((Path(d) / "complete.json").exists())


class CLITests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("air_hockey_eval.cli"), "CLI missing")
        from air_hockey_eval.cli import main
        return main

    def test_generate_and_check_are_cold_and_require_no_model_or_sapien(self):
        main = self.api()
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            (directory / "assets.json").write_text(json.dumps(ASSETS), encoding="utf-8")
            (directory / "config.json").write_text(json.dumps({"speeds_m_s": [1.0], "ttc_s": [1.0],
                                                               "repeats_per_cell": 1}), encoding="utf-8")
            suite = directory / "proposals.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["generate", "--config", str(directory / "config.json"),
                                       "--assets", str(directory / "assets.json"), "--output", str(suite)]), 0)
                self.assertEqual(main(["check", "--suite", str(suite)]), 0)
            self.assertEqual(len(json.loads(suite.read_text())["cases"]), 5)
            self.assertFalse(json.loads(suite.read_text())["prepared"])


if __name__ == "__main__":
    unittest.main()
