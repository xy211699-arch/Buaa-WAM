"""CPU-only benchmark contracts; no renderer or learned policy is imported."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ASSETS = {
    "physics": {"timestep_s": 0.001},
    "table": {"length_m": 2.128, "width_m": 1.218, "rail_width_m": 0.045,
              "top_height_m": 0.74, "goal_width_m": 0.4},
    "puck": {"diameter_m": 0.0633, "thickness_m": 0.02},
}


class SuiteTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("air_hockey_eval"),
                             "dynamic benchmark package is not implemented")
        from air_hockey_eval.schema import BenchmarkConfig
        from air_hockey_eval.suite import generate_manifest, validate_manifest, write_manifest, read_manifest
        return BenchmarkConfig, generate_manifest, validate_manifest, write_manifest, read_manifest

    def test_all_families_are_balanced_and_paired_without_policy_filtering(self):
        Config, generate, validate, _, _ = self.api()
        cfg = Config.from_dict({"speeds_m_s": [1.0], "ttc_s": [1.0], "repeats_per_cell": 2})
        m = generate(cfg, ASSETS)
        validate(m)
        self.assertEqual(len(m["cases"]), 10)  # 2 x (reaction + direct/bank + clean/perturbed)
        self.assertEqual(m, generate(cfg, ASSETS))
        ids = [c["case_id"] for c in m["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        clean = next(c for c in m["cases"] if c["condition"] == "clean")
        disturbed = next(c for c in m["cases"] if c["pair_id"] == clean["pair_id"] and c["condition"] == "perturbed")
        for field in ("position_m", "velocity_m_s", "seed", "nominal_ttc_s", "speed_m_s"):
            self.assertEqual(clean[field], disturbed[field])
        self.assertIsNone(clean["perturb_time_s"])
        self.assertGreater(disturbed["perturb_time_s"], 0)
        self.assertFalse(m["prepared"])

    def test_impossible_geometry_fails_instead_of_silently_removing_hard_cases(self):
        Config, generate, _, _, _ = self.api()
        cfg = Config.from_dict({"speeds_m_s": [0.1], "ttc_s": [0.1]})
        with self.assertRaisesRegex(ValueError, "bank|bounce|feasible"):
            generate(cfg, ASSETS)

    def test_configuration_rejects_nan_unknown_fields_and_bad_timing(self):
        Config, *_ = self.api()
        for values in ({"speeds_m_s": [float("nan")]}, {"history_frame": 2},
                       {"physics_dt_s": 0}, {"execution_steps": 0},
                       {"extra_latency_s": -0.01}, {"controlled_side": "both"},
                       {"observation_hz": 1001}, {"max_command_speed_rad_s": [1]}):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                Config.from_dict(values)

    def test_manifest_roundtrip_and_tamper_detection(self):
        Config, generate, validate, write, read = self.api()
        m = generate(Config.from_dict({"speeds_m_s": [1.0], "ttc_s": [1.0], "repeats_per_cell": 1}), ASSETS)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "suite.json"
            write(p, m)
            self.assertEqual(read(p), m)
            with self.assertRaises(FileExistsError):
                write(p, m)
        bad = copy.deepcopy(m)
        bad["cases"][0]["velocity_m_s"][0] = -3.0
        with self.assertRaisesRegex(ValueError, "digest|fingerprint|hash"):
            validate(bad)

    def test_right_arm_cases_are_mirrors_not_a_different_distribution(self):
        Config, generate, *_ = self.api()
        options = {"speeds_m_s": [1.0], "ttc_s": [1.0], "repeats_per_cell": 1}
        left = generate(Config.from_dict(options), ASSETS)
        right = generate(Config.from_dict(dict(options, controlled_side="right")), ASSETS)
        for a, b in zip(left["cases"], right["cases"]):
            self.assertAlmostEqual(a["position_m"][0], -b["position_m"][0])
            self.assertAlmostEqual(a["velocity_m_s"][0], -b["velocity_m_s"][0])
            self.assertEqual(a["position_m"][1], b["position_m"][1])

    def test_resealed_suite_cannot_omit_an_entire_difficult_family(self):
        Config, generate, validate, *_ = self.api()
        from air_hockey_eval.suite import seal_manifest
        m = generate(Config.from_dict({"speeds_m_s": [1.0], "ttc_s": [1.0], "repeats_per_cell": 1}), ASSETS)
        m["cases"] = [c for c in m["cases"] if c["family"] != "bounce"]
        with self.assertRaisesRegex(ValueError, "grid|case|complete"):
            validate(seal_manifest(m))

    def test_prepared_reference_must_match_ttc_bounce_and_actual_perturbation(self):
        Config, generate, validate, *_ = self.api()
        from air_hockey_eval.suite import seal_manifest
        m = generate(Config.from_dict({"speeds_m_s": [1.0], "ttc_s": [1.0], "repeats_per_cell": 1}), ASSETS)
        m.update(prepared=True, reference_backend="sapien-physx")
        for c in m["cases"]:
            c["reference"] = {"valid": True, "backend": "sapien-physx", "ttc_s": 1.,
                              "side_bounces": int(c["condition"] == "bank"),
                              "intercept_y_m": 0., "intercept_speed_m_s": .8}
            if c["condition"] == "perturbed":
                c["perturbed_reference"] = dict(c["reference"], perturbation_applied=True)
        validate(seal_manifest(m), require_prepared=True)
        for field in ("ttc", "bounce", "perturbation"):
            bad = copy.deepcopy(m)
            if field == "ttc": bad["cases"][0]["reference"]["ttc_s"] = 2.
            if field == "bounce": bad["cases"][0]["reference"]["side_bounces"] = 1
            if field == "perturbation":
                bad["cases"][-1]["perturbed_reference"]["perturbation_applied"] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(seal_manifest(bad), require_prepared=True)


if __name__ == "__main__":
    unittest.main()
