import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_dynamic_hockey_suite import ASSETS


def outcome(case_id, success):
    return {"case_id": case_id, "success": bool(success), "hit": bool(success),
            "returned": bool(success), "violations": [],
            "failure_reason": None if success else "no_contact", "duration_s": 1.0,
            "starvation_fraction": 0.1,
            "timings": [{"compute_s": 0.05, "observation_to_ready_s": 0.06,
                         "observation_to_apply_s": 0.07}]}


class StorageTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("air_hockey_eval.storage"), "result journal missing")
        from air_hockey_eval.storage import RunStore, read_run
        return RunStore, read_run

    def test_resume_is_exactly_once_and_rejects_different_protocol(self):
        Store, read = self.api()
        identity = {"suite_sha256": "abc", "policy_name": "FastWAM", "training_seed": 0}
        with tempfile.TemporaryDirectory() as d:
            with Store(d, identity, ["a", "b"]) as store:
                store.append(outcome("a", False))
                with self.assertRaises(ValueError): store.append(outcome("a", True))
                with self.assertRaises(ValueError): store.finalize()
            with Store(d, identity, ["a", "b"], resume=True) as store:
                self.assertEqual(set(store.completed), {"a"})
                store.append(outcome("b", True)); store.finalize()
            self.assertEqual(len(read(d)["results"]), 2)
            with self.assertRaises(ValueError):
                Store(d, dict(identity, suite_sha256="different"), ["a", "b"], resume=True)

    def test_only_a_truncated_final_record_is_recovered(self):
        Store, _ = self.api()
        with tempfile.TemporaryDirectory() as d:
            with Store(d, {"suite_sha256": "abc"}, ["a", "b"]) as store:
                store.append(outcome("a", True))
            path = Path(d) / "episodes.jsonl"
            with path.open("ab") as f: f.write(b'{"case_id": "b"')
            with Store(d, {"suite_sha256": "abc"}, ["a", "b"], resume=True) as store:
                self.assertEqual(len(store.completed), 1)
                store.append(outcome("b", False))
            self.assertEqual(len(path.read_text().splitlines()), 2)
            path.write_text('{bad}\n' + path.read_text(), encoding="utf-8")
            with self.assertRaises(ValueError):
                Store(d, {"suite_sha256": "abc"}, ["a", "b"], resume=True)

    def test_concurrent_writer_is_rejected_and_lock_is_released(self):
        Store, _ = self.api()
        with tempfile.TemporaryDirectory() as d:
            with Store(d, {"suite_sha256": "abc"}, ["a"]):
                with self.assertRaises((OSError, ValueError)):
                    Store(d, {"suite_sha256": "abc"}, ["a"], resume=True)
            with Store(d, {"suite_sha256": "abc"}, ["a"], resume=True) as store:
                store.append(outcome("a", True)); store.finalize()


class ReportTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("air_hockey_eval.report"), "analysis missing")
        from air_hockey_eval.report import aggregate_runs, write_report
        return aggregate_runs, write_report

    def fixture(self):
        from air_hockey_eval.schema import BenchmarkConfig, fingerprint
        from air_hockey_eval.suite import generate_manifest
        m = generate_manifest(BenchmarkConfig.from_dict({"speeds_m_s": [1.0], "ttc_s": [1.0],
                                                         "repeats_per_cell": 2}), ASSETS)
        ids = [c["case_id"] for c in m["cases"]]
        runs = []
        for method in ("FastWAM", "DP"):
            for seed in (0, 1, 2):
                identity = {"suite_sha256": m["suite_sha256"], "policy_name": method,
                            "training_seed": seed, "protocol_sha256": fingerprint(m["protocol"]),
                            "artifact_id": f"{method}-independent-training-{seed}",
                            "code_sha256": "fixture-evaluator-version",
                            "mode": "latency", "hardware_label": "test-fixture",
                            "training_data_id": "same-data", "action_semantics": "joint_drive_target"}
                runs.append({"identity": identity, "case_ids": ids,
                             "results": [outcome(c, method == "FastWAM") for c in reversed(ids)]})
        return m, runs

    def test_same_checkpoint_cannot_be_relabeled_as_independent_training(self):
        aggregate, _ = self.api()
        m, runs = self.fixture()
        runs[1]["identity"]["artifact_id"] = runs[0]["identity"]["artifact_id"]
        with self.assertRaisesRegex(ValueError, "checkpoint|artifact"):
            aggregate(runs, m, bootstrap_samples=100)

    def test_different_evaluator_implementations_cannot_be_pooled(self):
        aggregate, _ = self.api()
        m, runs = self.fixture()
        runs[0]["identity"]["code_sha256"] = "different-code"
        with self.assertRaisesRegex(ValueError, "implementation|code"):
            aggregate(runs, m, bootstrap_samples=100)

    def test_repeated_conditions_are_resampled_as_one_scene_cluster(self):
        import numpy as np
        from air_hockey_eval.report import _interval
        values = np.array([[0.] * 5 + [1.] * 5])
        lo, hi = _interval(values, np.random.default_rng(1), 1000, scene_units=["a"] * 5 + ["b"] * 5)
        self.assertEqual((lo, hi), (0., 1.))

    def test_perfect_small_sample_does_not_certify_a_reaction_threshold(self):
        aggregate, _ = self.api()
        m, runs = self.fixture()
        result = aggregate(runs, m, bootstrap_samples=100)
        threshold = next(r for r in result["reaction_thresholds"] if r["policy_name"] == "FastWAM")
        self.assertIsNone(threshold["min_tested_ttc_s"])
        self.assertEqual(threshold["observed_min_ttc_s"], 1.)
        self.assertEqual(threshold["status"], "insufficient_uncertainty_evidence")
        self.assertTrue(next(r for r in result["summary"] if r["policy_name"] == "FastWAM")["ci_degenerate"])

    def test_pairing_uses_case_ids_not_file_order_and_seeds_are_separate(self):
        aggregate, _ = self.api()
        m, runs = self.fixture()
        result = aggregate(runs, m, bootstrap_samples=200)
        by_name = {r["policy_name"]: r for r in result["summary"]}
        self.assertEqual(by_name["FastWAM"]["training_runs"], 3)
        self.assertEqual(by_name["FastWAM"]["success_rate"], 1.0)
        self.assertEqual(by_name["DP"]["success_rate"], 0.0)
        comp = next(x for x in result["method_comparisons"]
                    if x["method_a"] == "FastWAM" and x["method_b"] == "DP" and x["group"] == "all")
        self.assertEqual(comp["difference"], 1.0)
        self.assertEqual(comp["ci_low"], 1.0)
        self.assertEqual(comp["ci_high"], 1.0)

    def test_missing_cases_duplicates_and_unmatched_hardware_are_not_silently_pooled(self):
        aggregate, _ = self.api()
        m, runs = self.fixture()
        for mode in ("missing", "duplicate", "hardware", "protocol"):
            bad = copy.deepcopy(runs)
            if mode == "missing": bad[0]["results"].pop()
            if mode == "duplicate": bad[0]["results"].append(bad[0]["results"][0])
            if mode == "hardware": bad[0]["identity"]["hardware_label"] = "other GPU"
            if mode == "protocol": bad[0]["identity"]["mode"] = "zero_latency"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                aggregate(bad, m, bootstrap_samples=200)

    def test_failure_remains_in_denominator_and_condition_drop_is_paired(self):
        aggregate, _ = self.api()
        m, runs = self.fixture()
        for run in runs:
            if run["identity"]["policy_name"] == "FastWAM":
                for row in run["results"]:
                    if row["case_id"].endswith("-perturbed"):
                        row.update(success=False, returned=False, hit=False, failure_reason="policy_error")
        result = aggregate(runs, m, bootstrap_samples=200)
        summary = next(x for x in result["summary"] if x["policy_name"] == "FastWAM")
        self.assertAlmostEqual(summary["success_rate"], 0.8)
        paired = next(x for x in result["paired_conditions"] if x["policy_name"] == "FastWAM"
                      and x["family"] == "perturbation")
        self.assertEqual(paired["difference"], -1.0)
        effect = next(x for x in result["method_condition_effects"] if x["method_a"] == "FastWAM"
                      and x["method_b"] == "DP" and x["family"] == "perturbation")
        self.assertEqual(effect["difference_of_changes"], -1.0)
        ttc = next(x for x in result["method_comparisons"] if x["method_a"] == "FastWAM"
                   and x["method_b"] == "DP" and x["group"] == "reaction:ttc=1")
        self.assertEqual(ttc["difference"], 1.0)

    def test_report_writes_three_plots_and_machine_readable_tables(self):
        aggregate, write = self.api()
        m, runs = self.fixture()
        with tempfile.TemporaryDirectory() as d:
            write(aggregate(runs, m, bootstrap_samples=200), d)
            for name in ("summary.json", "summary.csv", "groups.csv", "paired_conditions.csv",
                         "method_comparisons.csv", "reaction.png", "bounce.png", "perturbation.png"):
                self.assertGreater((Path(d) / name).stat().st_size, 20, name)
            saved = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["complete"])


if __name__ == "__main__":
    unittest.main()
