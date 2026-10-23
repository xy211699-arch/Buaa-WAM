"""Seed-aware paired analysis. No incomplete or incompatible runs are pooled."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .schema import BenchmarkConfig, fingerprint
from .storage import _validate_row
from .suite import validate_manifest


def _scene_matrix(matrix, scene_units):
    if scene_units is None:
        return matrix
    if len(scene_units) != matrix.shape[1]:
        raise ValueError("scene cluster count differs from matrix columns")
    # Each generated cell contributes the same five related conditions. Keep
    # those conditions together, rather than pretending they are five scenes.
    labels = list(dict.fromkeys(scene_units))
    return np.column_stack([matrix[:, [i for i, key in enumerate(scene_units) if key == label]].mean(axis=1)
                            for label in labels])


def _interval(matrix: np.ndarray, rng, samples: int, *, scene_units=None) -> tuple[float, float]:
    matrix = _scene_matrix(matrix, scene_units)
    values = []
    for _ in range(samples):
        seeds = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        cases = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        values.append(float(matrix[seeds][:, cases].mean()))
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def _difference_interval(a, b, rng, samples: int, *, scene_units=None):
    a, b = _scene_matrix(a, scene_units), _scene_matrix(b, scene_units)
    values = []
    for _ in range(samples):
        # Train runs are independently resampled; scenes remain paired.
        ia = rng.integers(0, a.shape[0], size=a.shape[0])
        ib = rng.integers(0, b.shape[0], size=b.shape[0])
        ic = rng.integers(0, a.shape[1], size=a.shape[1])
        values.append(float(a[ia][:, ic].mean() - b[ib][:, ic].mean()))
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def _quantile(values, p):
    return float(np.quantile(values, p)) if values else None


def aggregate_runs(runs: list[dict], manifest: dict, *, bootstrap_samples: int = 2000,
                   random_seed: int = 20261023) -> dict:
    validate_manifest(manifest)
    if not runs or bootstrap_samples < 100:
        raise ValueError("at least one run and 100 bootstrap samples are required")
    cfg = BenchmarkConfig.from_dict(manifest["protocol"])
    cases = manifest["cases"]
    ids = [c["case_id"] for c in cases]
    scene_units = [c["pair_id"].partition("-")[2] for c in cases]
    expected = set(ids)
    methods, identities, artifacts = {}, set(), set()
    fairness = None
    for run in runs:
        identity = run["identity"]
        if identity.get("suite_sha256") != manifest["suite_sha256"]:
            raise ValueError("results reference a different frozen suite")
        if identity.get("protocol_sha256") != fingerprint(manifest["protocol"]):
            raise ValueError("results reference a different observation/control protocol")
        contract = tuple(identity.get(k) for k in ("mode", "hardware_label", "training_data_id", "action_semantics", "code_sha256"))
        if any(value is None or value == "" for value in contract):
            raise ValueError("missing fairness metadata")
        if fairness is not None and contract != fairness:
            raise ValueError("incompatible hardware, latency mode, training data, action semantics or evaluator code implementation")
        fairness = contract
        name, seed = identity.get("policy_name"), identity.get("training_seed")
        if not isinstance(name, str) or not name or type(seed) is not int:
            raise ValueError("policy name and integer training seed required")
        if (name, seed) in identities:
            raise ValueError("duplicate training run")
        identities.add((name, seed))
        artifact = identity.get("artifact_id")
        if not isinstance(artifact, str) or not artifact.strip():
            raise ValueError("missing actual checkpoint artifact identity")
        if (name, artifact) in artifacts:
            raise ValueError("same checkpoint artifact cannot count as independent training runs")
        artifacts.add((name, artifact))
        if set(run["case_ids"]) != expected or len(run["case_ids"]) != len(ids):
            raise ValueError("run manifest case list mismatch")
        by_id = {}
        for row in run["results"]:
            _validate_row(row, expected)
            if row["case_id"] in by_id:
                raise ValueError("duplicate result")
            by_id[row["case_id"]] = row
        if set(by_id) != expected:
            raise ValueError("incomplete run; missing cases cannot be silently excluded")
        methods.setdefault(name, []).append((seed, [by_id[i] for i in ids]))
    for entries in methods.values():
        entries.sort(key=lambda x: x[0])
    matrices = {name: np.array([[float(r["success"]) for r in rows] for _, rows in entries])
                for name, entries in methods.items()}
    rng = np.random.default_rng(random_seed)
    summary, groups, paired, comparisons, thresholds, effects = [], [], [], [], [], []
    condition_matrices = {}
    for name, entries in sorted(methods.items()):
        matrix = matrices[name]
        low, high = _interval(matrix, rng, bootstrap_samples, scene_units=scene_units)
        rows = [row for _, group in entries for row in group]
        ready = [t["observation_to_ready_s"] for r in rows for t in r.get("timings", [])
                 if t.get("observation_to_ready_s") is not None]
        applied = [t["observation_to_apply_s"] for r in rows for t in r.get("timings", [])
                   if t.get("observation_to_apply_s") is not None]
        summary.append({"policy_name": name, "training_runs": len(entries),
                        "training_seeds": [seed for seed, _ in entries], "episodes": len(rows),
                        "success_rate": float(matrix.mean()), "ci_low": low, "ci_high": high,
                        "ci_degenerate": high - low < 1e-12,
                        "scene_clusters": len(set(scene_units)),
                        "per_training_run_success": [float(row.mean()) for row in matrix],
                        "violation_rate": float(np.mean([bool(r.get("violations")) for r in rows])),
                        "hit_rate": float(np.mean([bool(r.get("hit")) for r in rows])),
                        "ready_latency_p50_s": _quantile(ready, .5),
                        "ready_latency_p95_s": _quantile(ready, .95),
                        "applied_latency_p95_s": _quantile(applied, .95),
                        "starvation_fraction": float(np.mean([r.get("starvation_fraction", 0) for r in rows]))})
        selectors = set()
        for case in cases:
            selectors.add((case["family"], case["condition"], case["speed_m_s"], case["nominal_ttc_s"]))
            selectors.add((case["family"], case["condition"], None, None))
            if case["family"] == "reaction":
                selectors.add((case["family"], case["condition"], None, case["nominal_ttc_s"]))
        for family, condition, speed, ttc in sorted(selectors, key=str):
            indices = [i for i, c in enumerate(cases) if c["family"] == family and c["condition"] == condition
                       and (speed is None or c["speed_m_s"] == speed)
                       and (ttc is None or c["nominal_ttc_s"] == ttc)]
            sub = matrix[:, indices]
            lo, hi = _interval(sub, rng, bootstrap_samples)
            groups.append({"policy_name": name, "family": family, "condition": condition,
                           "speed_m_s": speed, "nominal_ttc_s": ttc,
                           "cases_per_training_run": len(indices), "success_rate": float(sub.mean()),
                           "ci_low": lo, "ci_high": hi, "ci_degenerate": hi - lo < 1e-12})
        reaction = sorted((r for r in groups if r["policy_name"] == name and r["family"] == "reaction"
                           and r["speed_m_s"] is None and r["nominal_ttc_s"] is not None),
                          key=lambda r: r["nominal_ttc_s"], reverse=True)
        minimum, observed_minimum = None, None
        for row in reaction:
            if row["success_rate"] < cfg.success_threshold:
                break
            observed_minimum = row["nominal_ttc_s"]
        training_sufficient = len(entries) >= 3
        uncertainty_limited = not training_sufficient
        for row in reaction:
            if not training_sufficient:
                break
            if row["ci_degenerate"] and row["success_rate"] >= cfg.success_threshold:
                uncertainty_limited = True
                break
            if row["ci_low"] < cfg.success_threshold:
                break
            minimum = row["nominal_ttc_s"]
        thresholds.append({"policy_name": name, "success_threshold": cfg.success_threshold,
                           "min_tested_ttc_s": minimum, "observed_min_ttc_s": observed_minimum,
                           "status": ("estimated" if minimum is not None else
                                      "insufficient_uncertainty_evidence" if uncertainty_limited else "target_not_reached"),
                           "uncertainty_limited": uncertainty_limited,
                           "criterion": "nondegenerate percentile lower bound; >=3 training runs; every longer tested TTC passes",
                           "extrapolated": False})
        for family, control, treatment in (("bounce", "direct", "bank"),
                                           ("perturbation", "clean", "perturbed")):
            base = {c["pair_id"]: i for i, c in enumerate(cases)
                    if c["family"] == family and c["condition"] == control}
            alt = {c["pair_id"]: i for i, c in enumerate(cases)
                   if c["family"] == family and c["condition"] == treatment}
            if set(base) != set(alt):
                raise ValueError("condition pair mismatch")
            keys = sorted(base)
            difference = matrix[:, [alt[k] for k in keys]] - matrix[:, [base[k] for k in keys]]
            condition_matrices[name, family] = difference
            lo, hi = _interval(difference, rng, bootstrap_samples)
            paired.append({"policy_name": name, "family": family, "control": control,
                           "treatment": treatment, "pairs_per_training_run": len(keys),
                           "control_success_rate": float(matrix[:, list(base.values())].mean()),
                           "treatment_success_rate": float(matrix[:, list(alt.values())].mean()),
                           "difference": float(difference.mean()), "ci_low": lo, "ci_high": hi,
                           "ci_degenerate": hi - lo < 1e-12})
    comparison_groups = {"all": list(range(len(cases))),
                         "reaction": [i for i, c in enumerate(cases) if c["family"] == "reaction"]}
    for ttc in cfg.ttc_s:
        comparison_groups[f"reaction:ttc={ttc:g}"] = [i for i, c in enumerate(cases)
                                                      if c["family"] == "reaction" and c["nominal_ttc_s"] == ttc]
    for family, condition in (("bounce", "direct"), ("bounce", "bank"),
                               ("perturbation", "clean"), ("perturbation", "perturbed")):
        comparison_groups[f"{family}:{condition}"] = [i for i, c in enumerate(cases)
                                                     if c["family"] == family and c["condition"] == condition]
    for a, b in itertools.permutations(sorted(methods), 2):
        for group, indices in comparison_groups.items():
            ma, mb = matrices[a][:, indices], matrices[b][:, indices]
            lo, hi = _difference_interval(ma, mb, rng, bootstrap_samples,
                                          scene_units=[scene_units[i] for i in indices])
            comparisons.append({"method_a": a, "method_b": b, "group": group,
                                "difference": float(ma.mean() - mb.mean()), "ci_low": lo, "ci_high": hi,
                                "ci_degenerate": hi - lo < 1e-12})
        for family in ("bounce", "perturbation"):
            ma, mb = condition_matrices[a, family], condition_matrices[b, family]
            lo, hi = _difference_interval(ma, mb, rng, bootstrap_samples)
            effects.append({"method_a": a, "method_b": b, "family": family,
                            "change_a": float(ma.mean()), "change_b": float(mb.mean()),
                            "difference_of_changes": float(ma.mean() - mb.mean()),
                            "ci_low": lo, "ci_high": hi,
                            "ci_degenerate": hi - lo < 1e-12,
                            "interpretation": "positive means A loses less (or gains more) under the dynamic condition"})
    return {"schema_version": 1, "suite_sha256": manifest["suite_sha256"], "complete": True,
            "units": "rates and differences are fractions; latency/TTC are seconds",
            "bootstrap": {"samples": bootstrap_samples, "seed": random_seed,
                          "method": "training-run resampling, paired scene-cluster resampling",
                          "interval": "approximate 95% empirical percentile; zero width does not establish certainty"},
            "fairness": dict(zip(("mode", "hardware_label", "training_data_id", "action_semantics", "code_sha256"), fairness)),
            "warnings": ([f"{name}: fewer than three independent training runs" for name, rows in methods.items() if len(rows) < 3]
                         + [f"{name}: degenerate bootstrap intervals exist; treat those intervals as descriptive only"
                            for name in sorted(methods) if any(r["policy_name"] == name and r["ci_degenerate"] for r in groups)]),
            "summary": summary, "groups": groups, "paired_conditions": paired,
            "method_comparisons": comparisons, "method_condition_effects": effects, "reaction_thresholds": thresholds}


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def write_report(report: dict, directory, *, plots: bool = True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for key in ("summary", "groups", "paired_conditions", "method_comparisons", "method_condition_effects", "reaction_thresholds"):
        _write_csv(directory / f"{key}.csv", report[key])
    if not plots:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["policy_name"] for r in report["summary"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in names:
        rows = sorted((r for r in report["groups"] if r["policy_name"] == name and r["family"] == "reaction"
                       and r["speed_m_s"] is None and r["nominal_ttc_s"] is not None), key=lambda r: r["nominal_ttc_s"])
        xs = [r["nominal_ttc_s"] for r in rows]
        ax.plot(xs, [100 * r["success_rate"] for r in rows], "o-", label=name)
        ax.fill_between(xs, [100 * r["ci_low"] for r in rows], [100 * r["ci_high"] for r in rows], alpha=.15)
    ax.set(xlabel="Nominal available reaction time (s)", ylabel="Legal return success (%)", ylim=(0, 105))
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    for suffix in ("png", "svg"): fig.savefig(directory / f"reaction.{suffix}", dpi=180)
    plt.close(fig)
    for family, conditions in (("bounce", ("direct", "bank")), ("perturbation", ("clean", "perturbed"))):
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(names))
        for j, condition in enumerate(conditions):
            rows = [next(r for r in report["groups"] if r["policy_name"] == name and r["family"] == family
                         and r["condition"] == condition and r["speed_m_s"] is None and r["nominal_ttc_s"] is None)
                    for name in names]
            y = np.array([100 * r["success_rate"] for r in rows])
            errors = np.maximum(0, np.array([y - [100 * r["ci_low"] for r in rows],
                                             [100 * r["ci_high"] for r in rows] - y]))
            ax.bar(x + (j - .5) * .35, y, .35, label=condition, yerr=errors, capsize=3)
        ax.set(xticks=x, xticklabels=names, ylabel="Legal return success (%)", ylim=(0, 105))
        ax.legend(); ax.grid(axis="y", alpha=.2); fig.tight_layout()
        for suffix in ("png", "svg"): fig.savefig(directory / f"{family}.{suffix}", dpi=180)
        plt.close(fig)
