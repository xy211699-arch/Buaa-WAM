"""Policy-independent paired proposals and immutable, fingerprinted manifests."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from .schema import BenchmarkConfig, Geometry, canonical_json, fingerprint, finite_number


def seal_manifest(manifest: dict) -> dict:
    result = copy.deepcopy(manifest)
    result.pop("suite_sha256", None)
    result["suite_sha256"] = fingerprint(result)
    return result


def generate_manifest(config: BenchmarkConfig, assets: dict) -> dict:
    geometry = Geometry.from_assets(assets)
    if abs(config.intercept_line_x_m) >= geometry.x_limit_m or abs(config.return_line_x_m) >= geometry.x_limit_m:
        raise ValueError("scoring lines must be inside the playing field")
    rng = np.random.default_rng(config.seed)
    cases = []
    sign = config.direction
    for si, speed in enumerate(config.speeds_m_s):
        for ti, ttc in enumerate(config.ttc_s):
            for ri in range(config.repeats_per_cell):
                seed = int(rng.integers(0, 2**31 - 1))
                target_y = float(rng.uniform(-0.015, 0.015))
                side = 1 if ri % 2 == 0 else -1
                cell = f"v{si:02d}-t{ti:02d}-r{ri:04d}"
                bank_y = side * (geometry.y_limit_m - 0.10)
                unfolded_target_y = side * 2 * geometry.y_limit_m - target_y
                dy = unfolded_target_y - bank_y
                distance = speed * ttc
                if abs(dy) >= distance:
                    raise ValueError(f"bank bounce geometry is not feasible at {cell}; increase TTC/speed")
                bank_vx = -math.sqrt(distance**2 - dy**2) / ttc
                starts = {
                    "direct": ([config.intercept_line_x_m + distance, target_y], [-speed, 0.0]),
                    "bank": ([config.intercept_line_x_m - bank_vx * ttc, bank_y], [bank_vx, dy / ttc]),
                }
                for position, _ in starts.values():
                    if not (-geometry.x_limit_m < position[0] < geometry.x_limit_m and abs(position[1]) < geometry.y_limit_m):
                        raise ValueError(f"serve geometry outside playing field at {cell}; adjust grid")
                for family, condition, start in (("reaction", "direct", "direct"),
                                                  ("bounce", "direct", "direct"),
                                                  ("bounce", "bank", "bank"),
                                                  ("perturbation", "clean", "direct"),
                                                  ("perturbation", "perturbed", "direct")):
                    position, velocity = starts[start]
                    disturbed = condition == "perturbed"
                    cases.append({
                        "case_id": f"{family}-{cell}-{condition}", "pair_id": f"{family}-{cell}",
                        "family": family, "condition": condition, "seed": seed,
                        "nominal_ttc_s": ttc, "speed_m_s": speed,
                        "position_m": [sign * position[0], position[1]],
                        "velocity_m_s": [sign * velocity[0], velocity[1]], "spin_rad_s": 0.0,
                        "perturb_time_s": config.perturb_time_fraction * ttc if disturbed else None,
                        "delta_velocity_m_s": [sign * config.perturb_delta_v_m_s[0],
                                                side * config.perturb_delta_v_m_s[1]] if disturbed else [0.0, 0.0],
                    })
    return seal_manifest({"schema_version": 1, "protocol": config.to_dict(),
                          "assets": copy.deepcopy(assets), "asset_sha256": fingerprint(assets),
                          "prepared": False, "reference_backend": None, "cases": cases})


def validate_manifest(manifest: dict, *, require_prepared: bool = False) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported suite schema")
    payload = dict(manifest)
    digest = payload.pop("suite_sha256", None)
    if digest != fingerprint(payload):
        raise ValueError("suite digest mismatch; never edit a frozen manifest")
    config = BenchmarkConfig.from_dict(manifest["protocol"])
    geom = Geometry.from_assets(manifest["assets"])
    if manifest.get("asset_sha256") != fingerprint(manifest["assets"]):
        raise ValueError("asset fingerprint mismatch")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite must have cases")
    if type(manifest.get("prepared")) is not bool:
        raise ValueError("prepared must be a boolean")
    # A valid hash alone cannot establish completeness: an accidentally
    # re-sealed subset must not remove a difficult cell or entire family.
    proposals = {c["case_id"]: c for c in generate_manifest(config, manifest["assets"])["cases"]}
    if len(cases) != len(proposals) or {c.get("case_id") for c in cases} != set(proposals):
        raise ValueError("case grid is incomplete or differs from the configured suite")
    prepared = manifest["prepared"] or require_prepared
    seen, pairs = set(), {}
    for case in cases:
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in case_id):
            raise ValueError("unsafe case_id")
        if case_id in seen:
            raise ValueError("duplicate case_id")
        seen.add(case_id)
        proposal = proposals[case_id]
        for key in ("pair_id", "family", "condition", "seed", "nominal_ttc_s", "speed_m_s",
                    "velocity_m_s", "spin_rad_s", "perturb_time_s", "delta_velocity_m_s"):
            if case.get(key) != proposal[key]:
                raise ValueError(f"case {case_id} differs from its configured proposal in {key}")
        if not prepared and case.get("position_m") != proposal["position_m"]:
            raise ValueError("unprepared position differs from its generated proposal")
        if prepared and case["position_m"][1] != proposal["position_m"][1]:
            raise ValueError("reference calibration may change only launch X")
        expected_conditions = {"reaction": {"direct"}, "bounce": {"direct", "bank"},
                               "perturbation": {"clean", "perturbed"}}
        if case.get("condition") not in expected_conditions.get(case.get("family"), set()):
            raise ValueError("unknown case family/condition")
        if type(case.get("seed")) is not int or case["seed"] < 0:
            raise ValueError("invalid case seed")
        for key in ("position_m", "velocity_m_s", "delta_velocity_m_s"):
            if len(case[key]) != 2:
                raise ValueError(f"{key} must be 2D")
            for value in case[key]:
                finite_number(value, key)
        if abs(case["position_m"][0]) >= geom.x_limit_m or abs(case["position_m"][1]) >= geom.y_limit_m:
            raise ValueError("serve outside physical field")
        if config.direction * case["velocity_m_s"][0] >= 0:
            raise ValueError("serve must travel toward controlled arm")
        if finite_number(case["speed_m_s"], "speed") <= 0 or finite_number(case["nominal_ttc_s"], "TTC") <= 0:
            raise ValueError("nonpositive case speed/TTC")
        pt = case["perturb_time_s"]
        if case["condition"] == "perturbed":
            if pt is None or not 0 < finite_number(pt, "perturb time") < case["nominal_ttc_s"]:
                raise ValueError("perturbation must occur before nominal intercept")
        elif pt is not None or any(case["delta_velocity_m_s"]):
            raise ValueError("clean cases cannot have perturbations")
        pairs.setdefault((case["family"], case["pair_id"]), []).append(case)
        if prepared:
            reference = case.get("reference", {})
            _validate_reference(reference, geom)
            if abs(reference["ttc_s"] - case["nominal_ttc_s"]) > config.reference_ttc_tolerance_s:
                raise ValueError("reference TTC exceeds the configured matching tolerance")
            if reference.get("side_bounces") != int(case["condition"] == "bank"):
                raise ValueError("reference bounce count differs from condition")
            if case["condition"] == "perturbed":
                disturbed = case.get("perturbed_reference", {})
                _validate_reference(disturbed, geom)
                if disturbed.get("perturbation_applied") is not True or pt >= min(reference["ttc_s"], disturbed["ttc_s"]):
                    raise ValueError("perturbed reference must actually apply the change before interception")
    for (family, _), rows in pairs.items():
        want = {"reaction": {"direct"}, "bounce": {"direct", "bank"},
                "perturbation": {"clean", "perturbed"}}[family]
        if len(rows) != len(want) or {r["condition"] for r in rows} != want:
            raise ValueError("incomplete or duplicate matched pair")
        if family == "perturbation":
            for key in ("seed", "position_m", "velocity_m_s", "nominal_ttc_s", "speed_m_s"):
                if rows[0][key] != rows[1][key]:
                    raise ValueError(f"perturbation pair differs in {key}")
    if prepared and (manifest.get("prepared") is not True or manifest.get("reference_backend") != "sapien-physx"):
        raise ValueError("run requires a prepared PhysX manifest")


def _validate_reference(reference, geometry):
    if not isinstance(reference, dict) or reference.get("valid") is not True or reference.get("backend") != "sapien-physx":
        raise ValueError("case lacks valid PhysX reference")
    if finite_number(reference.get("ttc_s"), "reference TTC") <= 0:
        raise ValueError("invalid reference TTC")
    if abs(finite_number(reference.get("intercept_y_m"), "reference intercept Y")) > geometry.y_limit_m:
        raise ValueError("reference intercept outside playing width")
    if finite_number(reference.get("intercept_speed_m_s"), "reference speed") <= 0:
        raise ValueError("reference must arrive with nonzero speed")
    if type(reference.get("side_bounces")) is not int or reference["side_bounces"] < 0:
        raise ValueError("reference bounce count must be a nonnegative integer")


def write_manifest(path: str | Path, manifest: dict) -> None:
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        file.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_manifest(path: str | Path, *, require_prepared: bool = False) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        manifest = json.load(file)
    validate_manifest(manifest, require_prepared=require_prepared)
    return manifest
