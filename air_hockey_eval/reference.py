"""Prepare all cases using policy-free PhysX rollouts, or reject the suite."""
from __future__ import annotations

import copy

from .schema import BenchmarkConfig, Geometry, fingerprint
from .suite import seal_manifest, validate_manifest


def prepare_manifest(manifest: dict, *, reference_provider=None, progress=None) -> dict:
    validate_manifest(manifest)
    if manifest["prepared"]:
        raise ValueError("already prepared; start from the original proposals")
    cfg = BenchmarkConfig.from_dict(manifest["protocol"])
    geometry = Geometry.from_assets(manifest["assets"])
    owned = None
    if reference_provider is None:
        from .sapien_backend import PhysXReference
        owned = PhysXReference(manifest["assets"], cfg)
        reference_provider = owned.rollout
    output = copy.deepcopy(manifest)
    cache = {}
    errors = []
    try:
        for index, case in enumerate(output["cases"]):
            key = fingerprint({k: case[k] for k in ("position_m", "velocity_m_s", "spin_rad_s", "nominal_ttc_s")})
            if key not in cache:
                candidate = copy.deepcopy(case)
                original = list(candidate["position_m"])
                ref = None
                for attempt in range(10):
                    ref = reference_provider(candidate, perturbed=False)
                    if not ref.get("valid"):
                        break
                    error = float(ref["ttc_s"]) - case["nominal_ttc_s"]
                    if abs(error) <= min(0.01, cfg.reference_ttc_tolerance_s):
                        break
                    # Preserve launch speed, varying only launch X to match measured TTC.
                    candidate["position_m"][0] += candidate["velocity_m_s"][0] * error * 0.8
                    if abs(candidate["position_m"][0]) >= geometry.x_limit_m:
                        ref = {"valid": False, "reason": "calibration left the legal table"}
                        break
                cache[key] = (candidate["position_m"], ref, original, attempt + 1)
            position, ref, original, attempts = cache[key]
            case["position_m"] = list(position)
            case["proposal_position_m"] = original
            case["calibration_attempts"] = attempts
            want_bounces = 1 if case["condition"] == "bank" else 0
            if (not ref or not ref.get("valid") or ref.get("backend") != "sapien-physx"
                    or abs(float(ref["ttc_s"]) - case["nominal_ttc_s"]) > cfg.reference_ttc_tolerance_s
                    or ref.get("side_bounces") != want_bounces):
                errors.append(f"{case['case_id']}: reference preflight failed: {ref}")
                continue
            case["reference"] = copy.deepcopy(ref)
            if case["condition"] == "perturbed":
                disturbed = reference_provider(case, perturbed=True)
                if (not disturbed.get("valid") or case["perturb_time_s"] >= ref["ttc_s"]
                        or disturbed.get("backend") != "sapien-physx"
                        or disturbed.get("perturbation_applied") is not True):
                    errors.append(f"{case['case_id']}: perturbed reference preflight failed: {disturbed}")
                    continue
                case["perturbed_reference"] = disturbed
            if progress:
                progress(index + 1, len(output["cases"]), case["case_id"])
    finally:
        if owned is not None:
            owned.close()
    if errors:
        raise ValueError("reference preflight rejected the suite; no cases were filtered:\n" + "\n".join(errors))
    output["prepared"] = True
    output["reference_backend"] = "sapien-physx"
    output["proposal_sha256"] = manifest["suite_sha256"]
    output = seal_manifest(output)
    validate_manifest(output, require_prepared=True)
    return output
