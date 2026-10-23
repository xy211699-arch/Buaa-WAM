"""Resumable suite execution; loading a learned policy is explicit and opt-in."""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path

from .policies import make_policy, validate_policy_config
from .runner import run_episode
from .schema import BenchmarkConfig, canonical_json, fingerprint
from .storage import RunStore
from .suite import validate_manifest


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    # Include the actual scene/contact/robot builders and shared policy bridge,
    # not just the evaluator wrapper. All methods hash the same file set.
    files = set((root / "air_hockey_eval").glob("*.py"))
    for relative in ("envs/_air_hockey", "envs/robot", "XPolicyLab/utils", "XPolicyLab/client_server/ws"):
        files.update((root / relative).rglob("*.py"))
    for relative in ("XPolicyLab/policy/ACT", "XPolicyLab/policy/DP", "XPolicyLab/policy/FastWAM"):
        files.update((root / relative).rglob("*.py"))
    files.add(root / "env_cfg/robot/_robot_info.json")
    files.add(root / "env_cfg/air_hockey_eval_ur5.yml")
    return fingerprint({p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in sorted(files)})


def execute_suite(manifest: dict, policy_config: dict, directory, *, hardware_label: str,
                  mode: str = "latency", resume: bool = False, save_trace: bool = True,
                  backend_factory=None, policy_factory=None, progress=None) -> dict:
    validate_manifest(manifest, require_prepared=True)
    validate_policy_config(policy_config)
    if any(policy_config[key].strip().upper().startswith("REPLACE_")
           for key in ("artifact_id", "training_data_id")):
        raise ValueError("replace placeholder checkpoint and training dataset identities before run")
    if not hardware_label.strip() or mode not in ("latency", "zero_latency"):
        raise ValueError("explicit hardware label and valid latency mode required")
    cfg = BenchmarkConfig.from_dict(manifest["protocol"])
    declared_history = policy_config.get("observation_history_frames")
    if declared_history is not None and declared_history != cfg.history_frames:
        raise ValueError("declared observation history differs from the shared benchmark protocol")
    identity = {"suite_sha256": manifest["suite_sha256"], "protocol_sha256": fingerprint(manifest["protocol"]),
                "asset_sha256": manifest["asset_sha256"], "code_sha256": code_fingerprint(),
                "policy_name": policy_config["name"], "training_seed": policy_config["training_seed"],
                "artifact_id": policy_config["artifact_id"], "training_data_id": policy_config["training_data_id"],
                "policy_config_sha256": fingerprint(policy_config), "action_semantics": "joint_drive_target",
                "hardware_label": hardware_label, "mode": mode, "save_trace": bool(save_trace),
                "python": platform.python_version(), "platform": platform.platform()}
    ids = [c["case_id"] for c in manifest["cases"]]
    directory = Path(directory)
    with RunStore(directory, identity, ids, resume=resume) as store:
        pending = [c for c in manifest["cases"] if c["case_id"] not in store.completed]
        if not pending:
            store.finalize()
            return {"complete": True, "completed": len(ids), "resumed": len(ids)}
        if backend_factory is None:
            from .sapien_backend import SapienBackend
            backend_factory = SapienBackend
        if policy_factory is None:
            policy_factory = lambda options, config: make_policy(options, controlled_side=config.controlled_side)
        backend, policy, unresponsive = None, None, False
        try:
            backend = backend_factory(manifest["assets"], cfg)
            policy = policy_factory(policy_config, cfg)
            for case in pending:
                trace_file = None
                temporary = directory / "traces" / (case["case_id"] + ".jsonl.partial")
                try:
                    if save_trace:
                        temporary.parent.mkdir(parents=True, exist_ok=True)
                        trace_file = temporary.open("w", encoding="utf-8", newline="\n")
                    def trace(row):
                        trace_file.write(canonical_json(row) + "\n")
                    result = run_episode(backend, policy, case, cfg, mode=mode,
                                         trace=trace if trace_file else None)
                    result.update({key: case[key] for key in ("case_id", "pair_id", "family", "condition",
                                                              "nominal_ttc_s", "speed_m_s")})
                    result["reference"] = case["reference"]
                    if trace_file:
                        trace_file.flush(); os.fsync(trace_file.fileno()); trace_file.close()
                        trace_file = None
                        os.replace(temporary, temporary.with_suffix(""))
                    store.append(result)
                    if progress:
                        progress(len(store.completed), len(ids), case["case_id"], result)
                    if result.get("policy_unresponsive"):
                        unresponsive = True
                        raise RuntimeError("policy worker is unresponsive; failed case saved, stop and resume in a fresh process")
                finally:
                    if trace_file:
                        trace_file.close()
            store.finalize()
            return {"complete": True, "completed": len(ids), "resumed": len(ids) - len(pending)}
        except Exception as error:
            # Infrastructure/startup failures never become invented policy scores.
            record = {"complete": False, "error_type": type(error).__name__, "error": str(error),
                      "completed": len(store.completed), "total": len(ids)}
            (directory / "interruption.json").write_text(canonical_json(record) + "\n", encoding="utf-8")
            raise
        finally:
            if backend is not None:
                backend.close()
            # A timed-out RPC must not race a second close RPC on the same client.
            if policy is not None and not unresponsive and callable(getattr(policy, "close", None)):
                policy.close()
