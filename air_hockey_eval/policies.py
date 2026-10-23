"""Explicit, lazy policy boundaries. No policy is loaded during CLI validation."""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

from .control import validate_action_chunk
from .schema import canonical_json


def action_rows_to_chunk(actions, controlled_side: str):
    if controlled_side not in ("left", "right"):
        raise ValueError("controlled side must be left or right")
    key = f"{controlled_side}_arm_joint_state"
    if isinstance(actions, dict):
        actions = [actions]
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError("XPolicyLab must return nonempty action dictionaries")
    try:
        return validate_action_chunk([row[key] for row in actions])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"policy action must contain six {key} drive targets") from error


class XPolicyPolicy:
    """Use update_obs/get_action locally, or the identical RPCs over websocket."""
    def __init__(self, model, *, controlled_side: str, remote: bool = False,
                 require_act_chunk: bool = False):
        self.model, self.controlled_side, self.remote = model, controlled_side, remote
        self.require_act_chunk = require_act_chunk
        self.last_stamp = -float("inf")

    def reset(self):
        self.last_stamp = -float("inf")
        self.verify_capabilities()
        if self.remote:
            self.model.call(func_name="reset")
        else:
            self.model.reset()

    def verify_capabilities(self):
        if self.require_act_chunk:
            capability = (self.model.call(func_name="get_eval_capabilities") if self.remote else
                          self.model.get_eval_capabilities() if callable(getattr(self.model, "get_eval_capabilities", None)) else {})
            if not isinstance(capability, dict) or capability.get("adapter") != "act-native-chunk-v1":
                raise ValueError("ACT requires the native chunk benchmark adapter/server")

    def predict(self, packet):
        for frame in packet["frames"]:
            stamp = frame["timestamp_s"]
            if stamp > self.last_stamp + 1e-9:
                if self.remote:
                    self.model.call(func_name="update_obs", obs=frame)
                else:
                    self.model.update_obs(frame)
                self.last_stamp = stamp
        method = "get_action_chunk" if self.require_act_chunk else "get_action"
        actions = self.model.call(func_name=method) if self.remote else getattr(self.model, method)()
        return action_rows_to_chunk(actions, self.controlled_side)

    def close(self):
        if self.remote:
            self.model.close()


def validate_policy_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("policy config must be a mapping")
    allowed = {"name", "backend", "artifact_id", "training_data_id", "training_seed",
               "action_semantics", "factory", "kwargs", "model_config", "endpoint",
               "request_timeout_s", "observation_history_frames"}
    if set(config) - allowed:
        raise ValueError(f"unknown policy fields: {sorted(set(config) - allowed)}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(config.get("name", ""))):
        raise ValueError("policy name must be a safe identifier")
    for key in ("artifact_id", "training_data_id"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must identify the actual checkpoint/dataset version")
    if type(config.get("training_seed")) is not int:
        raise ValueError("training_seed must identify an independent training run")
    if config.get("action_semantics") != "joint_drive_target":
        raise ValueError("benchmark requires actual joint_drive_target supervision, not next measured state")
    kind = config.get("backend")
    if kind == "python":
        if not isinstance(config.get("factory"), str) or ":" not in config["factory"]:
            raise ValueError("python policy requires module:factory")
    elif kind == "xpolicy_local":
        if config["name"] not in ("FastWAM", "DP", "ACT") or not isinstance(config.get("model_config"), dict):
            raise ValueError("local policy requires FastWAM/DP/ACT and its model_config")
        if config["model_config"].get("allow_dummy_policy"):
            raise ValueError("dummy policies cannot produce benchmark results")
        if config["name"] == "ACT" and config["model_config"].get("temporal_agg") is not False:
            raise ValueError("ACT common chunk protocol requires explicit temporal_agg: false")
    elif kind == "xpolicy_ws":
        if not isinstance(config.get("endpoint"), str) or not config["endpoint"].startswith(("ws://", "wss://")):
            raise ValueError("xpolicy_ws requires a websocket endpoint")
    else:
        raise ValueError("backend must be python, xpolicy_local or xpolicy_ws")
    canonical_json(config)


def make_policy(config: dict, *, controlled_side: str):
    validate_policy_config(config)
    kind = config["backend"]
    if kind == "python":
        module_name, factory_name = config["factory"].rsplit(":", 1)
        factory = getattr(importlib.import_module(module_name), factory_name)
        model = factory(**config.get("kwargs", {}))
        if not callable(getattr(model, "predict", None)) or not callable(getattr(model, "reset", None)):
            raise ValueError("policy factory must return reset()/predict(packet)")
        return model
    if kind == "xpolicy_local":
        model = load_xpolicy_model(config["name"], config["model_config"])
        return XPolicyPolicy(model, controlled_side=controlled_side, require_act_chunk=config["name"] == "ACT")
    # Existing client_server imports are checkout-relative, not pip-installed.
    root = Path(__file__).resolve().parents[1] / "XPolicyLab"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from client_server.ws.model_client import WsModelClient
    client = WsModelClient(url=config["endpoint"], evaluation_id="air-hockey-dynamic",
                           trial_id=f"{config['name']}-seed-{config['training_seed']}",
                           request_timeout_s=float(config.get("request_timeout_s", 5.0)),
                           max_connect_seconds=30.0, max_connect_attempts=3)
    policy = XPolicyPolicy(client, controlled_side=controlled_side, remote=True,
                            require_act_chunk=config["name"] == "ACT")
    try:
        policy.verify_capabilities()
    except Exception:
        client.close()
        raise
    return policy


def load_xpolicy_model(name: str, model_config: dict):
    """Load only on explicit run/server, with the actual robot dimensions.

    Native ACT reads ACT_ACTION_DIM while building its network, independently
    of model_cfg.action_dim. Supply both from the runtime registry and restore
    the process environment after construction.
    """
    from .cli import read_mapping
    if name not in ("FastWAM", "DP", "ACT") or not isinstance(model_config, dict):
        raise ValueError("expected FastWAM/DP/ACT deployment config")
    config = dict(model_config)
    if config.get("allow_dummy_policy"):
        raise ValueError("dummy policies cannot produce benchmark results")
    if config.get("action_type") != "joint":
        raise ValueError("UR5 benchmark requires joint action_type")
    profile_name = config.get("env_cfg_type")
    if not isinstance(profile_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", profile_name):
        raise ValueError("env_cfg_type must identify a registered UR5 environment profile")
    root = Path(__file__).resolve().parents[1]
    profile = read_mapping(root / "env_cfg" / (profile_name + ".yml"))
    registry = json.loads((root / "env_cfg/robot/_robot_info.json").read_text(encoding="utf-8"))
    dimensions = registry.get(profile.get("config", {}).get("robot"))
    if dimensions != {"arm_dim": [6, 6], "ee_dim": [2, 2]}:
        raise ValueError("benchmark requires two six-joint UR5 arms and two fingers per arm")
    if config.get("action_dim") not in (None, 16):
        raise ValueError("checkpoint action dimension must be 16 for dual UR5-WSG")
    config["action_dim"] = 16
    if name == "ACT" and config.get("temporal_agg") is not False:
        raise ValueError("ACT common chunk protocol requires explicit temporal_agg: false")
    previous = os.environ.get("ACT_ACTION_DIM")
    try:
        if name == "ACT":
            os.environ["ACT_ACTION_DIM"] = "16"
        module = importlib.import_module(f"XPolicyLab.policy.{name}.model")
        model = module.Model(config)
    finally:
        if name == "ACT":
            if previous is None:
                os.environ.pop("ACT_ACTION_DIM", None)
            else:
                os.environ["ACT_ACTION_DIM"] = previous
    if name == "ACT":
        from .act_chunk import ACTChunkModel
        model = ACTChunkModel(model)
    return model
