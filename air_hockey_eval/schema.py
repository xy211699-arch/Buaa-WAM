"""Validated SI-unit protocol shared by generation, execution and analysis."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 20261023
    repeats_per_cell: int = 20
    speeds_m_s: tuple[float, ...] = (0.9, 1.05, 1.2)
    ttc_s: tuple[float, ...] = (0.8, 1.0, 1.1)
    controlled_side: str = "left"
    intercept_line_x_m: float = -0.6  # canonical left-arm frame
    return_line_x_m: float = 0.55
    min_return_speed_m_s: float = 0.05
    return_y_margin_m: float = 0.01
    physics_dt_s: float = 0.001
    observation_hz: int = 30
    action_hz: int = 30
    history_frames: int = 3
    camera_width: int = 640
    camera_height: int = 360
    camera_height_m: float = 2.0
    camera_fovy_deg: float = 55.0
    camera_aliases: dict[str, str] = field(default_factory=lambda: {"cam_head": "cam_head"})
    replan_interval_s: float = 0.1
    execution_steps: int = 6
    extra_latency_s: float = 0.0
    policy_timeout_s: float = 5.0
    episode_timeout_s: float = 5.0
    max_command_speed_rad_s: tuple[float, ...] = (3.14,) * 6
    workspace_x_m: tuple[float, float] = (-1.04, -0.25)
    workspace_y_m: tuple[float, float] = (-0.50, 0.50)
    striker_height_tolerance_m: float = 0.02
    perturb_time_fraction: float = 0.25
    perturb_delta_v_m_s: tuple[float, float] = (-0.15, 0.20)
    reference_ttc_tolerance_s: float = 0.12
    success_threshold: float = 0.8
    instruction: str = "Return the incoming puck to the opposite side."

    def __post_init__(self):
        for name in ("seed", "repeats_per_cell", "observation_hz", "action_hz",
                     "history_frames", "execution_steps", "camera_width", "camera_height"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a positive integer (seed may be zero)")
        for name in ("physics_dt_s", "min_return_speed_m_s", "replan_interval_s",
                     "policy_timeout_s", "episode_timeout_s", "striker_height_tolerance_m",
                     "camera_height_m", "camera_fovy_deg", "reference_ttc_tolerance_s"):
            if finite_number(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("extra_latency_s", "return_y_margin_m"):
            if finite_number(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("speeds_m_s", "ttc_s"):
            values = tuple(getattr(self, name))
            if not values or any(finite_number(v, name) <= 0 for v in values):
                raise ValueError(f"{name} must contain positive finite values")
            if list(values) != sorted(set(values)):
                raise ValueError(f"{name} must be sorted with no duplicates")
            object.__setattr__(self, name, values)
        limits = tuple(self.max_command_speed_rad_s)
        if len(limits) != 6 or any(finite_number(v, "joint speed") <= 0 for v in limits):
            raise ValueError("max_command_speed_rad_s must contain six positive limits")
        object.__setattr__(self, "max_command_speed_rad_s", limits)
        for name in ("workspace_x_m", "workspace_y_m"):
            pair = tuple(getattr(self, name))
            if len(pair) != 2 or finite_number(pair[0], name) >= finite_number(pair[1], name):
                raise ValueError(f"{name} must be an increasing pair")
            object.__setattr__(self, name, pair)
        delta = tuple(self.perturb_delta_v_m_s)
        if len(delta) != 2 or not any(delta):
            raise ValueError("perturb_delta_v_m_s must be a nonzero 2D vector")
        for v in delta:
            finite_number(v, "perturbation")
        object.__setattr__(self, "perturb_delta_v_m_s", delta)
        if self.controlled_side not in ("left", "right"):
            raise ValueError("controlled_side must be left or right")
        if not 0 < finite_number(self.perturb_time_fraction, "perturb_time_fraction") < 1:
            raise ValueError("perturb_time_fraction must be inside (0, 1)")
        if not 0 < finite_number(self.success_threshold, "success_threshold") < 1:
            raise ValueError("success_threshold must be inside (0, 1)")
        if self.camera_fovy_deg >= 170:
            raise ValueError("camera_fovy_deg must be below 170")
        if self.observation_hz * self.physics_dt_s > 1 or self.action_hz * self.physics_dt_s > 1:
            raise ValueError("sampling rates cannot exceed physics rate")
        if finite_number(self.intercept_line_x_m, "intercept_line") >= finite_number(self.return_line_x_m, "return_line"):
            raise ValueError("intercept line must precede return line")
        if self.episode_timeout_s <= max(self.ttc_s):
            raise ValueError("episode must outlast every nominal TTC")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction must be nonempty")
        aliases = dict(self.camera_aliases)
        if not aliases or aliases.get("cam_head") != "cam_head" or any(v != "cam_head" for v in aliases.values()):
            raise ValueError("camera aliases must explicitly refer to the sole real cam_head")
        if any(k not in {"cam_head", "cam_left_wrist", "cam_right_wrist"} for k in aliases):
            raise ValueError("unknown camera alias")
        object.__setattr__(self, "camera_aliases", aliases)

    @classmethod
    def from_dict(cls, values: dict) -> "BenchmarkConfig":
        if not isinstance(values, dict):
            raise ValueError("benchmark config must be a mapping")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown benchmark fields: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict:
        return json.loads(canonical_json(asdict(self)))

    @property
    def direction(self) -> float:
        """World sign of a successful outgoing return."""
        return 1.0 if self.controlled_side == "left" else -1.0


@dataclass(frozen=True)
class Geometry:
    x_limit_m: float
    y_limit_m: float
    table_height_m: float
    puck_radius_m: float
    puck_height_m: float
    goal_half_width_m: float

    @classmethod
    def from_assets(cls, assets: dict) -> "Geometry":
        table, puck = assets["table"], assets["puck"]
        radius = float(puck["diameter_m"]) / 2
        result = cls(float(table["length_m"]) / 2 - float(table["rail_width_m"]) - radius,
                     float(table["width_m"]) / 2 - float(table["rail_width_m"]) - radius,
                     float(table["top_height_m"]), radius, float(puck["thickness_m"]),
                     float(table["goal_width_m"]) / 2 - radius)
        if any(not math.isfinite(v) or v <= 0 for v in asdict(result).values()):
            raise ValueError("invalid physical geometry")
        return result
