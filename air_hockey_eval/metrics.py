"""Physical event scoring. Ground truth is never part of a policy observation."""
from __future__ import annotations

from dataclasses import dataclass

from .schema import BenchmarkConfig, finite_number


@dataclass(frozen=True)
class FrameState:
    time_s: float
    puck_xy_m: tuple[float, float]
    puck_velocity_m_s: tuple[float, float]
    contact: bool = False
    violations: tuple[str, ...] = ()
    opponent_contact: bool = False
    out_of_bounds: bool = False


class EpisodeScorer:
    def __init__(self, config: BenchmarkConfig, *, y_limit_m: float):
        if finite_number(y_limit_m, "y_limit") <= config.return_y_margin_m:
            raise ValueError("return width has no clearance")
        self.config = config
        self.y_limit = y_limit_m - config.return_y_margin_m
        self.previous = None
        self.contact_time = None
        self.return_time = None
        self.return_y = None
        self.return_speed = None
        self.violations = set()
        self.failure = None
        self.last_time = 0.0

    def observe(self, frame: FrameState) -> None:
        t = finite_number(frame.time_s, "frame time")
        for values in (frame.puck_xy_m, frame.puck_velocity_m_s):
            if len(values) != 2:
                raise ValueError("puck truth must be 2D")
            for value in values:
                finite_number(value, "puck truth")
        if t < 0 or (self.previous is not None and t <= self.previous.time_s):
            raise ValueError("physics timestamps must strictly increase")
        self.last_time = t
        self.violations.update(frame.violations)
        if frame.contact and self.contact_time is None:
            self.contact_time = t
        if frame.opponent_contact:
            self.fail("opponent_interference")
        if frame.out_of_bounds:
            self.fail("out_of_bounds")
        if self.previous is not None and self.return_time is None and self.contact_time is not None:
            direction = self.config.direction
            x0 = direction * self.previous.puck_xy_m[0]
            x1 = direction * frame.puck_xy_m[0]
            line = self.config.return_line_x_m
            if x0 < line <= x1:
                fraction = (line - x0) / (x1 - x0)
                crossing_time = self.previous.time_s + fraction * (t - self.previous.time_s)
                y = self.previous.puck_xy_m[1] + fraction * (frame.puck_xy_m[1] - self.previous.puck_xy_m[1])
                vx = direction * (self.previous.puck_velocity_m_s[0] + fraction *
                                  (frame.puck_velocity_m_s[0] - self.previous.puck_velocity_m_s[0]))
                if (crossing_time + 1e-10 >= self.contact_time and abs(y) <= self.y_limit
                        and vx >= self.config.min_return_speed_m_s
                        and crossing_time <= self.config.episode_timeout_s):
                    self.return_time, self.return_y, self.return_speed = crossing_time, y, vx
        self.previous = frame

    def fail(self, reason: str) -> None:
        if self.failure is None:
            self.failure = reason

    def result(self) -> dict:
        returned = self.return_time is not None
        success = returned and not self.violations and self.failure is None
        return {"success": bool(success), "hit": self.contact_time is not None,
                "returned": returned, "violations": sorted(self.violations),
                "failure_reason": None if success else (self.failure or
                                  ("safety_violation" if self.violations else
                                   "no_return" if self.contact_time is not None else "no_contact")),
                "contact_time_s": self.contact_time, "return_time_s": self.return_time,
                "return_y_m": self.return_y, "return_speed_m_s": self.return_speed,
                "duration_s": self.last_time}
