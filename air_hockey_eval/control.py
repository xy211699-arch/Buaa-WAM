"""Bounded timestamped action chunks; latency never shifts an old plan forward."""
from __future__ import annotations

from collections import deque

import numpy as np

from .schema import finite_number


def validate_action_chunk(actions) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim == 1 and array.shape == (6,):
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != 6 or array.shape[0] < 1 or not np.isfinite(array).all():
        raise ValueError("policy must return a finite, nonempty (steps, 6) drive-target array")
    return array.copy()


class TimedActionQueue:
    def __init__(self, *, action_hz: int, execution_steps: int, initial_command):
        if action_hz <= 0 or execution_steps <= 0:
            raise ValueError("positive action_hz and execution_steps required")
        self.period = 1.0 / action_hz
        self.execution_steps = execution_steps
        self.current = validate_action_chunk(initial_command)[0]
        self.source = None
        self._queue = deque()
        self._coverage_until = -float("inf")
        self._sequence = 0

    def accept(self, actions, *, observed_s: float, ready_s: float, now_s: float) -> dict:
        actions = validate_action_chunk(actions)[:self.execution_steps]
        for value in (observed_s, ready_s, now_s):
            finite_number(value, "action time")
        if observed_s < 0 or ready_s < observed_s - 1e-9 or now_s < ready_s - 1e-9:
            raise ValueError("actions cannot arrive before observation or be accepted before ready")
        self._sequence += 1
        pending = []
        for i, action in enumerate(actions):
            stamp = observed_s + i * self.period
            if stamp + 1e-9 >= now_s:
                source = {"sequence": self._sequence, "step": i, "scheduled_s": stamp}
                pending.append((stamp, action, source))
        if pending:
            self._queue = deque(pending)
            self._coverage_until = observed_s + len(actions) * self.period
        return {"sequence": self._sequence, "accepted_steps": len(pending),
                "expired_steps": len(actions) - len(pending)}

    def command_at(self, now_s: float):
        while self._queue and self._queue[0][0] <= now_s + 1e-9:
            _, self.current, self.source = self._queue.popleft()
        return self.current.copy(), self.source

    def has_coverage(self, now_s: float) -> bool:
        return now_s < self._coverage_until - 1e-9

    @property
    def pending_steps(self) -> int:
        return len(self._queue)
