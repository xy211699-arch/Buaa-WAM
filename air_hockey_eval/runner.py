"""One-in-flight asynchronous closed loop, shared by all learned policies.

While inference runs, physics executes the previously available controls. A
logical-time gate prevents a fast CPU simulator from outrunning elapsed model
wall time. A slow simulator still uses the measured observation-to-ready delay;
commands that have already expired are discarded rather than silently retimed.
"""
from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque

import numpy as np

from .control import TimedActionQueue, validate_action_chunk
from .metrics import EpisodeScorer
from .schema import BenchmarkConfig


class _Job:
    def __init__(self, function, *args):
        self.done = threading.Event()
        self.value, self.error, self.elapsed_s = None, None, None
        self.started = time.perf_counter()
        def execute():
            try:
                self.value = function(*args)
            except Exception as error:
                self.error = error
            finally:
                self.elapsed_s = time.perf_counter() - self.started
                self.done.set()
        self.thread = threading.Thread(target=execute, name="hockey-policy", daemon=True)
        self.thread.start()


_STATE_FIELDS = {f"{side}_{key}" for side in ("left", "right") for key in
                 ("arm_joint_state", "arm_joint_velocity", "ee_joint_state", "ee_pose", "tcp_pose")}


def policy_packet(history, config: BenchmarkConfig) -> dict:
    """Whitelist the observation boundary; privileged fields cannot leak."""
    frames = []
    for raw in history:
        camera = raw["vision"]["cam_head"]
        image = np.asarray(camera["color"])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("backend must provide uint8 HWC RGB")
        cameras = {}
        for alias in config.camera_aliases:
            cameras[alias] = {"color": image.copy(), "shape": image.shape[:2]}
            for key in ("intrinsic_matrix", "extrinsics_matrix"):
                if key in camera:
                    cameras[alias][key] = copy.deepcopy(camera[key])
        state = {key: copy.deepcopy(value) for key, value in raw.get("state", {}).items()
                 if key in _STATE_FIELDS}
        frames.append({"vision": cameras, "state": state, "timestamp_s": float(raw["timestamp_s"]),
                       "env_idx": 0, "data_format_version": "v1.0",
                       "instruction": config.instruction,
                       "additional_info": {"frequency": config.action_hz}})
    if not frames:
        raise ValueError("policy history is empty")
    while len(frames) < config.history_frames:
        frames.insert(0, copy.deepcopy(frames[0]))
    return {"frames": frames, "instruction": config.instruction,
            "action_hz": config.action_hz, "action_semantics": "joint_drive_target"}


def run_episode(backend, policy, case: dict, config: BenchmarkConfig, *,
                mode: str = "latency", trace=None) -> dict:
    if mode not in ("latency", "zero_latency"):
        raise ValueError("mode must be latency or zero_latency")
    backend.reset(case)
    scorer = EpisodeScorer(config, y_limit_m=backend.y_limit_m)
    queue = TimedActionQueue(action_hz=config.action_hz, execution_steps=config.execution_steps,
                             initial_command=backend.initial_command())
    reset = _Job(policy.reset)
    if not reset.done.wait(config.policy_timeout_s):
        return dict(scorer.result(), failure_reason="policy_reset_timeout", policy_unresponsive=True,
                    timings=[], perturbation_applied=False, starvation_fraction=1.0)
    if reset.error is not None:
        scorer.fail("policy_reset_error")
        return dict(scorer.result(), policy_error=str(reset.error), timings=[],
                    perturbation_applied=False, starvation_fraction=1.0)
    history = deque(maxlen=config.history_frames)
    next_capture, next_request = 0.0, 0.0
    latest_capture_cost, last_requested_stamp = 0.0, -1.0
    pending, pending_meta = None, None
    timings, expired, starving, steps = [], 0, 0, 0
    applied_source, applied_perturbation = None, False
    policy_error = None
    unresponsive = False
    while True:
        state = backend.frame()
        scorer.observe(state)
        # A treatment must actually precede the hit. Check before the success
        # early exit, including a hit and return completed before planned time.
        if (case.get("perturb_time_s") is not None and not applied_perturbation
                and scorer.contact_time is not None):
            scorer.fail("contact_before_perturbation")
        if trace is not None:
            trace({"time_s": state.time_s, "puck_xy_m": state.puck_xy_m,
                   "puck_velocity_m_s": state.puck_velocity_m_s, "contact": state.contact,
                   "violations": list(state.violations), "command": queue.current.tolist(),
                   "perturbation_applied": applied_perturbation})
        if scorer.result()["success"] or scorer.failure is not None or scorer.violations:
            break
        t = state.time_s
        if t >= config.episode_timeout_s - 1e-9:
            scorer.fail("timeout")
            break
        if t + 1e-9 >= next_capture:
            start = time.perf_counter()
            raw = backend.observe()
            raw["timestamp_s"] = t
            history.append(raw)
            latest_capture_cost = time.perf_counter() - start
            next_capture = (math.floor((t + 1e-9) * config.observation_hz) + 1) / config.observation_hz
        if pending is not None:
            if not pending.done.is_set() and time.perf_counter() - pending.started > config.policy_timeout_s:
                scorer.fail("policy_timeout")
                unresponsive = True
                break
            if pending.done.is_set():
                if pending.error is not None:
                    scorer.fail("policy_error")
                    policy_error = repr(pending.error)
                    break
                measured = pending_meta["capture_cost_s"] + pending.elapsed_s
                delay = (measured if mode == "latency" else 0.0) + config.extra_latency_s
                ready_s = pending_meta["request_s"] + delay
                if t + 1e-9 >= ready_s:
                    try:
                        accepted = queue.accept(pending.value, observed_s=pending_meta["observed_s"],
                                                ready_s=ready_s, now_s=t)
                    except (ValueError, TypeError) as error:
                        scorer.fail("invalid_action")
                        policy_error = repr(error)
                        break
                    expired += accepted["expired_steps"]
                    timings.append(dict(pending_meta, compute_s=pending.elapsed_s, ready_s=ready_s,
                                        observation_to_ready_s=ready_s - pending_meta["observed_s"], received_s=t,
                                        first_applied_s=None, **accepted))
                    pending, pending_meta = None, None
        if (pending is None and history and t + 1e-9 >= next_request
                and history[-1]["timestamp_s"] > last_requested_stamp + 1e-9):
            start = time.perf_counter()
            packet = policy_packet(history, config)
            cost = latest_capture_cost + time.perf_counter() - start
            last_requested_stamp = history[-1]["timestamp_s"]
            pending_meta = {"observed_s": last_requested_stamp, "request_s": t, "capture_cost_s": cost}
            pending = _Job(policy.predict, packet)
            next_request = t + config.replan_interval_s
        if pending is not None and not pending.done.is_set():
            if mode == "zero_latency":
                pending.done.wait(min(0.001, config.policy_timeout_s))
                # Do not observe a duplicate state on the next loop iteration.
                while not pending.done.wait(0.001):
                    if time.perf_counter() - pending.started > config.policy_timeout_s:
                        break
            else:
                while not pending.done.is_set():
                    elapsed = time.perf_counter() - pending.started
                    allowed = pending_meta["request_s"] + pending_meta["capture_cost_s"] + elapsed
                    if t + config.physics_dt_s <= allowed + 1e-9:
                        break
                    pending.done.wait(min(0.001, max(0.0001, t + config.physics_dt_s - allowed)))
                    if time.perf_counter() - pending.started > config.policy_timeout_s:
                        break
                # Waiting occurs before the next physics step, never re-scoring this frame.
        if pending is not None and mode == "zero_latency" and pending.done.is_set():
            # Deliver at this logical time, without one extra physics step.
            if pending.error is not None:
                scorer.fail("policy_error")
                policy_error = repr(pending.error)
                break
            ready_s = pending_meta["request_s"] + config.extra_latency_s
            if t + 1e-9 >= ready_s:
                try:
                    accepted = queue.accept(pending.value, observed_s=pending_meta["observed_s"],
                                            ready_s=ready_s, now_s=t)
                except (ValueError, TypeError) as error:
                    scorer.fail("invalid_action"); policy_error = repr(error); break
                expired += accepted["expired_steps"]
                timings.append(dict(pending_meta, compute_s=pending.elapsed_s, ready_s=ready_s,
                                    observation_to_ready_s=ready_s - pending_meta["observed_s"], received_s=t,
                                    first_applied_s=None, **accepted))
                pending, pending_meta = None, None
        command, source = queue.command_at(t)
        if source is not None and source != applied_source:
            violations = backend.apply(command)
            scorer.violations.update(violations)
            applied_source = dict(source)
            for timing in reversed(timings):
                if timing["sequence"] == source["sequence"] and timing["first_applied_s"] is None:
                    timing["first_applied_s"] = t
                    timing["observation_to_apply_s"] = t - timing["observed_s"]
                    break
            if violations:
                scorer.fail("safety_violation")
                break
        pt = case.get("perturb_time_s")
        if pt is not None and not applied_perturbation and t + 1e-9 >= pt:
            if scorer.contact_time is not None:
                scorer.fail("contact_before_perturbation")
                break
            backend.perturb(case["delta_velocity_m_s"])
            applied_perturbation = True
        starving += not queue.has_coverage(t)
        steps += 1
        backend.step()
    if pending is not None and not pending.done.wait(config.policy_timeout_s):
        unresponsive = True
        scorer.fail("policy_timeout")
    result = scorer.result()
    result.update(timings=timings, expired_action_steps=expired,
                  starvation_fraction=starving / steps if steps else 1.0,
                  perturbation_applied=applied_perturbation,
                  policy_unresponsive=unresponsive, policy_error=policy_error, mode=mode)
    return result
