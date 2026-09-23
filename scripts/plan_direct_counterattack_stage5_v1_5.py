#!/usr/bin/env python3
"""
Stage 5 v1.5 with grazing-impact edge-glide classification and bank fallback.

Planner:
    randomized opponent-side serve
      -> snapshot one fixed episode initial state
      -> coarse search over:
           contact time (therefore contact x/y)
           + hit angle offset
           + hit speed
      -> feasibility gates (workspace / timing / IK)
      -> PhysX forward simulation
      -> fine search around coarse BEST
      -> restore the exact initial episode state
      -> replay only the final BEST in Viewer

Important:
- Randomness is per EPISODE, not per candidate.
- Every candidate for one seed sees exactly the same serve.
- This is still a privileged simulator expert, not RL.

Run:
    cd ~/Buaa-WAM
    conda activate hockey
    PYTHONPATH=. python scripts/plan_direct_counterattack_stage2.py --seed 1

Try several randomized serves:
    for s in 1 2 3 4 5; do
        PYTHONPATH=. python scripts/plan_direct_counterattack_stage2.py --seed "$s"
    done
"""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import sapien.core as sapien


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from air_hockey_assets import configure_air_hockey_physics, load_asset_config  # noqa: E402
import scripts.collect_episode_rgb_auto_v1 as collector  # noqa: E402
from scripts.collect_episode_rgb_auto_v1 import (  # noqa: E402
    configure_left_kinematic_tool,
    get_left_tool_follow_max_error,
    physics_step,
    populate_robot_preview,
    reset_left_tool_follow_stats,
    restore_hit_state,
    snapshot_hit_state,
)


# ============================================================
# IK / motion helpers
# ============================================================

def solve_striker_xy_ik(
    *,
    left_robot,
    left_pinocchio,
    left_ee_index,
    arm_qmask,
    left_limits,
    striker_ee_offset_xy,
    ee_z,
    ee_q,
    target_xy,
    initial_qpos=None,
):
    """Solve one striker XY pose. No drive target is changed here."""
    target_xy = np.asarray(target_xy, dtype=float)

    ee_world = sapien.Pose(
        [
            float(target_xy[0] - striker_ee_offset_xy[0]),
            float(target_xy[1] - striker_ee_offset_xy[1]),
            float(ee_z),
        ],
        ee_q,
    )
    ee_local = left_robot.get_root_pose().inv() * ee_world

    q0 = (
        np.asarray(left_robot.get_qpos(), dtype=float)
        if initial_qpos is None
        else np.asarray(initial_qpos, dtype=float)
    )

    q_sol, ik_ok, _ = left_pinocchio.compute_inverse_kinematics(
        left_ee_index,
        ee_local,
        initial_qpos=q0,
        active_qmask=arm_qmask,
        max_iterations=100,
    )

    valid = (
        bool(ik_ok)
        and np.all(np.isfinite(q_sol))
        and np.all(q_sol >= left_limits[:, 0] - 1e-5)
        and np.all(q_sol <= left_limits[:, 1] + 1e-5)
    )

    return valid, np.asarray(q_sol, dtype=float)


def set_arm_drive_targets(left_active_joints, arm_joint_names, q_sol):
    for joint, q_target in zip(left_active_joints, q_sol):
        if joint.name in arm_joint_names:
            joint.set_drive_target(float(q_target))


def timed_cartesian_segment(
    *,
    scene,
    timestep,
    left_robot,
    left_striker,
    left_pinocchio,
    left_ee_index,
    left_active_joints,
    arm_joint_names,
    arm_qmask,
    left_limits,
    striker_ee_offset_xy,
    ee_z,
    ee_q,
    target_xy,
    duration,
    puck,
    viewer=None,
    render=False,
    contact_gate=None,
):
    """
    Linear striker XY command over exact simulation time.

    IK/control: 100 Hz
    PhysX:      full timestep
    Viewer:     ~60 Hz if render=True
    """
    duration = float(duration)
    if duration <= 0.0:
        return {
            "ok": False,
            "min_distance": float("inf"),
            "contact": False,
            "contact_local_time": None,
        }

    start_xy = np.asarray(left_striker.get_pose().p[:2], dtype=float).copy()
    target_xy = np.asarray(target_xy, dtype=float)
    delta = target_xy - start_xy

    n_steps = max(1, int(round(duration / timestep)))
    control_stride = max(1, int(round(0.010 / timestep)))
    render_stride = max(1, int(round((1.0 / 60.0) / timestep)))

    min_distance = float("inf")
    contact = False
    contact_local_time = None

    for i in range(1, n_steps + 1):
        if viewer is not None and viewer.closed:
            break

        if i == 1 or i == n_steps or (i - 1) % control_stride == 0:
            alpha = i / n_steps
            cmd_xy = start_xy + alpha * delta

            valid, q_sol = solve_striker_xy_ik(
                left_robot=left_robot,
                left_pinocchio=left_pinocchio,
                left_ee_index=left_ee_index,
                arm_qmask=arm_qmask,
                left_limits=left_limits,
                striker_ee_offset_xy=striker_ee_offset_xy,
                ee_z=ee_z,
                ee_q=ee_q,
                target_xy=cmd_xy,
            )
            if not valid:
                return {
                    "ok": False,
                    "min_distance": min_distance,
                    "contact": contact,
                    "contact_local_time": contact_local_time,
                }

            set_arm_drive_targets(
                left_active_joints,
                arm_joint_names,
                q_sol,
            )

        physics_step(scene)

        puck_xy = np.asarray(puck.get_pose().p[:2], dtype=float)
        striker_xy = np.asarray(left_striker.get_pose().p[:2], dtype=float)
        d = float(np.linalg.norm(puck_xy - striker_xy))
        min_distance = min(min_distance, d)

        if (
            contact_gate is not None
            and not contact
            and d <= float(contact_gate)
        ):
            contact = True
            contact_local_time = i * timestep

        if (
            render
            and viewer is not None
            and i % render_stride == 0
        ):
            scene.update_render()
            viewer.render()

    return {
        "ok": True,
        "min_distance": min_distance,
        "contact": contact,
        "contact_local_time": contact_local_time,
    }


# ============================================================
# Random serve / puck prediction
# ============================================================

def sample_stage5_physics(
    rng,
    config,
):
    """
    Randomize one physically coherent table for ONE episode.

    Baseline YAML:
      table/puck dynamic friction ~0.02
      rail dynamic friction       ~0.10
      rail restitution            ~0.80

    Stage 5 randomizes around those provisional values.
    The sampled values remain fixed throughout the whole episode.
    """
    table_mu_d = float(rng.uniform(0.012, 0.030))
    puck_mu_d = float(rng.uniform(0.012, 0.030))
    rail_mu_d = float(rng.uniform(0.060, 0.140))
    rail_e = float(rng.uniform(0.720, 0.880))

    table_mu_s = float(
        np.clip(
            1.25 * table_mu_d,
            table_mu_d,
            0.060,
        )
    )
    puck_mu_s = float(
        np.clip(
            1.25 * puck_mu_d,
            puck_mu_d,
            0.060,
        )
    )
    rail_mu_s = float(
        np.clip(
            1.50 * rail_mu_d,
            rail_mu_d,
            0.250,
        )
    )

    config["table"]["surface_dynamic_friction"] = table_mu_d
    config["table"]["surface_static_friction"] = table_mu_s

    config["puck"]["dynamic_friction"] = puck_mu_d
    config["puck"]["static_friction"] = puck_mu_s

    config["table"]["rail_dynamic_friction"] = rail_mu_d
    config["table"]["rail_static_friction"] = rail_mu_s
    config["table"]["rail_restitution"] = rail_e

    return {
        "table_dynamic_friction": table_mu_d,
        "table_static_friction": table_mu_s,
        "puck_dynamic_friction": puck_mu_d,
        "puck_static_friction": puck_mu_s,
        "rail_dynamic_friction": rail_mu_d,
        "rail_static_friction": rail_mu_s,
        "rail_restitution": rail_e,
        "linear_damping": float(config["puck"]["linear_damping"]),
        "angular_damping": float(config["puck"]["angular_damping"]),
    }


def sample_stage5_serve(
    rng,
    *,
    serve_x,
    y_limit,
):
    """
    Stage-5 serve proposal.

    Unlike the old direct-inbound curriculum, side-wall contact is ALLOWED.
    Legality is decided later by an exact PhysX free-rollout.

    Randomization:
      speed      0.70 .. 1.20 m/s
      heading   -65 .. +65 deg relative to -X
      omega_z    -8 .. +8 rad/s
    """
    spawn_y_limit = min(
        0.42,
        max(0.05, float(y_limit) - 0.06),
    )

    serve_y = float(
        rng.uniform(
            -spawn_y_limit,
            +spawn_y_limit,
        )
    )
    speed = float(
        rng.uniform(
            0.70,
            1.20,
        )
    )
    angle_deg = float(
        rng.uniform(
            -65.0,
            +65.0,
        )
    )
    omega_z = float(
        rng.uniform(
            -8.0,
            +8.0,
        )
    )

    angle = np.deg2rad(angle_deg)

    vx = -speed * float(np.cos(angle))
    vy = +speed * float(np.sin(angle))

    return {
        "serve_x": float(serve_x),
        "serve_y": serve_y,
        "speed": speed,
        "angle_deg": angle_deg,
        "vx": float(vx),
        "vy": float(vy),
        "omega_z_rad_s": omega_z,
    }


def predict_puck_xy_no_bounce(
    *,
    x0,
    y0,
    vx,
    vy,
    t,
):
    """Legacy analytical helper; Stage 5 planning uses PhysX reference rollout."""
    return np.asarray(
        [
            float(x0 + vx * t),
            float(y0 + vy * t),
        ],
        dtype=float,
    )


def build_physx_puck_reference(
    *,
    scene,
    timestep,
    assembly,
    initial_state,
    puck,
    puck_body,
    horizon=2.2,
    stop_x=-0.72,
):
    """
    Build a privileged reference puck trajectory from the exact same
    episode snapshot using real PhysX stepping.

    This replaces the Stage-2-v1 assumption x=x0+vx*t, y=y0+vy*t.
    The reference therefore includes the simulator's actual friction /
    integration behavior.  We stop before the deep defensive region so the
    stationary home striker cannot contaminate the free-puck prediction.
    """
    restore_hit_state(assembly, initial_state)

    ts = [0.0]
    xys = [np.asarray(puck.get_pose().p[:2], dtype=float).copy()]
    vxys = [
        np.asarray(
            puck_body.get_linear_velocity()[:2],
            dtype=float,
        ).copy()
    ]
    omega_zs = [
        float(
            puck_body.get_angular_velocity()[2]
        )
    ]

    n_steps = max(1, int(round(float(horizon) / timestep)))

    for i in range(1, n_steps + 1):
        physics_step(scene)
        xy = np.asarray(puck.get_pose().p[:2], dtype=float).copy()
        vxy = np.asarray(
            puck_body.get_linear_velocity()[:2],
            dtype=float,
        ).copy()

        omega_z = float(
            puck_body.get_angular_velocity()[2]
        )

        ts.append(i * timestep)
        xys.append(xy)
        vxys.append(vxy)
        omega_zs.append(omega_z)

        # First arrival into the deep left-side planning boundary is enough.
        if xy[0] <= float(stop_x):
            break

        # Do not waste several simulated seconds on a serve that has already
        # lost essentially all translational speed.
        if (
            i * timestep > 0.25
            and float(np.linalg.norm(vxy)) < 0.025
        ):
            break

    restore_hit_state(assembly, initial_state)

    return {
        "t": np.asarray(ts, dtype=float),
        "xy": np.asarray(xys, dtype=float),
        "vxy": np.asarray(vxys, dtype=float),
        "omega_z": np.asarray(omega_zs, dtype=float),
    }


def reference_xy_at_time(reference, t):
    """Linear interpolation of the PhysX reference trajectory."""
    ts = reference["t"]
    xy = reference["xy"]
    t = float(t)

    if t < ts[0] - 1e-9 or t > ts[-1] + 1e-9:
        return None

    x = float(np.interp(t, ts, xy[:, 0]))
    y = float(np.interp(t, ts, xy[:, 1]))
    return np.asarray([x, y], dtype=float)


def reference_state_at_time(reference, t):
    """Interpolate privileged PhysX puck position + velocity at time t."""
    ts = reference["t"]
    xy = reference["xy"]
    vxy = reference["vxy"]
    omega_z = reference.get("omega_z")
    t = float(t)

    if t < ts[0] - 1e-9 or t > ts[-1] + 1e-9:
        return None

    px = float(np.interp(t, ts, xy[:, 0]))
    py = float(np.interp(t, ts, xy[:, 1]))
    vx = float(np.interp(t, ts, vxy[:, 0]))
    vy = float(np.interp(t, ts, vxy[:, 1]))

    state = {
        "xy": np.asarray([px, py], dtype=float),
        "vxy": np.asarray([vx, vy], dtype=float),
    }

    if omega_z is not None:
        state["omega_z"] = float(
            np.interp(
                t,
                ts,
                omega_z,
            )
        )

    return state


def reference_crossing_time(reference, x_target):
    """
    First time the reference puck crosses x_target while travelling toward -X.
    """
    ts = reference["t"]
    xs = reference["xy"][:, 0]
    x_target = float(x_target)

    for i in range(1, len(ts)):
        x0 = float(xs[i - 1])
        x1 = float(xs[i])

        if x0 >= x_target >= x1 and x1 < x0:
            denom = x1 - x0
            if abs(denom) < 1e-12:
                return float(ts[i])

            a = (x_target - x0) / denom
            return float(ts[i - 1] + a * (ts[i] - ts[i - 1]))

    return None


def count_reference_side_bounces(
    reference,
    y_limit,
    *,
    boundary_fraction=0.82,
    min_gap_steps=20,
):
    """
    Diagnostic side-wall bounce counter.

    Count a bounce when vy changes sign close to a side rail.  The debounce
    prevents several solver steps around one contact from being counted twice.
    """
    xy = reference["xy"]
    vxy = reference["vxy"]

    count = 0
    last_idx = -10**9

    for i in range(1, len(xy)):
        vy0 = float(vxy[i - 1, 1])
        vy1 = float(vxy[i, 1])

        sign_flip = (
            vy0 * vy1 < 0.0
            and abs(vy0 - vy1) > 0.02
        )
        near_rail = (
            abs(float(xy[i, 1]))
            >= boundary_fraction * float(y_limit)
        )

        if (
            sign_flip
            and near_rail
            and i - last_idx >= int(min_gap_steps)
        ):
            count += 1
            last_idx = i

    return int(count)


def find_reference_side_bounces(
    reference,
    y_limit,
    *,
    boundary_fraction=0.82,
    min_gap_steps=20,
):
    """
    Return side-wall bounce events from the exact PhysX reference.

    Each event stores the first sample after the near-rail vy sign reversal.
    This is used by Stage-5 v1.4 to add post-bounce DIRECT contact candidates.
    """
    ts = reference["t"]
    xy = reference["xy"]
    vxy = reference["vxy"]

    events = []
    last_idx = -10**9

    for i in range(1, len(ts)):
        vy0 = float(vxy[i - 1, 1])
        vy1 = float(vxy[i, 1])

        sign_flip = (
            vy0 * vy1 < 0.0
            and abs(vy0 - vy1) > 0.02
        )
        near_rail = (
            abs(float(xy[i, 1]))
            >= boundary_fraction * float(y_limit)
        )

        if (
            sign_flip
            and near_rail
            and i - last_idx >= int(min_gap_steps)
        ):
            events.append(
                {
                    "index": int(i),
                    "time": float(ts[i]),
                    "xy": np.asarray(xy[i], dtype=float).copy(),
                    "vxy": np.asarray(vxy[i], dtype=float).copy(),
                }
            )
            last_idx = i

    return events


# ============================================================
# Candidate planning / scoring
# ============================================================

def rollout_candidate(
    *,
    scene,
    timestep,
    puck,
    puck_body,
    table_length,
    rail_width,
    puck_radius,
    goal_width,
    contact_time,
    contact_x,
    prep_avg_speed,
    hit_speed,
    goal_target_y,
    desired_out_speed,
    viewer=None,
    render=False,
    seconds=2.50,
):
    """Roll forward after the strike and compute a goal-aware expert score."""
    goal_plane_x = table_length / 2.0 - rail_width
    own_goal_plane_x = -goal_plane_x
    legal_goal_y = max(0.0, goal_width / 2.0 - puck_radius)

    initial_p = np.asarray(puck.get_pose().p, dtype=float).copy()
    prev = initial_p.copy()

    max_x = float(initial_p[0])
    y_at_max_x = float(initial_p[1])
    min_goal_distance = float("inf")

    event = "TIMEOUT"
    goal_success = False
    y_cross = None
    cross_time = None
    cross_speed = None

    n_steps = max(1, int(round(seconds / timestep)))
    render_stride = max(1, int(round((1.0 / 60.0) / timestep)))

    for i in range(n_steps):
        if viewer is not None and viewer.closed:
            break

        physics_step(scene)

        p = np.asarray(puck.get_pose().p, dtype=float).copy()
        v = np.asarray(puck_body.get_linear_velocity(), dtype=float)
        speed = float(np.linalg.norm(v[:2]))

        if p[0] > max_x:
            max_x = float(p[0])
            y_at_max_x = float(p[1])

        min_goal_distance = min(
            min_goal_distance,
            float(np.hypot(goal_plane_x - p[0], p[1])),
        )

        if prev[0] < goal_plane_x <= p[0]:
            dx = float(p[0] - prev[0])
            a = (
                float(np.clip((goal_plane_x - prev[0]) / dx, 0.0, 1.0))
                if abs(dx) > 1e-12
                else 1.0
            )
            y_cross = float(prev[1] + a * (p[1] - prev[1]))
            cross_time = float((i + a) * timestep)
            cross_speed = speed

            if abs(y_cross) <= legal_goal_y:
                event = "GOAL"
                goal_success = True
            else:
                event = "END_LINE_MISS"
            break

        if prev[0] > own_goal_plane_x >= p[0]:
            event = "OWN_GOAL"
            break

        elapsed = (i + 1) * timestep
        if elapsed > 0.25 and speed < 0.03:
            event = "STOP"
            break

        prev = p

        if (
            render
            and viewer is not None
            and i % render_stride == 0
        ):
            scene.update_render()
            viewer.render()

    final_p = np.asarray(puck.get_pose().p, dtype=float)
    final_v = np.asarray(puck_body.get_linear_velocity(), dtype=float)

    progress = min(max_x, goal_plane_x) - float(initial_p[0])

    # --------------------------------------------------------
    # Stage-5 GOAL-CONDITIONED score.
    #
    # The target is explicit now:
    #   1) put the puck through the opponent goal opening
    #   2) preferably near the selected goal_target_y
    #   3) with a useful crossing speed
    #
    # Non-goal returns stay much lower than any legal GOAL.
    # --------------------------------------------------------
    if goal_success:
        target_error = abs(float(y_cross) - float(goal_target_y))
        target_quality = max(
            0.0,
            1.0 - target_error / max(legal_goal_y, 1e-6),
        )

        speed_error = abs(
            float(cross_speed if cross_speed is not None else 0.0)
            - float(desired_out_speed)
        )

        score = (
            150.0
            + 20.0 * target_quality
            + 0.40 * (cross_speed if cross_speed is not None else 0.0)
            - 0.60 * speed_error
            - 0.30 * contact_time
            - 0.15 * prep_avg_speed
            - 0.05 * max(0.0, hit_speed - 1.0)
        )

    elif event == "OWN_GOAL":
        score = -250.0

    elif event == "END_LINE_MISS":
        target_error = abs(float(y_cross) - float(goal_target_y))
        outside_error = max(
            0.0,
            abs(float(y_cross)) - legal_goal_y,
        )

        score = (
            -40.0
            - 25.0 * outside_error
            - 5.0 * target_error
        )

    else:
        # Still reward a safe aggressive return, but keep it far below GOAL.
        return_quality = (
            8.0 * max(0.0, float(final_v[0]))
            + 6.0 * max(0.0, max_x)
        )

        score = (
            return_quality
            + 4.0 * progress
            - 2.5 * min_goal_distance
            - 1.0 * abs(y_at_max_x - float(goal_target_y))
            - 0.20 * contact_time
            - 0.10 * prep_avg_speed
        )

    return {
        "score": float(score),
        "event": event,
        "goal_success": bool(goal_success),
        "y_cross": None if y_cross is None else float(y_cross),
        "cross_time": None if cross_time is None else float(cross_time),
        "cross_speed": None if cross_speed is None else float(cross_speed),
        "max_x": float(max_x),
        "y_at_max_x": float(y_at_max_x),
        "min_goal_distance": float(min_goal_distance),
        "final_x": float(final_p[0]),
        "final_y": float(final_p[1]),
        "final_vx": float(final_v[0]),
        "final_vy": float(final_v[1]),
        "goal_target_y": float(goal_target_y),
        "desired_out_speed": float(desired_out_speed),
    }


def evaluate_candidate(
    *,
    scene,
    timestep,
    assembly,
    initial_state,
    left_robot,
    left_striker,
    left_pinocchio,
    left_ee_index,
    left_active_joints,
    arm_joint_names,
    arm_qmask,
    left_limits,
    striker_ee_offset_xy,
    ee_z,
    ee_q,
    puck_body,
    table_length,
    table_width,
    rail_width,
    puck_radius,
    striker_radius,
    striker_y_limit,
    goal_width,
    serve,
    puck_reference,
    contact_time,
    goal_target_y,
    desired_out_speed,
    attack_mode="DIRECT",
    direction_correction_deg=0.0,
    speed_scale=1.0,
    command_lead_s=0.0,
    viewer=None,
    render=False,
):
    """Evaluate one explicit offensive intention: (t, goal target, desired puck speed)."""
    restore_hit_state(assembly, initial_state)
    reset_left_tool_follow_stats()

    puck = assembly["puck"]

    # Contact point + incoming velocity come from the exact PhysX reference.
    ref_state = reference_state_at_time(
        puck_reference,
        contact_time,
    )
    if ref_state is None:
        return {
            "valid": False,
            "reason": "REFERENCE_TIME_RANGE",
            "score": -1e9,
        }

    puck_contact_xy = ref_state["xy"]
    puck_in_vxy = ref_state["vxy"]
    puck_in_omega_z = float(
        ref_state.get("omega_z", 0.0)
    )

    contact_x = float(puck_contact_xy[0])
    contact_y = float(puck_contact_xy[1])

    # We only optimize direct hits in the useful left-side interception zone.
    if contact_x < -0.72 or contact_x > -0.28:
        return {
            "valid": False,
            "reason": "CONTACT_X_WINDOW",
            "score": -1e9,
        }

    y_limit = table_width / 2.0 - rail_width - puck_radius
    if abs(contact_y) > y_limit:
        return {
            "valid": False,
            "reason": "PUCK_Y_LIMIT",
            "score": -1e9,
        }

    # --------------------------------------------------------
    # Stage-5 v1.1 explicit offensive intention.
    #
    # DIRECT:
    #   aim directly toward the selected goal entry point.
    #
    # BANK_NEAR_RAIL:
    #   for a puck trapped close to a side rail, direct aiming can be
    #   geometrically impossible because the larger striker would have to
    #   move outside the table.  In that case aim the puck INTO the near rail
    #   using the mirror-goal construction. PhysX resolves the lossy,
    #   frictional rebound; the rollout score still rewards the REAL final
    #   opponent-goal crossing.
    # --------------------------------------------------------
    goal_x = table_length / 2.0 - rail_width

    attack_mode = str(attack_mode).upper()
    virtual_goal_y = float(goal_target_y)
    bank_wall_y = None

    if attack_mode == "DIRECT":
        virtual_goal_y = float(goal_target_y)

    elif attack_mode == "BANK_NEAR_RAIL":
        # Only use a bank shot when the incoming puck is genuinely close to
        # one side rail.  The nearer rail is selected from the puck sign.
        if abs(contact_y) < 0.72 * y_limit:
            return {
                "valid": False,
                "reason": "BANK_NOT_NEAR_RAIL",
                "score": -1e9,
                "attack_mode": attack_mode,
            }

        bank_wall_y = (
            +y_limit
            if contact_y >= 0.0
            else -y_limit
        )

        # Mirror the desired goal entry across the contacted side rail.
        # A straight line to this virtual point corresponds to a one-bounce
        # specular path in ideal geometry; PhysX then perturbs it according to
        # the randomized restitution/friction/spin.
        virtual_goal_y = (
            2.0 * bank_wall_y
            - float(goal_target_y)
        )

    else:
        return {
            "valid": False,
            "reason": "UNKNOWN_ATTACK_MODE",
            "score": -1e9,
            "attack_mode": attack_mode,
        }

    goal_vec = np.asarray(
        [
            goal_x - contact_x,
            virtual_goal_y - contact_y,
        ],
        dtype=float,
    )

    goal_norm = float(np.linalg.norm(goal_vec))
    if goal_norm < 1e-8:
        return {
            "valid": False,
            "reason": "BAD_GOAL_VECTOR",
            "score": -1e9,
        }

    desired_puck_dir = goal_vec / goal_norm
    desired_puck_vxy = (
        float(desired_out_speed)
        * desired_puck_dir
    )

    delta_v = desired_puck_vxy - puck_in_vxy
    delta_v_norm = float(np.linalg.norm(delta_v))

    if delta_v_norm < 1e-8:
        return {
            "valid": False,
            "reason": "BAD_IMPULSE_VECTOR",
            "score": -1e9,
        }

    nominal_direction = delta_v / delta_v_norm
    nominal_theta = float(
        np.arctan2(
            nominal_direction[1],
            nominal_direction[0],
        )
    )

    theta = (
        nominal_theta
        + np.deg2rad(float(direction_correction_deg))
    )

    direction = np.asarray(
        [
            np.cos(theta),
            np.sin(theta),
        ],
        dtype=float,
    )

    # Empirical striker-speed map for the current simulated geometry.
    # The physics intent comes from ||delta_v||; the 0.72 factor maps that
    # puck velocity change to a practical striker command speed.
    nominal_hit_speed = float(
        np.clip(
            0.72 * delta_v_norm,
            1.00,
            1.85,
        )
    )

    hit_speed_min = (
        0.70
        if attack_mode == "BANK_NEAR_RAIL"
        else 0.90
    )

    hit_speed = float(
        np.clip(
            nominal_hit_speed * float(speed_scale),
            hit_speed_min,
            1.90,
        )
    )

    contact_distance = puck_radius + striker_radius

    # Striker center at geometric contact.
    striker_contact_xy = (
        puck_contact_xy
        - direction * contact_distance
    )

    pre_distance = 0.080

    # Direct shots can use a long follow-through.  A bank shot close to the
    # rail must not command the striker through the side wall after contact.
    followthrough = (
        0.018
        if attack_mode == "BANK_NEAR_RAIL"
        else 0.100
    )

    prehit_xy = (
        striker_contact_xy
        - direction * pre_distance
    )
    hit_target_xy = (
        striker_contact_xy
        + direction * followthrough
    )

    if (
        abs(prehit_xy[1]) > striker_y_limit
        or abs(striker_contact_xy[1]) > striker_y_limit
        or abs(hit_target_xy[1]) > striker_y_limit
    ):
        return {
            "valid": False,
            "reason": "STRIKER_Y_LIMIT",
            "score": -1e9,
            "attack_mode": attack_mode,
            "contact_x": float(contact_x),
            "contact_y": float(contact_y),
            "striker_contact_y": float(striker_contact_xy[1]),
            "prehit_y": float(prehit_xy[1]),
            "hit_target_y": float(hit_target_xy[1]),
        }

    # Conservative left-robot table/workspace window for Stage 2.
    x_min = -table_length / 2.0 + rail_width + striker_radius
    if (
        prehit_xy[0] < x_min
        or hit_target_xy[0] > +0.05
    ):
        return {
            "valid": False,
            "reason": "STRIKER_X_LIMIT",
            "score": -1e9,
        }

    strike_lead_time = pre_distance / max(float(hit_speed), 1e-6)

    # The commanded Cartesian path is followed by a dynamic UR5 articulation,
    # not teleported exactly.  Positive command_lead_s starts the strike
    # command earlier to compensate tracking lag.
    prep_time = float(
        contact_time
        - strike_lead_time
        - float(command_lead_s)
    )

    if prep_time < 0.10:
        return {
            "valid": False,
            "reason": "TOO_LATE",
            "score": -1e9,
        }

    start_xy = np.asarray(
        left_striker.get_pose().p[:2],
        dtype=float,
    )
    prep_distance_xy = float(
        np.linalg.norm(prehit_xy - start_xy)
    )
    required_avg_prep_speed = prep_distance_xy / prep_time

    # Cheap timing gate before any long PhysX rollout.
    if required_avg_prep_speed > 1.15:
        return {
            "valid": False,
            "reason": "PREP_TOO_FAST",
            "score": -1e9,
            "required_avg_prep_speed": float(required_avg_prep_speed),
        }

    # Cheap endpoint IK gates.
    q0 = np.asarray(left_robot.get_qpos(), dtype=float)

    pre_ok, q_pre = solve_striker_xy_ik(
        left_robot=left_robot,
        left_pinocchio=left_pinocchio,
        left_ee_index=left_ee_index,
        arm_qmask=arm_qmask,
        left_limits=left_limits,
        striker_ee_offset_xy=striker_ee_offset_xy,
        ee_z=ee_z,
        ee_q=ee_q,
        target_xy=prehit_xy,
        initial_qpos=q0,
    )
    if not pre_ok:
        return {
            "valid": False,
            "reason": "PREHIT_IK",
            "score": -1e9,
        }

    hit_ok, _ = solve_striker_xy_ik(
        left_robot=left_robot,
        left_pinocchio=left_pinocchio,
        left_ee_index=left_ee_index,
        arm_qmask=arm_qmask,
        left_limits=left_limits,
        striker_ee_offset_xy=striker_ee_offset_xy,
        ee_z=ee_z,
        ee_q=ee_q,
        target_xy=hit_target_xy,
        initial_qpos=q_pre,
    )
    if not hit_ok:
        return {
            "valid": False,
            "reason": "HIT_IK",
            "score": -1e9,
        }

    contact_gate = contact_distance + 0.005

    # --------------------------------------------------------
    # Phase A: move to the run-up point while puck is inbound.
    # Recorder calls are no-ops during hidden candidate search because the
    # recorder is configured only AFTER planning. They become active for the
    # single selected expert execution.
    # --------------------------------------------------------
    collector.episode_set_phase("DIRECT_PREP")

    prep = timed_cartesian_segment(
        scene=scene,
        timestep=timestep,
        left_robot=left_robot,
        left_striker=left_striker,
        left_pinocchio=left_pinocchio,
        left_ee_index=left_ee_index,
        left_active_joints=left_active_joints,
        arm_joint_names=arm_joint_names,
        arm_qmask=arm_qmask,
        left_limits=left_limits,
        striker_ee_offset_xy=striker_ee_offset_xy,
        ee_z=ee_z,
        ee_q=ee_q,
        target_xy=prehit_xy,
        duration=prep_time,
        puck=puck,
        viewer=viewer,
        render=render,
        contact_gate=contact_gate,
    )

    if not prep["ok"]:
        return {
            "valid": False,
            "reason": "PREP_PATH_IK",
            "score": -1e9,
        }

    # We do not want to touch the puck during preparation.
    if prep["contact"]:
        return {
            "valid": False,
            "reason": "EARLY_CONTACT",
            "score": -1e9,
        }

    # --------------------------------------------------------
    # Phase B: timed strike through moving puck.
    # --------------------------------------------------------
    collector.episode_set_phase("DIRECT_HIT")

    strike_distance = pre_distance + followthrough
    strike_duration = strike_distance / max(float(hit_speed), 1e-6)

    strike = timed_cartesian_segment(
        scene=scene,
        timestep=timestep,
        left_robot=left_robot,
        left_striker=left_striker,
        left_pinocchio=left_pinocchio,
        left_ee_index=left_ee_index,
        left_active_joints=left_active_joints,
        arm_joint_names=arm_joint_names,
        arm_qmask=arm_qmask,
        left_limits=left_limits,
        striker_ee_offset_xy=striker_ee_offset_xy,
        ee_z=ee_z,
        ee_q=ee_q,
        target_xy=hit_target_xy,
        duration=strike_duration,
        puck=puck,
        viewer=viewer,
        render=render,
        contact_gate=contact_gate,
    )

    if not strike["ok"]:
        return {
            "valid": False,
            "reason": "STRIKE_PATH_IK",
            "score": -1e9,
        }

    if not strike["contact"]:
        return {
            "valid": False,
            "reason": "NO_CONTACT",
            "score": -1e9,
            "contact_time": float(contact_time),
            "contact_x": contact_x,
            "contact_y": contact_y,
            "goal_target_y": float(goal_target_y),
            "desired_out_speed": float(desired_out_speed),
            "attack_mode": attack_mode,
            "bank_wall_y": (
                None
                if bank_wall_y is None
                else float(bank_wall_y)
            ),
            "direction_correction_deg": float(direction_correction_deg),
            "speed_scale": float(speed_scale),
            "nominal_hit_speed": float(nominal_hit_speed),
            "hit_speed": float(hit_speed),
            "command_lead_s": float(command_lead_s),
            "min_contact_distance": float(strike["min_distance"]),
        }

    # --------------------------------------------------------
    # Phase C: after-contact forward simulation and score.
    # --------------------------------------------------------
    collector.episode_set_phase("FLIGHT")

    rollout = rollout_candidate(
        scene=scene,
        timestep=timestep,
        puck=puck,
        puck_body=puck_body,
        table_length=table_length,
        rail_width=rail_width,
        puck_radius=puck_radius,
        goal_width=goal_width,
        contact_time=float(contact_time),
        contact_x=contact_x,
        prep_avg_speed=float(required_avg_prep_speed),
        hit_speed=float(hit_speed),
        goal_target_y=float(goal_target_y),
        desired_out_speed=float(desired_out_speed),
        viewer=viewer,
        render=render,
    )

    result = {
        "valid": True,
        "reason": "OK",
        "contact_time": float(contact_time),
        "contact_x": contact_x,
        "contact_y": contact_y,
        "goal_target_y": float(goal_target_y),
        "desired_out_speed": float(desired_out_speed),
        "attack_mode": attack_mode,
        "bank_wall_y": (
            None
            if bank_wall_y is None
            else float(bank_wall_y)
        ),
        "virtual_goal_y": float(virtual_goal_y),
        "direction_correction_deg": float(direction_correction_deg),
        "speed_scale": float(speed_scale),
        "nominal_theta_deg": float(np.rad2deg(nominal_theta)),
        "nominal_hit_speed": float(nominal_hit_speed),
        "hit_speed": float(hit_speed),
        "incoming_vx": float(puck_in_vxy[0]),
        "incoming_vy": float(puck_in_vxy[1]),
        "incoming_omega_z": float(puck_in_omega_z),
        "desired_puck_vx": float(desired_puck_vxy[0]),
        "desired_puck_vy": float(desired_puck_vxy[1]),
        "command_lead_s": float(command_lead_s),
        "prep_time": prep_time,
        "required_avg_prep_speed": float(required_avg_prep_speed),
        "min_contact_distance": float(strike["min_distance"]),
        "tool_follow_max_error": float(get_left_tool_follow_max_error()),
    }
    result.update(rollout)
    return result


def unique_values(values, ndigits=6):
    seen = set()
    out = []
    for v in values:
        k = round(float(v), ndigits)
        if k not in seen:
            seen.add(k)
            out.append(float(v))
    return out


def run_goal_search(
    *,
    stage_name,
    t_values,
    goal_y_values,
    desired_speed_values,
    evaluate_fn,
):
    results = []
    total = (
        len(t_values)
        * len(goal_y_values)
        * len(desired_speed_values)
    )
    idx = 0

    print("")
    print(
        f"=== {stage_name} GOAL-CONDITIONED SEARCH: "
        f"{total} nominal candidates ==="
    )

    for t_contact in t_values:
        for goal_target_y in goal_y_values:
            for desired_out_speed in desired_speed_values:
                idx += 1

                r = evaluate_fn(
                    contact_time=float(t_contact),
                    goal_target_y=float(goal_target_y),
                    desired_out_speed=float(desired_out_speed),
                )
                r["search_stage"] = stage_name
                results.append(r)

                if r.get("valid", False):
                    print(
                        f"[{stage_name} {idx:03d}/{total:03d}] "
                        f"t={r['contact_time']:.3f}s "
                        f"p=({r['contact_x']:+.3f},{r['contact_y']:+.3f}) "
                        f"goal_y={r['goal_target_y']:+.3f} "
                        f"vout*={r['desired_out_speed']:.2f} "
                        f"mode={r.get('attack_mode', 'DIRECT')} "
                        f"hit={r['hit_speed']:.2f} "
                        f"score={r['score']:+.3f} "
                        f"event={r['event']} "
                        f"lead={r.get('command_lead_s', 0.0)*1000:.0f}ms"
                    )
                else:
                    dmin = r.get("min_contact_distance")
                    dtext = (
                        ""
                        if dmin is None
                        else f" dmin={float(dmin)*1000:.1f}mm"
                    )

                    mode = r.get(
                        "attack_mode",
                        "?"
                    )
                    lead_ms = (
                        1000.0
                        * float(
                            r.get(
                                "command_lead_s",
                                0.0,
                            )
                        )
                    )
                    hit_text = (
                        ""
                        if r.get("hit_speed") is None
                        else f" hit={float(r['hit_speed']):.2f}"
                    )

                    print(
                        f"[{stage_name} {idx:03d}/{total:03d}] "
                        f"t={t_contact:.3f}s "
                        f"goal_y={goal_target_y:+.3f} "
                        f"vout*={desired_out_speed:.2f} "
                        f"mode={mode} "
                        f"INVALID {r.get('reason', 'UNKNOWN')}"
                        f" lead={lead_ms:.0f}ms"
                        f"{hit_text}"
                        f"{dtext}"
                    )

    valid = [
        r
        for r in results
        if r.get("valid", False)
    ]

    best = (
        max(valid, key=lambda x: x["score"])
        if valid
        else None
    )

    return results, best


# ============================================================
# Main
# ============================================================

def candidate_identity(r):
    """
    Stable key for deduplicating the same strategy appearing in coarse/fine.
    command_lead_s is included because it affects execution timing.
    """
    return (
        str(r.get("attack_mode", "DIRECT")),
        round(float(r["contact_time"]), 6),
        round(float(r["goal_target_y"]), 5),
        round(float(r["desired_out_speed"]), 4),
        round(float(r.get("direction_correction_deg", 0.0)), 3),
        round(float(r.get("speed_scale", 1.0)), 3),
        round(float(r.get("command_lead_s", 0.0)), 4),
    )


def ranked_unique_candidates(*result_sets):
    """Merge valid candidates, deduplicate them, and rank by search score."""
    best_by_key = {}

    for results in result_sets:
        for r in results:
            if not r.get("valid", False):
                continue

            key = candidate_identity(r)
            previous = best_by_key.get(key)

            if previous is None or r["score"] > previous["score"]:
                best_by_key[key] = r

    return sorted(
        best_by_key.values(),
        key=lambda r: float(r["score"]),
        reverse=True,
    )


def verify_candidate(
    *,
    candidate,
    verify_repeats,
    evaluate_once_fn,
):
    """
    Replay one candidate repeatedly from the exact same episode snapshot.

    A robust GOAL candidate must produce GOAL on every verification replay.
    We also record a conservative min score for fallback ranking.
    """
    trials = []

    for repeat_i in range(int(verify_repeats)):
        r = evaluate_once_fn(
            contact_time=candidate["contact_time"],
            goal_target_y=candidate["goal_target_y"],
            desired_out_speed=candidate["desired_out_speed"],
            attack_mode=candidate.get("attack_mode", "DIRECT"),
            direction_correction_deg=candidate.get("direction_correction_deg", 0.0),
            speed_scale=candidate.get("speed_scale", 1.0),
            command_lead_s=candidate.get("command_lead_s", 0.0),
            live_viewer=None,
            live_render=False,
        )

        trials.append(
            {
                "repeat": int(repeat_i + 1),
                "valid": bool(r.get("valid", False)),
                "event": r.get("event", r.get("reason", "UNKNOWN")),
                "score": float(r.get("score", -1e9)),
                "final_vx": (
                    None
                    if r.get("final_vx") is None
                    else float(r["final_vx"])
                ),
                "max_x": (
                    None
                    if r.get("max_x") is None
                    else float(r["max_x"])
                ),
            }
        )

    valid_all = all(t["valid"] for t in trials)
    events = [t["event"] for t in trials]
    scores = [t["score"] for t in trials]

    goal_all = (
        valid_all
        and all(event == "GOAL" for event in events)
    )

    own_goal_any = any(event == "OWN_GOAL" for event in events)

    # Conservative fallback criterion:
    # every replay is valid, no own-goal, and at least returns the puck
    # meaningfully toward / into the opponent half.
    useful_return_all = True
    if valid_all and not own_goal_any:
        for t in trials:
            vx = t["final_vx"]
            mx = t["max_x"]
            if not (
                (vx is not None and vx > 0.05)
                or (mx is not None and mx > 0.20)
                or t["event"] in ("GOAL", "END_LINE_MISS")
            ):
                useful_return_all = False
                break
    else:
        useful_return_all = False

    return {
        "candidate": {
            "contact_time": float(candidate["contact_time"]),
            "contact_x": float(candidate["contact_x"]),
            "contact_y": float(candidate["contact_y"]),
            "goal_target_y": float(candidate["goal_target_y"]),
            "desired_out_speed": float(candidate["desired_out_speed"]),
            "attack_mode": candidate.get("attack_mode", "DIRECT"),
            "direction_correction_deg": float(
                candidate.get("direction_correction_deg", 0.0)
            ),
            "speed_scale": float(candidate.get("speed_scale", 1.0)),
            "hit_speed": float(candidate["hit_speed"]),
            "command_lead_s": float(candidate.get("command_lead_s", 0.0)),
            "search_event": candidate.get("event"),
            "search_score": float(candidate["score"]),
        },
        "trials": trials,
        "goal_all": bool(goal_all),
        "valid_all": bool(valid_all),
        "own_goal_any": bool(own_goal_any),
        "useful_return_all": bool(useful_return_all),
        "min_verify_score": float(min(scores) if scores else -1e9),
        "mean_verify_score": float(np.mean(scores) if scores else -1e9),
    }


def choose_verified_candidates(
    *,
    ranked_candidates,
    verify_top_k,
    verify_repeats,
    evaluate_once_fn,
):
    """
    SEARCH -> VERIFY -> ACCEPT shortlist.

    Priority:
      1) candidates whose search predicted GOAL
      2) other high-score candidates

    Acceptance queue:
      robust GOALs first, then robust useful-return fallbacks.
    """
    # Goal candidates first so we do not waste verification budget on
    # low-value STOP/TIMEOUT candidates while unverified GOALs remain.
    verification_order = sorted(
        ranked_candidates,
        key=lambda r: (
            0 if r.get("event") == "GOAL" else 1,
            -float(r["score"]),
        ),
    )

    verification_order = verification_order[: int(verify_top_k)]

    reports = []
    robust_goals = []
    robust_fallbacks = []

    print("")
    print("=== VERIFY STAGE ===")
    print(
        f"checking up to {len(verification_order)} candidates, "
        f"{verify_repeats} replay(s) each"
    )

    for idx, candidate in enumerate(verification_order, start=1):
        report = verify_candidate(
            candidate=candidate,
            verify_repeats=verify_repeats,
            evaluate_once_fn=evaluate_once_fn,
        )
        reports.append(report)

        event_seq = ",".join(
            str(t["event"])
            for t in report["trials"]
        )

        print(
            f"[VERIFY {idx:02d}/{len(verification_order):02d}] "
            f"search={candidate.get('event')} "
            f"score={candidate['score']:+.3f} "
            f"t={candidate['contact_time']:.4f}s "
            f"p=({candidate['contact_x']:+.3f},"
            f"{candidate['contact_y']:+.3f}) "
            f"goal_y={candidate['goal_target_y']:+.3f} "
            f"vout*={candidate['desired_out_speed']:.2f} "
            f"mode={candidate.get('attack_mode', 'DIRECT')} "
            f"hit={candidate['hit_speed']:.2f} "
            f"lead={candidate.get('command_lead_s', 0.0)*1000:.0f}ms "
            f"replay=[{event_seq}]"
        )

        if report["goal_all"]:
            robust_goals.append(
                (candidate, report)
            )
        elif report["useful_return_all"]:
            robust_fallbacks.append(
                (candidate, report)
            )

    # For robust goals, retain the original planner objective but use
    # verification score as a tie-breaker.
    robust_goals.sort(
        key=lambda pair: (
            float(pair[0]["score"]),
            float(pair[1]["min_verify_score"]),
        ),
        reverse=True,
    )

    # Fallbacks use maximin verification score for robustness.
    robust_fallbacks.sort(
        key=lambda pair: (
            float(pair[1]["min_verify_score"]),
            float(pair[0]["score"]),
        ),
        reverse=True,
    )

    return reports, robust_goals, robust_fallbacks


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "env_cfg" / "air_hockey_assets.yml",
    )
    parser.add_argument(
        "--coarse-only",
        action="store_true",
        help="skip fine search and replay coarse BEST",
    )
    parser.add_argument(
        "--render-search",
        action="store_true",
        help="render candidate searches too (very slow; normally leave off)",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=3.0,
        help="hold final Viewer frame for N seconds",
    )
    parser.add_argument(
        "--verify-repeats",
        type=int,
        default=3,
        help="off-screen replay count required for robust verification (default 3)",
    )
    parser.add_argument(
        "--verify-top-k",
        type=int,
        default=12,
        help="maximum ranked candidates to verify off-screen (used only with --verify)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="enable Stage-3-style repeated verification; OFF by default for fast collection",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="record ONLY the selected BEST execution using the existing RGB/state recorder",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="do not open Viewer for the selected execution",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="optional deterministic planner summary JSON path",
    )
    args = parser.parse_args()

    if args.verify and args.verify_repeats < 1:
        parser.error("--verify-repeats must be >= 1 when --verify is enabled")
    if args.verify and args.verify_top_k < 1:
        parser.error("--verify-top-k must be >= 1 when --verify is enabled")

    # Separate deterministic random streams so serve rejection/resampling does
    # not change the physical material sampled for this seed.
    seed_sequence = np.random.SeedSequence(args.seed)
    physics_ss, serve_ss = seed_sequence.spawn(2)
    physics_rng = np.random.default_rng(physics_ss)
    serve_rng = np.random.default_rng(serve_ss)

    config = load_asset_config(args.config)

    # Stage 5 physical domain randomization happens BEFORE materials/entities
    # are created.  One sampled table is then fixed for the whole episode.
    stage5_physics = sample_stage5_physics(
        physics_rng,
        config,
    )

    from sapien.utils.viewer import Viewer

    configure_air_hockey_physics(config)

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    # IMPORTANT (SAPIEN 3 compatibility):
    # Engine.create_scene(scene_config) internally applies the supplied PhysX
    # scene config again. Passing a fresh default SceneConfig here would reset
    # bounce_threshold back to its default value and suppress normal-speed
    # rail rebounds. Therefore the threshold must also be written into the
    # exact SceneConfig used to create this scene.
    scene_config = sapien.SceneConfig()
    scene_config.bounce_threshold = float(
        config["physics"]["bounce_threshold_m_s"]
    )

    scene = engine.create_scene(scene_config)

    # Runtime diagnostic: this global value should now agree with the YAML and
    # with the SceneConfig that created the PhysX system.
    try:
        actual_bounce_threshold = float(
            sapien.physx.get_scene_config().bounce_threshold
        )
    except Exception:
        actual_bounce_threshold = float("nan")

    timestep = float(config["physics"]["timestep_s"])
    scene.set_timestep(timestep)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light(
        [0, 0.5, -1],
        [0.7, 0.7, 0.7],
        shadow=True,
    )

    # Same validation-time stronger grasp values used by current baseline.
    config["grasp"]["finger_force_limit_N"] = 150.0
    config["grasp"]["finger_drive_stiffness_N_m"] = 2200.0
    config["grasp"]["finger_drive_damping_Ns_m"] = 180.0
    config["grasp"]["finger_static_friction"] = 4.0
    config["grasp"]["finger_dynamic_friction"] = 3.0

    assembly = populate_robot_preview(scene, config)

    puck = assembly["puck"]
    puck_body = puck.find_component_by_type(
        sapien.physx.PhysxRigidDynamicComponent
    )
    if puck_body is None:
        raise RuntimeError("puck has no PhysxRigidDynamicComponent")

    left_robot = assembly["left_robot"]
    left_striker = assembly["left_striker"]

    robot_cfg = config["robot_preview"]
    arm_joint_names = set(robot_cfg["arm_joint_names"])

    left_links = left_robot.get_links()
    left_ee_index = next(
        i
        for i, link in enumerate(left_links)
        if link.get_name() == robot_cfg["ee_link_name"]
    )
    left_ee = left_links[left_ee_index].entity

    left_pinocchio = left_robot.create_pinocchio_model()
    left_active_joints = left_robot.get_active_joints()
    left_limits = np.asarray(left_robot.get_qlimits(), dtype=float)

    arm_qmask = np.asarray(
        [
            1 if joint.name in arm_joint_names else 0
            for joint in left_active_joints
        ],
        dtype=np.int32,
    )

    configure_left_kinematic_tool(left_striker, left_ee)

    table_length = float(config["table"]["length_m"])
    table_width = float(config["table"]["width_m"])
    rail_width = float(config["table"]["rail_width_m"])
    goal_width = float(config["table"]["goal_width_m"])
    puck_radius = float(config["puck"]["diameter_m"]) / 2.0
    striker_radius = float(config["striker"]["diameter_m"]) / 2.0

    y_limit = (
        table_width / 2.0
        - rail_width
        - puck_radius
    )
    striker_y_limit = (
        table_width / 2.0
        - rail_width
        - striker_radius
    )

    for _ in range(config["grasp"]["settle_steps"]):
        physics_step(scene)

    striker_initial = np.asarray(
        left_striker.get_pose().p,
        dtype=float,
    )
    ee_initial = left_ee.pose
    ee_initial_p = np.asarray(
        ee_initial.p,
        dtype=float,
    )

    striker_ee_offset_xy = (
        striker_initial[:2]
        - ee_initial_p[:2]
    )
    ee_z = float(ee_initial_p[2])
    ee_q = np.asarray(ee_initial.q, dtype=float).copy()

    # --------------------------------------------------------
    # Stage-5 large-angle randomized serve.
    #
    # Side-wall bounces are allowed. A proposal is accepted only if the exact
    # PhysX free-rollout under THIS episode's randomized materials reaches the
    # planner interception region.  We keep 0-2 side-wall bounces in this
    # first Stage-5 curriculum.
    # --------------------------------------------------------
    serve = None
    initial_state = None
    puck_reference = None
    reference_side_bounces = None
    accepted_serve_attempt = None

    max_serve_attempts = 120
    reference_horizon_s = 6.0

    puck_pose_home = puck.get_pose()
    puck_z = float(puck_pose_home.p[2])
    puck_q = np.asarray(
        puck_pose_home.q,
        dtype=float,
    ).copy()

    for serve_attempt in range(1, max_serve_attempts + 1):
        proposal = sample_stage5_serve(
            serve_rng,
            serve_x=+0.70,
            y_limit=y_limit,
        )

        puck.set_pose(
            sapien.Pose(
                [
                    proposal["serve_x"],
                    proposal["serve_y"],
                    puck_z,
                ],
                puck_q,
            )
        )

        puck_body.set_linear_velocity(
            np.asarray(
                [
                    proposal["vx"],
                    proposal["vy"],
                    0.0,
                ],
                dtype=float,
            )
        )

        puck_body.set_angular_velocity(
            np.asarray(
                [
                    0.0,
                    0.0,
                    proposal["omega_z_rad_s"],
                ],
                dtype=float,
            )
        )

        trial_state = snapshot_hit_state(
            assembly
        )

        trial_reference = build_physx_puck_reference(
            scene=scene,
            timestep=timestep,
            assembly=assembly,
            initial_state=trial_state,
            puck=puck,
            puck_body=puck_body,
            horizon=reference_horizon_s,
            # Stage-5 v1.4 keeps a deeper left-side reference so a puck that
            # bounces near x=-0.65..-0.72 still has enough post-bounce
            # trajectory for a later DIRECT interception.
            stop_x=-0.92,
        )

        trial_bounces = count_reference_side_bounces(
            trial_reference,
            y_limit,
        )

        # At least the front planner interception line must be reached.
        reaches_planner = (
            reference_crossing_time(
                trial_reference,
                -0.43,
            )
            is not None
        )

        # Stage-5 v1 curriculum permits direct, single-wall, and double-wall
        # inbound trajectories. More complicated multi-bounce states can be
        # enabled later.
        bounce_ok = (
            0 <= trial_bounces <= 2
        )

        if reaches_planner and bounce_ok:
            serve = proposal
            initial_state = trial_state
            puck_reference = trial_reference
            reference_side_bounces = trial_bounces
            accepted_serve_attempt = serve_attempt
            break

    if serve is None:
        raise RuntimeError(
            "Stage-5 serve sampler failed to find a PhysX-valid "
            "0-2-bounce inbound trajectory after "
            f"{max_serve_attempts} attempts"
        )

    # Ensure candidate search starts from the accepted serve, not from the end
    # of the free-rollout used for validity checking.
    restore_hit_state(
        assembly,
        initial_state,
    )

    print("")
    print("======================================================")
    print("STAGE 5 / LARGE-ANGLE + SPIN + WALL-BOUNCE OFFENSIVE PLANNER")
    print("======================================================")
    print(f"seed          : {args.seed}")
    print(f"serve line x  : {serve['serve_x']:+.3f} m")
    print(f"serve y       : {serve['serve_y']:+.3f} m")
    print(f"serve speed   : {serve['speed']:.3f} m/s")
    print(f"serve heading : {serve['angle_deg']:+.2f} deg from -X")
    print(
        f"serve velocity: "
        f"({serve['vx']:+.4f},{serve['vy']:+.4f}) m/s"
    )
    print(
        f"initial spin  : "
        f"{serve['omega_z_rad_s']:+.3f} rad/s"
    )
    print(
        f"side bounces  : "
        f"{reference_side_bounces}"
    )
    print(
        f"serve attempts: "
        f"{accepted_serve_attempt}"
    )
    print(
        "physics       : "
        f"table_mu_d={stage5_physics['table_dynamic_friction']:.4f}, "
        f"puck_mu_d={stage5_physics['puck_dynamic_friction']:.4f}, "
        f"rail_mu_d={stage5_physics['rail_dynamic_friction']:.4f}, "
        f"rail_e={stage5_physics['rail_restitution']:.4f}"
    )
    print(
        f"bounce thresh : "
        f"{actual_bounce_threshold:.4f} m/s"
    )
    print("candidate randomness: NONE (same episode physics/serve for all candidates)")
    print("offensive intent: EXPLICIT goal target + desired outgoing puck speed")
    print("======================================================")

    # --------------------------------------------------------
    # Stage-5 v1.4 bounce-aware contact-time generation.
    #
    # Keep the original fixed-X interception points, but ALSO detect exact
    # PhysX side-wall rebounds and add several post-bounce times. This fixes
    # the failure mode where all fixed-X candidates occur while the puck is
    # still squeezing toward a rail.
    # --------------------------------------------------------
    front_x = -0.43
    back_x = -0.67

    coarse_x = [-0.43, -0.49, -0.55, -0.61, -0.67]
    coarse_t = []

    print("")
    print("=== PHYSX REFERENCE CONTACT TIMES ===")
    for x_candidate in coarse_x:
        tc = reference_crossing_time(
            puck_reference,
            x_candidate,
        )
        if tc is not None:
            xy = reference_xy_at_time(
                puck_reference,
                tc,
            )
            coarse_t.append(tc)
            print(
                f"x={x_candidate:+.3f} -> "
                f"t={tc:.4f}s, "
                f"y={xy[1]:+.4f}"
            )
        else:
            print(
                f"x={x_candidate:+.3f} -> NO CROSSING"
            )

    bounce_events = find_reference_side_bounces(
        puck_reference,
        y_limit,
    )

    post_bounce_offsets_s = [
        0.080,
        0.160,
        0.240,
        0.320,
    ]

    print("")
    print("=== BOUNCE-AWARE EXTRA CONTACT TIMES ===")

    if bounce_events:
        ref_t_end = float(puck_reference["t"][-1])

        for bi, event in enumerate(
            bounce_events,
            start=1,
        ):
            print(
                f"bounce {bi}: "
                f"t={event['time']:.4f}s "
                f"p=({event['xy'][0]:+.4f},{event['xy'][1]:+.4f}) "
                f"v=({event['vxy'][0]:+.4f},{event['vxy'][1]:+.4f})"
            )

            for dt_after in post_bounce_offsets_s:
                tc = float(
                    event["time"]
                    + dt_after
                )

                if tc > ref_t_end - 1e-9:
                    continue

                state = reference_state_at_time(
                    puck_reference,
                    tc,
                )
                if state is None:
                    continue

                xy = state["xy"]
                vxy = state["vxy"]

                # Only add post-bounce states that are still on the leftward
                # incoming leg and not already beyond a practical deep-left
                # interception region.
                if (
                    float(vxy[0]) < -0.02
                    and float(xy[0]) >= -0.90
                ):
                    coarse_t.append(tc)
                    print(
                        f"  +{dt_after*1000:3.0f}ms -> "
                        f"t={tc:.4f}s "
                        f"p=({xy[0]:+.4f},{xy[1]:+.4f}) "
                        f"v=({vxy[0]:+.4f},{vxy[1]:+.4f})"
                    )
    else:
        print("no side-wall bounce detected")

    # Deterministic de-duplication after combining fixed-X and post-bounce
    # candidates.
    coarse_t = sorted(
        {
            round(float(t), 6)
            for t in coarse_t
        }
    )

    if not coarse_t:
        raise RuntimeError(
            "PhysX reference produced no usable contact times"
        )

    t_front = min(coarse_t)
    t_back = max(coarse_t)

    # --------------------------------------------------------
    # Explicit offensive intention.
    #
    # Search goal entry points inside the legal opening and desired
    # outgoing puck speed.  The strike direction is derived from:
    #
    #   desired_out_velocity - incoming_puck_velocity
    #
    # instead of blindly sweeping absolute angle offsets.
    # --------------------------------------------------------
    legal_goal_y = max(
        0.0,
        goal_width / 2.0 - puck_radius,
    )

    coarse_goal_y = [
        -0.60 * legal_goal_y,
        0.0,
        +0.60 * legal_goal_y,
    ]

    coarse_desired_out_speeds = [
        1.00,
        1.35,
        1.70,
    ]

    print("")
    print("=== EXPLICIT OFFENSIVE TARGETS ===")
    print(
        "goal_y candidates: "
        + ", ".join(
            f"{v:+.3f}"
            for v in coarse_goal_y
        )
    )
    print(
        "desired puck out-speed candidates: "
        + ", ".join(
            f"{v:.2f}"
            for v in coarse_desired_out_speeds
        )
        + " m/s"
    )

    search_viewer = None
    if args.render_search:
        search_viewer = Viewer(renderer)
        search_viewer.set_scene(scene)
        search_viewer.set_camera_xyz(
            x=2.1,
            y=-1.7,
            z=3.0,
        )
        search_viewer.set_camera_rpy(
            r=0,
            p=-0.75,
            y=-2.46,
        )

    def evaluate_once(
        *,
        contact_time,
        goal_target_y,
        desired_out_speed,
        attack_mode,
        direction_correction_deg,
        speed_scale,
        command_lead_s,
        live_viewer=None,
        live_render=False,
    ):
        return evaluate_candidate(
            scene=scene,
            timestep=timestep,
            assembly=assembly,
            initial_state=initial_state,
            left_robot=left_robot,
            left_striker=left_striker,
            left_pinocchio=left_pinocchio,
            left_ee_index=left_ee_index,
            left_active_joints=left_active_joints,
            arm_joint_names=arm_joint_names,
            arm_qmask=arm_qmask,
            left_limits=left_limits,
            striker_ee_offset_xy=striker_ee_offset_xy,
            ee_z=ee_z,
            ee_q=ee_q,
            puck_body=puck_body,
            table_length=table_length,
            table_width=table_width,
            rail_width=rail_width,
            puck_radius=puck_radius,
            striker_radius=striker_radius,
            striker_y_limit=striker_y_limit,
            goal_width=goal_width,
            serve=serve,
            puck_reference=puck_reference,
            contact_time=contact_time,
            goal_target_y=goal_target_y,
            desired_out_speed=desired_out_speed,
            attack_mode=attack_mode,
            direction_correction_deg=direction_correction_deg,
            speed_scale=speed_scale,
            command_lead_s=command_lead_s,
            viewer=live_viewer,
            render=live_render,
        )

    def evaluate_fn(
        *,
        contact_time,
        goal_target_y,
        desired_out_speed,
        live_viewer=None,
        live_render=False,
    ):
        # ----------------------------------------------------
        # Stage-5 v1.4 bounce-aware strategy scheduling.
        #
        # Near rail:
        #   direct inward shots are geometrically impossible for the larger
        #   striker, so evaluate ONLY bank shots.
        #
        # Central region:
        #   use the original direct attack.
        #
        # Bank shots get:
        #   - slower striker sweep trials
        #   - much wider command-lead search
        #
        # This explicitly searches the robot's real dynamic tracking delay
        # rather than assuming the Stage-4 0/20/40-ms range is enough.
        # ----------------------------------------------------
        ref_state = reference_state_at_time(
            puck_reference,
            contact_time,
        )

        contact_y_for_mode = (
            0.0
            if ref_state is None
            else float(ref_state["xy"][1])
        )

        near_rail = (
            abs(contact_y_for_mode)
            >= 0.72 * y_limit
        )

        contact_vy = (
            0.0
            if ref_state is None
            else float(ref_state["vxy"][1])
        )

        # Stage-5 v1.5 uses a finite normal-speed threshold.  A tiny sign
        # reversal after a grazing rail impact (e.g. vy=-0.004 m/s) is NOT
        # treated as a meaningful inward rebound.
        rail_normal_speed_eps = 0.030

        signed_normal_speed = (
            np.sign(contact_y_for_mode)
            * contact_vy
        )

        # Positive: motion toward the nearest rail.
        moving_toward_rail = (
            near_rail
            and signed_normal_speed
            > rail_normal_speed_eps
        )

        # Negative with sufficient magnitude: real inward rebound.
        moving_away_from_rail = (
            near_rail
            and signed_normal_speed
            < -rail_normal_speed_eps
        )

        # Near-zero normal velocity while still close to the rail means a
        # grazing/edge-glide state. Waiting will not create useful clearance
        # on the time scale of the episode, so use the bank-shot fallback.
        edge_glide = (
            near_rail
            and abs(signed_normal_speed)
            <= rail_normal_speed_eps
        )

        if moving_toward_rail:
            # Do not force a bank shot while the puck is naturally about to
            # hit the rail. Bounce-aware extra contact times are generated
            # later in the reference instead.
            return {
                "valid": False,
                "reason": "WAIT_FOR_BOUNCE",
                "score": -1e9,
                "attack_mode": "WAIT_FOR_BOUNCE",
                "contact_time": float(contact_time),
                "contact_x": (
                    None
                    if ref_state is None
                    else float(ref_state["xy"][0])
                ),
                "contact_y": float(contact_y_for_mode),
                "incoming_vy": float(contact_vy),
                "rail_state": "TOWARD_RAIL",
                "rail_normal_speed": float(signed_normal_speed),
            }

        if moving_away_from_rail:
            # Only a meaningful inward normal velocity qualifies for DIRECT.
            attack_modes = ["DIRECT"]

            correction_trials = [
                (0.0, 1.00),
                (-4.0, 1.00),
                (+4.0, 1.00),
            ]

            lead_trials = [
                0.000,
                0.020,
                0.040,
            ]

        elif edge_glide:
            # Grazing impact / rail glide.  The puck remains too close to the
            # side rail for a direct inward strike because the striker would
            # need to stand outside the table.  Try an inside-origin bank shot.
            attack_modes = ["BANK_NEAR_RAIL"]

            correction_trials = [
                (0.0, 0.75),
                (-5.0, 0.75),
                (+5.0, 0.75),
                (0.0, 0.90),
                (-5.0, 0.90),
                (+5.0, 0.90),
                (0.0, 1.00),
            ]

            lead_trials = [
                0.000,
                0.040,
                0.080,
                0.120,
                0.160,
                0.200,
            ]

        elif near_rail:
            attack_modes = ["BANK_NEAR_RAIL"]

            correction_trials = [
                (0.0, 0.75),
                (-5.0, 0.75),
                (+5.0, 0.75),
                (0.0, 0.90),
                (-5.0, 0.90),
                (+5.0, 0.90),
                (0.0, 1.00),
            ]

            lead_trials = [
                0.000,
                0.040,
                0.080,
                0.120,
                0.160,
                0.200,
            ]

        else:
            attack_modes = ["DIRECT"]

            correction_trials = [
                (0.0, 1.00),
                (-4.0, 1.00),
                (+4.0, 1.00),
            ]

            lead_trials = [
                0.000,
                0.020,
                0.040,
            ]

        valid_trials = []
        failure_trials = []

        for attack_mode in attack_modes:
            for correction_deg, speed_scale in correction_trials:
                for lead_s in lead_trials:
                    r = evaluate_once(
                        contact_time=contact_time,
                        goal_target_y=goal_target_y,
                        desired_out_speed=desired_out_speed,
                        attack_mode=attack_mode,
                        direction_correction_deg=correction_deg,
                        speed_scale=speed_scale,
                        command_lead_s=lead_s,
                        live_viewer=live_viewer,
                        live_render=live_render,
                    )

                    if r.get("valid", False):
                        valid_trials.append(r)
                        # A valid physical contact is enough for this local
                        # correction/speed pair. Larger lead would only move
                        # the impact earlier.
                        break

                    failure_trials.append(r)

                    # Timing lead can only repair NO_CONTACT.  Geometry / IK /
                    # workspace failures should immediately move to the next
                    # correction or speed-scale trial.
                    if r.get("reason") != "NO_CONTACT":
                        break

        if valid_trials:
            return max(
                valid_trials,
                key=lambda r: r["score"],
            )

        if failure_trials:
            # Prefer the failed trial that came physically closest to contact.
            # Failures without a measured distance sort behind NO_CONTACT.
            return min(
                failure_trials,
                key=lambda r: float(
                    r.get(
                        "min_contact_distance",
                        float("inf"),
                    )
                ),
            )

        return {
            "valid": False,
            "reason": "NO_TRIALS",
            "score": -1e9,
        }

    coarse_results, coarse_best = run_goal_search(
        stage_name="COARSE",
        t_values=coarse_t,
        goal_y_values=coarse_goal_y,
        desired_speed_values=coarse_desired_out_speeds,
        evaluate_fn=lambda **kw: evaluate_fn(
            **kw,
            live_viewer=search_viewer,
            live_render=args.render_search,
        ),
    )

    if coarse_best is None:
        restore_hit_state(assembly, initial_state)
        if search_viewer is not None:
            search_viewer.close()
        raise RuntimeError(
            "No valid coarse candidates. "
            "Inspect INVALID reasons above."
        )

    print("")
    print("=== COARSE BEST ===")
    print(
        f"t={coarse_best['contact_time']:.3f}s  "
        f"contact=({coarse_best['contact_x']:+.3f},"
        f"{coarse_best['contact_y']:+.3f})"
    )
    print(
        f"goal_y*={coarse_best['goal_target_y']:+.3f}  "
        f"desired_vout={coarse_best['desired_out_speed']:.2f}m/s  "
        f"mode={coarse_best.get('attack_mode', 'DIRECT')}  "
        f"hit={coarse_best['hit_speed']:.2f}m/s  "
        f"corr={coarse_best.get('direction_correction_deg', 0.0):+.1f}deg  "
        f"event={coarse_best['event']}  "
        f"score={coarse_best['score']:+.3f}  "
        f"lead={coarse_best.get('command_lead_s', 0.0)*1000:.0f}ms"
    )

    fine_results = []
    final_best = coarse_best

    if not args.coarse_only:
        # Fine Stage-5 search around the best explicit offensive intention:
        # 3 contact points × 3 goal entry points × 3 desired out-speeds.
        bx = coarse_best["contact_x"]
        bgy = coarse_best["goal_target_y"]
        bvo = coarse_best["desired_out_speed"]

        fine_x = unique_values(
            [
                min(front_x, bx + 0.030),
                bx,
                max(back_x, bx - 0.030),
            ]
        )

        fine_t = []
        for x_candidate in fine_x:
            tc = reference_crossing_time(
                puck_reference,
                x_candidate,
            )
            if tc is not None:
                fine_t.append(tc)

        fine_t = unique_values(fine_t)

        goal_delta = 0.20 * legal_goal_y
        fine_goal_y = unique_values(
            [
                float(np.clip(
                    bgy - goal_delta,
                    -0.85 * legal_goal_y,
                    +0.85 * legal_goal_y,
                )),
                bgy,
                float(np.clip(
                    bgy + goal_delta,
                    -0.85 * legal_goal_y,
                    +0.85 * legal_goal_y,
                )),
            ]
        )

        fine_desired_out_speeds = unique_values(
            [
                max(0.85, bvo - 0.15),
                bvo,
                min(1.85, bvo + 0.15),
            ]
        )

        fine_results, fine_best = run_goal_search(
            stage_name="FINE",
            t_values=fine_t,
            goal_y_values=fine_goal_y,
            desired_speed_values=fine_desired_out_speeds,
            evaluate_fn=lambda **kw: evaluate_fn(
                **kw,
                live_viewer=search_viewer,
                live_render=args.render_search,
            ),
        )

        if (
            fine_best is not None
            and fine_best["score"] > final_best["score"]
        ):
            final_best = fine_best

    if search_viewer is not None:
        search_viewer.close()

    # ========================================================
    # FAST ACCEPT PATH
    # ========================================================
    ranked_candidates = ranked_unique_candidates(
        coarse_results,
        fine_results,
    )

    if not ranked_candidates:
        restore_hit_state(assembly, initial_state)
        raise RuntimeError("No valid candidates available after search")

    verification_reports = []

    if args.verify:
        verification_reports, robust_goals, robust_fallbacks = (
            choose_verified_candidates(
                ranked_candidates=ranked_candidates,
                verify_top_k=args.verify_top_k,
                verify_repeats=args.verify_repeats,
                evaluate_once_fn=evaluate_once,
            )
        )

        if robust_goals:
            accepted_queue = [pair[0] for pair in robust_goals]
            acceptance_mode = "ROBUST_GOAL"
        elif robust_fallbacks:
            accepted_queue = [pair[0] for pair in robust_fallbacks]
            acceptance_mode = "ROBUST_RETURN_FALLBACK"
        else:
            restore_hit_state(assembly, initial_state)
            raise RuntimeError(
                "Search found candidates, but none survived verification"
            )
    else:
        # Fast dataset path: trust search ranking, then execute the selected
        # trajectory exactly once. The ACTUAL execution result is stored in
        # planner_label.json and can be filtered later.
        accepted_queue = ranked_candidates
        acceptance_mode = "SEARCH_BEST_NO_REPEAT_VERIFY"

    final_best = accepted_queue[0]

    print("")
    print("======================================================")
    print("PLANNER BEST")
    print("======================================================")
    print(f"accept mode  : {acceptance_mode}")
    print(f"seed         : {args.seed}")
    print(
        f"contact time : {final_best['contact_time']:.4f} s"
    )
    print(
        f"contact point: "
        f"({final_best['contact_x']:+.4f}, "
        f"{final_best['contact_y']:+.4f}) m"
    )
    print(
        f"goal target y: "
        f"{final_best['goal_target_y']:+.4f} m"
    )
    print(
        f"desired vout : "
        f"{final_best['desired_out_speed']:.3f} m/s"
    )
    print(
        f"attack mode  : "
        f"{final_best.get('attack_mode', 'DIRECT')}"
    )
    print(
        f"nominal angle: "
        f"{final_best['nominal_theta_deg']:+.2f} deg"
    )
    print(
        f"dir correction: "
        f"{final_best.get('direction_correction_deg', 0.0):+.2f} deg"
    )
    print(
        f"hit speed    : "
        f"{final_best['hit_speed']:.3f} m/s"
    )
    print(
        f"command lead : "
        f"{final_best.get('command_lead_s', 0.0)*1000:.0f} ms"
    )
    print(f"event        : {final_best['event']}")
    print(f"score        : {final_best['score']:+.4f}")
    print(
        f"prep avg v   : "
        f"{final_best['required_avg_prep_speed']:.3f} m/s"
    )
    print("======================================================")

    # --------------------------------------------------------
    # Save planner summary BEFORE live replay.
    # --------------------------------------------------------
    output_dir = (
        ROOT
        / "outputs"
        / "direct_planner_stage5_v1_5"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")

    if args.summary_path is not None:
        summary_path = args.summary_path
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    else:
        summary_path = (
            output_dir
            / f"seed_{args.seed:06d}_{stamp}.json"
        )

    summary = {
        "seed": int(args.seed),
        "serve": serve,
        "stage5_physics": stage5_physics,
        "reference_horizon_s": float(reference_horizon_s),
        "reference_side_bounces": int(reference_side_bounces),
        "accepted_serve_attempt": int(accepted_serve_attempt),
        "search_window": {
            "front_x": float(front_x),
            "back_x": float(back_x),
            "t_front": float(t_front),
            "t_back": float(t_back),
        },
        "coarse_best": coarse_best,
        "final_best": final_best,
        "acceptance_mode": acceptance_mode,
        "verification_reports": verification_reports,
        "coarse_candidate_count": len(coarse_results),
        "fine_candidate_count": len(fine_results),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"planner summary: {summary_path}")

    # --------------------------------------------------------
    # Execute ONLY the selected BEST once.
    #
    # Hidden search rollouts are never recorded. If --collect is supplied,
    # arm the existing 640x360/30-FPS synchronized recorder now, after search.
    # --------------------------------------------------------
    restore_hit_state(
        assembly,
        initial_state,
    )

    collection_dir = None

    if args.collect:
        lock_path = (
            ROOT
            / "outputs"
            / ".stage5_episode_recorder.lock"
        )
        lock_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # configure_episode_recorder uses timestamp-based folder names.
        # Serialize only folder creation so parallel seed workers cannot race.
        with lock_path.open("w") as lock_fp:
            fcntl.flock(
                lock_fp.fileno(),
                fcntl.LOCK_EX,
            )

            collector.configure_episode_recorder(
                scene,
                config,
                assembly,
                left_ee,
            )

            collection_dir = Path(
                collector._EP_REC["out_dir"]
            )

            fcntl.flock(
                lock_fp.fileno(),
                fcntl.LOCK_UN,
            )

        # Start the recorder at the exact restored inbound state WITHOUT
        # advancing physics. evaluate_candidate will then set DIRECT_PREP,
        # DIRECT_HIT, and FLIGHT phases.
        collector.episode_recorder_after_step(
            scene
        )

        print("")
        print("=== SELECTED EXPERT RECORDING ARMED ===")
        print(f"seed   : {args.seed}")
        print(f"output : {collection_dir}")
        print("candidate search: NOT RECORDED")
        print("=======================================")

    viewer = None

    if not args.headless:
        print("")
        print("Opening Viewer and executing selected BEST once...")

        viewer = Viewer(renderer)
        viewer.set_scene(scene)
        viewer.set_camera_xyz(
            x=2.1,
            y=-1.7,
            z=3.0,
        )
        viewer.set_camera_rpy(
            r=0,
            p=-0.75,
            y=-2.46,
        )

        scene.update_render()
        viewer.render()
    else:
        print("")
        print("Headless selected-BEST execution...")

    actual = evaluate_once(
        contact_time=final_best["contact_time"],
        goal_target_y=final_best["goal_target_y"],
        desired_out_speed=final_best["desired_out_speed"],
        attack_mode=final_best.get(
            "attack_mode",
            "DIRECT",
        ),
        direction_correction_deg=final_best.get(
            "direction_correction_deg",
            0.0,
        ),
        speed_scale=final_best.get(
            "speed_scale",
            1.0,
        ),
        command_lead_s=final_best.get(
            "command_lead_s",
            0.0,
        ),
        live_viewer=viewer,
        live_render=(viewer is not None),
    )

    actual_event = actual.get(
        "event",
        actual.get("reason", "UNKNOWN"),
    )

    if args.collect:
        collector.episode_finish(
            actual_event
        )

        planner_label = {
            "planner": "stage5_v1_5_edge_glide_bank_fallback",
            "seed": int(args.seed),
            "serve": serve,
            "stage5_physics": stage5_physics,
            "reference_side_bounces": int(reference_side_bounces),
            "accepted_serve_attempt": int(accepted_serve_attempt),
            "acceptance_mode": acceptance_mode,
            "selected_search_candidate": final_best,
            "actual_execution": actual,
            "actual_event": actual_event,
            "search_predicted_event": final_best.get("event"),
            "search_predicted_score": float(final_best.get("score", -1e9)),
        }

        label_path = (
            collection_dir
            / "planner_label.json"
        )
        label_path.write_text(
            json.dumps(
                planner_label,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            f"planner label: {label_path}"
        )

    print("")
    print("======================================================")
    print("SELECTED BEST ACTUAL EXECUTION")
    print("======================================================")
    print(f"seed         : {args.seed}")
    print(f"predicted    : {final_best.get('event')}")
    print(f"actual       : {actual_event}")
    print(f"valid        : {actual.get('valid')}")
    if actual.get("valid", False):
        print(
            f"contact      : "
            f"({actual['contact_x']:+.4f}, "
            f"{actual['contact_y']:+.4f})"
        )
        print(
            f"goal target y: "
            f"{actual['goal_target_y']:+.4f} m"
        )
        print(
            f"attack mode  : "
            f"{actual.get('attack_mode', 'DIRECT')}"
        )
        print(
            f"hit speed    : "
            f"{actual['hit_speed']:.3f} m/s"
        )
        print(
            f"score        : "
            f"{actual['score']:+.4f}"
        )
        print(
            f"tool error   : "
            f"{actual['tool_follow_max_error']*1000:.2f} mm"
        )
    if collection_dir is not None:
        print(f"dataset dir  : {collection_dir}")
    print("======================================================")

    summary["actual_execution"] = actual
    summary["actual_event"] = actual_event
    summary["collection_dir"] = (
        None
        if collection_dir is None
        else str(collection_dir)
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if viewer is not None:
        deadline = (
            time.monotonic()
            + max(0.0, float(args.hold))
        )
        while (
            not viewer.closed
            and time.monotonic() < deadline
        ):
            scene.update_render()
            viewer.render()

        viewer.close()


if __name__ == "__main__":
    main()
