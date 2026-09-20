"""Open the air-hockey scene with two interactive UR5-WSG arms.

Run from the repository root with ``python scripts/preview_air_hockey_assets.py``.
Use ``--check`` to inspect the scene without a graphics device.
"""

import argparse
import sys
import time
import json
import atexit

import imageio.v3 as iio
from pathlib import Path

import numpy as np
import sapien.core as sapien


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from air_hockey_assets import (  # noqa: E402
    build_puck,
    build_striker,
    build_table,
    configure_air_hockey_physics,
    load_asset_config,
)
from scripts.ur5_reachability import install_drag_controller  # noqa: E402
from scripts.puck_keyboard_control import install_puck_keyboard_controller  # noqa: E402


def populate_static_preview(scene, config, *, render=True):
    """Add only a fixed table, central puck, and one handle-free disc per side."""
    table = build_table(scene, config, render=render)
    puck = build_puck(scene, config, render=render, static=True)
    left = build_striker(scene, config, render=render, static=True,
                         x=-0.65, y=0, name="left_striker")
    right = build_striker(scene, config, render=render, static=True,
                          x=0.65, y=0, name="right_striker")
    return {"table": table, "puck": puck, "left_striker": left, "right_striker": right}


def robot_mount_poses(config):
    """Put both UR5 bases behind goal centers at the playing-surface height."""
    x = config["table"]["length_m"] / 2 + config["robot_preview"]["base_offset_from_end_m"]
    z = config["table"]["top_height_m"]
    return {
        "left_robot": sapien.Pose([-x, 0, z], [1, 0, 0, 0]),
        "right_robot": sapien.Pose([x, 0, z], [0, 0, 0, 1]),
    }


def finger_gap_center_y(robot):
    """Find the free center between the WSG's actual finger collision meshes."""
    boundaries = []
    for name in ("finger_left", "finger_right"):
        link = next(link for link in robot.get_links() if link.get_name() == name)
        body = link.entity.find_component_by_type(sapien.physx.PhysxArticulationLinkComponent)
        vertices = []
        for shape in body.collision_shapes:
            points = np.asarray(shape.vertices)
            pose = (link.entity.pose * shape.local_pose).to_transformation_matrix()
            vertices.append((pose[:3, :3] @ points.T).T + pose[:3, 3])
        if not vertices:
            raise ValueError(f"{name} has no collision geometry for a physical grasp")
        ys = np.concatenate(vertices)[:, 1]
        boundaries.append((float(np.mean(ys)), float(np.min(ys)), float(np.max(ys))))
    lower, upper = sorted(boundaries, key=lambda item: item[0])
    return (lower[2] + upper[1]) / 2


def populate_robot_preview(scene, config, *, render=True, interactive_physics=True):
    """Add two UR5-WSG arms; optionally attach dynamic striker discs."""
    if interactive_physics:
        result = {"table": build_table(scene, config, render=render),
                  "puck": build_puck(scene, config, render=render)}
    else:
        result = populate_static_preview(scene, config, render=render)
    mounts = robot_mount_poses(config)
    robot_config = config["robot_preview"]
    urdf_path = ROOT / robot_config["urdf_path"]
    if not urdf_path.is_file():
        raise FileNotFoundError(f"UR5-WSG URDF not found: {urdf_path}")

    for side in ("left", "right"):
        mount = mounts[f"{side}_robot"]
        height = config["table"]["top_height_m"]
        builder = scene.create_actor_builder()
        builder.set_physx_body_type("static")
        stand_pose = sapien.Pose([mount.p[0], 0, height / 2])
        half_size = [robot_config["stand_length_m"] / 2,
                     robot_config["stand_width_m"] / 2, height / 2]
        builder.add_box_collision(pose=stand_pose, half_size=half_size)
        if render:
            builder.add_box_visual(pose=stand_pose, half_size=half_size,
                                   material=[0.35, 0.39, 0.45], name=f"{side}_stand")
        result[f"{side}_stand"] = builder.build(name=f"{side}_stand")

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(str(urdf_path))
        robot.set_root_pose(mount)
        pan_joint = next(joint for joint in robot.get_active_joints()
                         if joint.name == "shoulder_pan_joint")
        pan_joint.set_limits(np.asarray([robot_config["shoulder_pan_limits_rad"]],
                                        dtype=np.float32))
        home_qpos = np.asarray(robot_config["home_qpos"], dtype=float)
        limits = robot.get_qlimits()
        if len(home_qpos) != robot.dof or np.any(home_qpos < limits[:, 0]) or np.any(home_qpos > limits[:, 1]):
            raise ValueError("robot_preview.home_qpos must match UR5 joint count and limits")
        robot.set_qpos(home_qpos)
        if interactive_physics:
            grasp = config["grasp"]
            finger_material = scene.create_physical_material(
                grasp["finger_static_friction"], grasp["finger_dynamic_friction"],
                grasp["finger_restitution"])
            for link in robot.get_links():
                if link.get_name() in ("finger_left", "finger_right"):
                    body = link.entity.find_component_by_type(
                        sapien.physx.PhysxArticulationLinkComponent)
                    for shape in body.collision_shapes:
                        shape.physical_material = finger_material
            for joint, target in zip(robot.get_active_joints(), home_qpos):
                if joint.name in robot_config["arm_joint_names"]:
                    joint.set_drive_property(10000, 1000, force_limit=10000)
                    joint.set_drive_target(float(target))
                else:
                    joint.set_drive_property(
                        grasp["finger_drive_stiffness_N_m"],
                        grasp["finger_drive_damping_Ns_m"],
                        force_limit=grasp["finger_force_limit_N"])
                    closed = grasp["finger_closed_target_m"]
                    joint.set_drive_target(-closed if joint.name.endswith("left") else closed)

            ee = next(link.entity for link in robot.get_links()
                      if link.get_name() == robot_config["ee_link_name"])
            striker = build_striker(scene, config, render=render,
                                    x=float(ee.pose.p[0]), y=finger_gap_center_y(robot),
                                    name=f"{side}_striker")
            result[f"{side}_striker"] = striker
        result[f"{side}_robot"] = robot
    return result



def predict_defend_intercept(x, y, vx, vy, x_defend, y_limit, max_time=3.0):
    """
    第一版 2D puck predictor:
    - x 方向匀速
    - y 方向考虑上下边界镜面反弹
    - 使用当前 SAPIEN 实测 x/y/vx/vy
    """

    # 左侧机器人只处理向左运动的 puck
    if vx >= -1e-5:
        return None

    t = (x_defend - x) / vx

    if t <= 0.0 or t > max_time:
        return None

    # y 在 [-y_limit, +y_limit] 之间反弹
    interval = 2.0 * y_limit
    period = 2.0 * interval

    u0 = y + y_limit
    u = (u0 + vy * t) % period

    if u > interval:
        u = period - u

    y_hit = u - y_limit

    return float(t), float(y_hit)


def cartesian_rate_step(position, velocity, target, vmax, amax, dt):
    """
    1D 速度/加速度受限轨迹。
    根据剩余距离自动提前减速，避免撞到目标后剧烈反向。
    """
    error = float(target - position)

    if abs(error) < 0.002 and abs(velocity) < 0.12:
        return float(target), 0.0

    direction = 1.0 if error > 0.0 else -1.0

    # 保证仍有足够距离刹车
    braking_speed = (2.0 * amax * abs(error)) ** 0.5
    desired_speed = direction * min(vmax, braking_speed)

    max_dv = amax * dt
    dv = float(np.clip(desired_speed - velocity, -max_dv, max_dv))
    new_velocity = velocity + dv
    new_position = position + new_velocity * dt

    # 防止数值上越过目标
    if (target - position) * (target - new_position) <= 0.0:
        return float(target), 0.0

    return float(new_position), float(new_velocity)


def _copy_pose(pose):
    return sapien.Pose(
        np.asarray(pose.p, dtype=float).copy(),
        np.asarray(pose.q, dtype=float).copy(),
    )




# ============================================================
# CONTINUOUS RGB EPISODE RECORDER
#
# Demonstration timeline:
#
# DEFEND
# CONTACT_HOLD
# HIT_PREP
# HIT_EXECUTE
# FLIGHT
#
# Snapshot validation / candidate search:
#   RGB OFF
#   episode clock PAUSED
# ============================================================

_EP_REC = None


def _ep_pose_dict(pose):
    return {
        "p": [
            float(x)
            for x in np.asarray(
                pose.p,
                dtype=float,
            )
        ],
        "q": [
            float(x)
            for x in np.asarray(
                pose.q,
                dtype=float,
            )
        ],
    }


def _ep_joint_drive_target(joint):
    try:
        value = None

        if hasattr(
            joint,
            "get_drive_target",
        ):
            value = (
                joint.get_drive_target()
            )

        elif hasattr(
            joint,
            "drive_target",
        ):
            value = (
                joint.drive_target
            )

        if value is None:
            return None

        arr = np.asarray(
            value,
            dtype=float,
        ).reshape(-1)

        if arr.size == 0:
            return None

        if arr.size == 1:
            return float(
                arr[0]
            )

        return [
            float(x)
            for x in arr
        ]

    except Exception:
        return None


def _ep_look_at_pose(
    eye,
    target,
):
    eye = np.asarray(
        eye,
        dtype=float,
    )

    target = np.asarray(
        target,
        dtype=float,
    )

    world_up = np.asarray(
        [0.0, 0.0, 1.0],
        dtype=float,
    )

    forward = target - eye

    forward /= np.linalg.norm(
        forward
    )

    left = np.cross(
        world_up,
        forward,
    )

    left /= np.linalg.norm(
        left
    )

    up = np.cross(
        forward,
        left,
    )

    T = np.eye(
        4,
        dtype=float,
    )

    T[:3, :3] = np.stack(
        [
            forward,
            left,
            up,
        ],
        axis=1,
    )

    T[:3, 3] = eye

    return sapien.Pose(T)


def _ep_write_meta():
    state = _EP_REC

    if state is None:
        return

    meta = {
        "format": (
            "air_hockey_episode_rgb_v1"
        ),
        "fps": state["fps"],
        "width": state["width"],
        "height": state["height"],
        "timestep": state["timestep"],
        "frame_count": (
            state["frame_index"]
        ),
        "episode_time": (
            state["episode_time"]
        ),
        "episode_steps": (
            state["episode_steps"]
        ),
        "planner_steps_excluded": (
            state["planner_steps"]
        ),
        "started": state["started"],
        "finished": state["finished"],
        "finish_reason": (
            state["finish_reason"]
        ),
        "timeline_policy": {
            "snapshot_test": "excluded",
            "candidate_search": "excluded",
            "hit_prep": "included",
            "best_live_replay": "included",
        },
    }

    state["meta_path"].write_text(
        json.dumps(
            meta,
            indent=2,
            ensure_ascii=False,
        )
    )


def _ep_close():
    state = _EP_REC

    if state is None:
        return

    _ep_write_meta()

    fp = state.get(
        "frames_fp"
    )

    if (
        fp is not None
        and not fp.closed
    ):
        fp.flush()
        fp.close()


def configure_episode_recorder(
    scene,
    config,
    assembly,
    left_ee,
):
    global _EP_REC

    width = 640
    height = 360
    fps = 30.0

    camera = scene.add_camera(
        "episode_rgb_front",
        width,
        height,
        np.deg2rad(55.0),
        0.05,
        10.0,
    )

    pose = _ep_look_at_pose(
        eye=[
            -0.15,
            -1.75,
            2.35,
        ],
        target=[
            0.0,
            0.0,
            float(
                config[
                    "table"
                ][
                    "top_height_m"
                ]
            ),
        ],
    )

    if hasattr(
        camera,
        "set_entity_pose",
    ):
        camera.set_entity_pose(
            pose
        )

    elif hasattr(
        camera,
        "set_pose",
    ):
        camera.set_pose(
            pose
        )

    else:
        camera.entity.set_pose(
            pose
        )

    root = (
        ROOT
        / "outputs"
        / "episode_rgb_v1"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    out_dir = (
        root
        / f"episode_{stamp}"
    )

    suffix = 1

    while out_dir.exists():
        out_dir = (
            root
            / (
                f"episode_{stamp}"
                f"_{suffix:02d}"
            )
        )

        suffix += 1

    rgb_dir = (
        out_dir
        / "rgb"
    )

    rgb_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    frames_path = (
        out_dir
        / "frames.jsonl"
    )

    meta_path = (
        out_dir
        / "meta.json"
    )

    puck_body = (
        assembly["puck"]
        .find_component_by_type(
            sapien.physx
            .PhysxRigidDynamicComponent
        )
    )

    if puck_body is None:
        raise RuntimeError(
            "recorder cannot find puck body"
        )

    _EP_REC = {
        "scene": scene,
        "camera": camera,
        "assembly": assembly,
        "left_ee": left_ee,
        "puck_body": puck_body,

        "out_dir": out_dir,
        "rgb_dir": rgb_dir,
        "frames_path": frames_path,
        "meta_path": meta_path,
        "frames_fp": (
            frames_path.open("w")
        ),

        "width": width,
        "height": height,
        "fps": fps,

        "timestep": float(
            config[
                "physics"
            ][
                "timestep_s"
            ]
        ),

        "started": False,
        "paused": False,
        "finished": False,

        "phase": "WAIT_INBOUND",
        "pause_reason": None,
        "finish_reason": None,

        "episode_time": 0.0,
        "next_frame_time": 0.0,

        "episode_steps": 0,
        "planner_steps": 0,
        "frame_index": 0,
    }

    atexit.register(
        _ep_close
    )

    _ep_write_meta()

    print("")
    print(
        "=== RGB EPISODE RECORDER ARMED ==="
    )
    print(
        "RGB: 640x360 @ 30 FPS"
    )
    print(
        "auto start: puck vx < -0.05"
    )
    print(
        "output:",
        out_dir,
    )
    print(
        "candidate RGB: EXCLUDED"
    )
    print(
        "=================================="
    )


def episode_set_phase(
    phase,
):
    state = _EP_REC

    if state is None:
        return

    if state["phase"] != phase:
        state["phase"] = phase

        if state["started"]:
            print(
                f"[REC] PHASE -> {phase}"
            )


def episode_pause(
    reason,
):
    state = _EP_REC

    if (
        state is None
        or not state["started"]
        or state["finished"]
    ):
        return

    if state["paused"]:
        return

    state["paused"] = True
    state["pause_reason"] = reason

    print(
        f"[REC] PAUSE: {reason} "
        f"at t="
        f"{state['episode_time']:.3f}s"
    )


def episode_resume(
    phase,
):
    state = _EP_REC

    if (
        state is None
        or not state["started"]
        or state["finished"]
    ):
        return

    state["paused"] = False
    state["pause_reason"] = None

    episode_set_phase(
        phase
    )

    print(
        f"[REC] RESUME: {phase} "
        f"at t="
        f"{state['episode_time']:.3f}s"
    )


def _ep_capture_frame(
    force=False,
):
    state = _EP_REC

    if (
        state is None
        or not state["started"]
        or state["finished"]
    ):
        return

    if (
        state["paused"]
        and not force
    ):
        return

    scene = state["scene"]
    camera = state["camera"]
    assembly = state["assembly"]

    scene.update_render()
    camera.take_picture()

    color = np.asarray(
        camera.get_picture(
            "Color"
        )
    )

    rgb = np.clip(
        color[..., :3] * 255.0,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    frame_index = int(
        state["frame_index"]
    )

    rgb_name = (
        f"{frame_index:06d}.png"
    )

    iio.imwrite(
        state["rgb_dir"]
        / rgb_name,
        rgb,
    )

    left_robot = (
        assembly["left_robot"]
    )

    right_robot = (
        assembly["right_robot"]
    )

    puck = assembly["puck"]

    puck_vel = np.asarray(
        state["puck_body"]
        .get_linear_velocity(),
        dtype=float,
    )

    left_joints = (
        left_robot.get_active_joints()
    )

    record = {
        "frame": frame_index,

        "episode_time": float(
            state["episode_time"]
        ),

        "phase": state["phase"],

        "rgb": (
            "rgb/"
            + rgb_name
        ),

        "left_robot": {
            "q": [
                float(x)
                for x in np.asarray(
                    left_robot.get_qpos(),
                    dtype=float,
                )
            ],

            "dq": [
                float(x)
                for x in np.asarray(
                    left_robot.get_qvel(),
                    dtype=float,
                )
            ],
        },

        "right_robot": {
            "q": [
                float(x)
                for x in np.asarray(
                    right_robot.get_qpos(),
                    dtype=float,
                )
            ],

            "dq": [
                float(x)
                for x in np.asarray(
                    right_robot.get_qvel(),
                    dtype=float,
                )
            ],
        },

        "tcp": _ep_pose_dict(
            state["left_ee"].pose
        ),

        "action": {
            "left_joint_drive_target": [
                _ep_joint_drive_target(
                    joint
                )
                for joint in left_joints
            ],
        },

        "puck_gt": {
            "pose": _ep_pose_dict(
                puck.get_pose()
            ),

            "linear_velocity": [
                float(x)
                for x in puck_vel
            ],
        },

        "left_striker_pose": (
            _ep_pose_dict(
                assembly[
                    "left_striker"
                ].get_pose()
            )
        ),
    }

    state["frames_fp"].write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )

    state["frames_fp"].flush()

    state["frame_index"] += 1

    if (
        frame_index % 10
        == 0
    ):
        print(
            f"[REC] "
            f"frame={frame_index:04d} "
            f"t="
            f"{state['episode_time']:.3f}s "
            f"phase={state['phase']}"
        )


def episode_recorder_after_step(
    scene,
):
    state = _EP_REC

    if (
        state is None
        or state["finished"]
    ):
        return

    vel = np.asarray(
        state["puck_body"]
        .get_linear_velocity(),
        dtype=float,
    )

    speed_xy = float(
        np.linalg.norm(
            vel[:2]
        )
    )

    # Start only when a real inbound
    # puck is moving toward the left side.
    if not state["started"]:

        if (
            vel[0] < -0.05
            and speed_xy > 0.10
        ):
            state["started"] = True
            state["paused"] = False
            state["phase"] = "DEFEND"

            state["episode_time"] = 0.0
            state["next_frame_time"] = 0.0

            print("")
            print(
                "=================================="
            )
            print(
                "[REC] EPISODE START"
            )
            print(
                f"[REC] puck "
                f"vx={vel[0]:+.3f} "
                f"vy={vel[1]:+.3f}"
            )
            print(
                "=================================="
            )

            _ep_capture_frame(
                force=True
            )

            state[
                "next_frame_time"
            ] = (
                1.0
                / state["fps"]
            )

        return

    # Candidate / snapshot physics
    # never advances demonstration time.
    if state["paused"]:
        state["planner_steps"] += 1
        return

    state["episode_steps"] += 1

    state["episode_time"] += (
        state["timestep"]
    )

    if (
        state["phase"]
        == "HIT_EXECUTE"
        and speed_xy > 0.08
    ):
        episode_set_phase(
            "FLIGHT"
        )

    if (
        state["episode_time"]
        + 1e-12
        >= state["next_frame_time"]
    ):
        _ep_capture_frame()

        frame_dt = (
            1.0
            / state["fps"]
        )

        while (
            state["next_frame_time"]
            <= state["episode_time"]
            + 1e-12
        ):
            state[
                "next_frame_time"
            ] += frame_dt


def episode_finish(
    reason,
):
    state = _EP_REC

    if (
        state is None
        or not state["started"]
        or state["finished"]
    ):
        return

    state["paused"] = False

    _ep_capture_frame(
        force=True
    )

    state["finished"] = True
    state["finish_reason"] = str(
        reason
    )

    _ep_write_meta()

    state["frames_fp"].flush()
    state["frames_fp"].close()

    print("")
    print(
        "=================================="
    )
    print(
        "[REC] EPISODE FINISH"
    )
    print(
        "reason:",
        state["finish_reason"],
    )
    print(
        "frames:",
        state["frame_index"],
    )
    print(
        "episode_time:",
        f"{state['episode_time']:.3f}s",
    )
    print(
        "planner_steps_excluded:",
        state["planner_steps"],
    )
    print(
        "output:",
        state["out_dir"],
    )
    print(
        "=================================="
    )


# ============================================================
# LEFT STRIKER KINEMATIC TOOL FOLLOWER
#
# left striker 不再是靠摩擦夹持的自由 dynamic tool。
# 它仍然拥有碰撞形状，但 pose 由 left EE 唯一决定。
# ============================================================

_LEFT_TOOL_FOLLOWER = None


def _copy_tool_pose(pose):
    return sapien.Pose(
        np.asarray(
            pose.p,
            dtype=float,
        ).copy(),
        np.asarray(
            pose.q,
            dtype=float,
        ).copy(),
    )


def configure_left_kinematic_tool(
    left_striker,
    left_ee,
):
    global _LEFT_TOOL_FOLLOWER

    body = left_striker.find_component_by_type(
        sapien.physx.PhysxRigidDynamicComponent
    )

    if body is None:
        raise RuntimeError(
            "left striker has no "
            "PhysxRigidDynamicComponent"
        )

    # 保存 tool 相对 EE 的完整 6D 刚体变换。
    tool_local_pose = (
        left_ee.pose.inv()
        * left_striker.get_pose()
    )

    body.set_kinematic(True)

    expected = (
        left_ee.pose
        * tool_local_pose
    )

    # 初始化时直接严格对齐。
    left_striker.set_pose(
        _copy_tool_pose(expected)
    )

    body.set_kinematic_target(
        _copy_tool_pose(expected)
    )

    _LEFT_TOOL_FOLLOWER = {
        "entity": left_striker,
        "body": body,
        "ee": left_ee,
        "local_pose": _copy_tool_pose(
            tool_local_pose
        ),
        "max_error": 0.0,
        "last_error": 0.0,
        "step_count": 0,
    }

    print("")
    print(
        "=== KINEMATIC STRIKER TOOL ENABLED ==="
    )
    print(
        "mode: EE pose -> kinematic target"
    )
    print(
        "dynamic puck collision remains enabled"
    )
    print(
        "striker is no longer friction-grasped"
    )
    print(
        "======================================"
    )

    return body


def _expected_left_tool_pose():
    state = _LEFT_TOOL_FOLLOWER

    if state is None:
        return None

    return (
        state["ee"].pose
        * state["local_pose"]
    )


def sync_left_tool_target():
    state = _LEFT_TOOL_FOLLOWER

    if state is None:
        return

    expected = _expected_left_tool_pose()

    state["body"].set_kinematic_target(
        _copy_tool_pose(expected)
    )


def measure_left_tool_follow_error():
    state = _LEFT_TOOL_FOLLOWER

    if state is None:
        return 0.0

    expected = _expected_left_tool_pose()

    actual = np.asarray(
        state["entity"].get_pose().p,
        dtype=float,
    )

    desired = np.asarray(
        expected.p,
        dtype=float,
    )

    error = float(
        np.linalg.norm(
            actual - desired
        )
    )

    state["last_error"] = error

    state["max_error"] = max(
        state["max_error"],
        error,
    )

    state["step_count"] += 1

    return error


def physics_step(scene):
    sync_left_tool_target()

    scene.step()

    error = (
        measure_left_tool_follow_error()
    )

    sync_left_tool_target()

    episode_recorder_after_step(
        scene
    )

    return error


def reset_left_tool_follow_stats():
    state = _LEFT_TOOL_FOLLOWER

    if state is None:
        return

    state["max_error"] = 0.0
    state["last_error"] = 0.0
    state["step_count"] = 0


def get_left_tool_follow_max_error():
    state = _LEFT_TOOL_FOLLOWER

    if state is None:
        return 0.0

    return float(
        state["max_error"]
    )


def restore_left_tool_from_robot():
    # robot qpos 已 restore 后，
    # striker 不从独立 snapshot 决定位姿。
    #
    # 唯一真值：
    # restored EE pose * fixed tool transform
    state = _LEFT_TOOL_FOLLOWER

    if state is None:
        return

    expected = _expected_left_tool_pose()

    state["entity"].set_pose(
        _copy_tool_pose(expected)
    )

    state["body"].set_kinematic_target(
        _copy_tool_pose(expected)
    )

    state["max_error"] = 0.0
    state["last_error"] = 0.0



def snapshot_rigid(entity):
    body = entity.find_component_by_type(
        sapien.physx.PhysxRigidDynamicComponent
    )
    if body is None:
        raise RuntimeError(
            f"{entity.get_name()} has no dynamic rigid body"
        )

    return {
        "pose": _copy_pose(entity.get_pose()),
        "linear_velocity": np.asarray(
            body.get_linear_velocity(),
            dtype=float,
        ).copy(),
        "angular_velocity": np.asarray(
            body.get_angular_velocity(),
            dtype=float,
        ).copy(),
    }


def restore_rigid(entity, state):
    body = entity.find_component_by_type(
        sapien.physx.PhysxRigidDynamicComponent
    )

    entity.set_pose(
        _copy_pose(state["pose"])
    )

    body.set_linear_velocity(
        np.asarray(
            state["linear_velocity"],
            dtype=float,
        )
    )

    body.set_angular_velocity(
        np.asarray(
            state["angular_velocity"],
            dtype=float,
        )
    )


def snapshot_robot(robot):
    return {
        "qpos": np.asarray(
            robot.get_qpos(),
            dtype=float,
        ).copy(),

        "qvel": np.asarray(
            robot.get_qvel(),
            dtype=float,
        ).copy(),
    }


def restore_robot(robot, state):
    qpos = np.asarray(
        state["qpos"],
        dtype=float,
    ).copy()

    qvel = np.asarray(
        state["qvel"],
        dtype=float,
    ).copy()

    robot.set_qpos(qpos)
    robot.set_qvel(qvel)

    # 非常重要：
    # candidate 执行以后 joint drive target 仍可能指向击球姿态。
    # restore 后把 drive target 同步到恢复后的 qpos，
    # 防止下一次 physics_step(scene) 又自己运动。
    for joint, q in zip(
        robot.get_active_joints(),
        qpos,
    ):
        try:
            joint.set_drive_target(
                float(q)
            )
        except Exception:
            pass

        try:
            joint.set_drive_velocity_target(
                0.0
            )
        except Exception:
            pass


def snapshot_hit_state(assembly):
    return {
        "puck": snapshot_rigid(
            assembly["puck"]
        ),

        "left_striker": snapshot_rigid(
            assembly["left_striker"]
        ),

        "right_striker": snapshot_rigid(
            assembly["right_striker"]
        ),

        "left_robot": snapshot_robot(
            assembly["left_robot"]
        ),

        "right_robot": snapshot_robot(
            assembly["right_robot"]
        ),
    }


def restore_hit_state(assembly, state):

    # 先恢复机械臂
    restore_robot(
        assembly["left_robot"],
        state["left_robot"],
    )

    restore_robot(
        assembly["right_robot"],
        state["right_robot"],
    )

    # 再恢复自由刚体
    restore_rigid(
        assembly["puck"],
        state["puck"],
    )

    # left striker 是 EE-attached kinematic tool。
    # 不再从独立 rigid snapshot 恢复。
    restore_left_tool_from_robot()

    restore_rigid(
        assembly["right_striker"],
        state["right_striker"],
    )


def rigid_position_error(entity, state):
    now = np.asarray(
        entity.get_pose().p,
        dtype=float,
    )

    ref = np.asarray(
        state["pose"].p,
        dtype=float,
    )

    return float(
        np.linalg.norm(now - ref)
    )


def move_left_striker_segment(
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
    command_speed,
):
    """
    100 Hz IK controller + full-rate PhysX.

    IK 不再每个 physics step 求解，
    但 physics_step 仍按原 timestep 执行。
    """

    start_xy = np.asarray(
        left_striker.get_pose().p[:2],
        dtype=float,
    ).copy()

    target_xy = np.asarray(
        target_xy,
        dtype=float,
    )

    delta = target_xy - start_xy

    distance = float(
        np.linalg.norm(delta)
    )

    if distance < 1e-5:
        return True

    duration = (
        distance
        / max(
            float(command_speed),
            0.05,
        )
    )

    physics_steps = max(
        1,
        int(
            np.ceil(
                duration / timestep
            )
        ),
    )

    # 100 Hz IK/control.
    control_dt = 0.010

    control_stride = max(
        1,
        int(
            round(
                control_dt / timestep
            )
        ),
    )

    last_q = None

    for step_i in range(
        1,
        physics_steps + 1,
    ):

        need_control = (
            step_i == 1
            or step_i == physics_steps
            or (
                (step_i - 1)
                % control_stride
                == 0
            )
        )

        if need_control:

            alpha = (
                step_i
                / physics_steps
            )

            cmd_xy = (
                start_xy
                + alpha * delta
            )

            ee_world = sapien.Pose(
                [
                    float(
                        cmd_xy[0]
                        - striker_ee_offset_xy[0]
                    ),
                    float(
                        cmd_xy[1]
                        - striker_ee_offset_xy[1]
                    ),
                    float(ee_z),
                ],
                ee_q,
            )

            ee_local = (
                left_robot
                .get_root_pose()
                .inv()
                * ee_world
            )

            q_now = np.asarray(
                left_robot.get_qpos(),
                dtype=float,
            )

            (
                q_sol,
                ik_ok,
                ik_err,
            ) = (
                left_pinocchio
                .compute_inverse_kinematics(
                    left_ee_index,
                    ee_local,
                    initial_qpos=q_now,
                    active_qmask=arm_qmask,
                    max_iterations=100,
                )
            )

            valid = (
                bool(ik_ok)
                and np.all(
                    np.isfinite(q_sol)
                )
                and np.all(
                    q_sol
                    >= left_limits[:, 0]
                    - 1e-5
                )
                and np.all(
                    q_sol
                    <= left_limits[:, 1]
                    + 1e-5
                )
            )

            if not valid:
                return False

            for (
                joint,
                q_target,
            ) in zip(
                left_active_joints,
                q_sol,
            ):

                if (
                    joint.name
                    in arm_joint_names
                ):
                    joint.set_drive_target(
                        float(q_target)
                    )

            last_q = q_sol.copy()

        # PhysX 仍然全频率执行。
        # Kinematic striker follower
        # 也仍然每个 physics step 更新。
        physics_step(scene)


    # 让机械臂追上最后一个 IK target。
    if last_q is not None:

        for (
            joint,
            q_target,
        ) in zip(
            left_active_joints,
            last_q,
        ):
            if (
                joint.name
                in arm_joint_names
            ):
                joint.set_drive_target(
                    float(q_target)
                )

        settle_steps = max(
            1,
            int(
                round(
                    0.04 / timestep
                )
            ),
        )

        for _ in range(
            settle_steps
        ):
            physics_step(scene)

    return True


def simulate_hit_candidate(
    scene,
    timestep,
    assembly,
    saved_state,
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
    rail_width,
    puck_radius,
    striker_radius,
    striker_y_limit,
    goal_width,
    angle_offset_deg,
    hit_speed,
):
    """
    真正执行一次候选 HIT：
    restore -> reposition -> strike -> coast -> score
    """

    restore_hit_state(
        assembly,
        saved_state,
    )

    reset_left_tool_follow_stats()

    puck = assembly["puck"]

    puck_xy = np.asarray(
        puck.get_pose().p[:2],
        dtype=float,
    ).copy()

    striker_xy = np.asarray(
        left_striker.get_pose().p[:2],
        dtype=float,
    ).copy()

    # --------------------------------------------------------
    # opponent goal center
    # --------------------------------------------------------

    goal_plane_x = (
        table_length / 2.0
        - rail_width
    )

    goal_x = goal_plane_x

    goal_center_limit = max(
        0.0,
        goal_width / 2.0
        - puck_radius,
    )

    goal_y = 0.0

    goal_vec = np.asarray(
        [
            goal_x - puck_xy[0],
            goal_y - puck_xy[1],
        ],
        dtype=float,
    )

    base_angle = float(
        np.arctan2(
            goal_vec[1],
            goal_vec[0],
        )
    )

    theta = (
        base_angle
        + np.deg2rad(angle_offset_deg)
    )

    direction = np.asarray(
        [
            np.cos(theta),
            np.sin(theta),
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # 先把 striker 放到 puck 后面
    # --------------------------------------------------------

    contact_distance = (
        puck_radius
        + striker_radius
        + 0.006
    )

    prehit_xy = (
        puck_xy
        - direction * contact_distance
    )

    if abs(prehit_xy[1]) > striker_y_limit:
        return {
            "valid": False,
            "score": -1e9,
            "reason": "PREHIT_Y_LIMIT",
        }

    # 第一步先安全向左退开
    safe_x = min(
        float(striker_xy[0]),
        float(
            puck_xy[0]
            - contact_distance
            - 0.055
        ),
    )

    safe1 = np.asarray(
        [
            safe_x,
            striker_xy[1],
        ],
        dtype=float,
    )

    # 第二步在安全 x 上调整 y
    safe2 = np.asarray(
        [
            safe_x,
            prehit_xy[1],
        ],
        dtype=float,
    )

    # 第三步接近 puck 后方
    reposition_speed = 0.55

    for target in (
        safe1,
        safe2,
        prehit_xy,
    ):

        ok = move_left_striker_segment(
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
            target_xy=target,
            command_speed=reposition_speed,
        )

        if not ok:
            return {
                "valid": False,
                "score": -1e9,
                "reason": "REPOSITION_IK",
            }

    # --------------------------------------------------------
    # strike
    # --------------------------------------------------------

    hit_distance = 0.17

    hit_target_xy = (
        prehit_xy
        + direction * hit_distance
    )

    if abs(hit_target_xy[1]) > striker_y_limit:
        return {
            "valid": False,
            "score": -1e9,
            "reason": "HIT_Y_LIMIT",
        }

    puck_start = np.asarray(
        puck.get_pose().p,
        dtype=float,
    ).copy()

    ok = move_left_striker_segment(
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
        command_speed=hit_speed,
    )

    if not ok:
        return {
            "valid": False,
            "score": -1e9,
            "reason": "HIT_IK",
        }

    # --------------------------------------------------------
    # Forward simulation after strike
    # --------------------------------------------------------

    rollout_time = 2.50

    rollout_steps = max(
        1,
        int(round(
            rollout_time / timestep
        )),
    )

    initial_rollout_p = np.asarray(
        puck.get_pose().p,
        dtype=float,
    ).copy()

    prev_p = initial_rollout_p.copy()

    max_x = float(prev_p[0])
    y_at_max_x = float(prev_p[1])

    min_goal_distance = float("inf")

    goal_event = "TIMEOUT"
    goal_success = False

    y_cross = None
    cross_time = None
    cross_speed = None

    for rollout_i in range(rollout_steps):

        physics_step(scene)

        p = np.asarray(
            puck.get_pose().p,
            dtype=float,
        ).copy()

        if p[0] > max_x:
            max_x = float(p[0])
            y_at_max_x = float(p[1])

        goal_distance = float(
            np.hypot(
                goal_plane_x - p[0],
                p[1],
            )
        )

        min_goal_distance = min(
            min_goal_distance,
            goal_distance,
        )

        # ----------------------------------------------------
        # 第一次从桌内向 +X 穿过 opponent goal plane
        # ----------------------------------------------------
        if (
            prev_p[0] < goal_plane_x
            and p[0] >= goal_plane_x
        ):

            segment_dx = float(
                p[0] - prev_p[0]
            )

            if abs(segment_dx) > 1e-12:

                alpha = float(
                    np.clip(
                        (
                            goal_plane_x
                            - prev_p[0]
                        )
                        / segment_dx,
                        0.0,
                        1.0,
                    )
                )

            else:
                alpha = 1.0

            y_cross = float(
                prev_p[1]
                + alpha
                * (
                    p[1]
                    - prev_p[1]
                )
            )

            cross_time = float(
                (
                    rollout_i
                    + alpha
                )
                * timestep
            )

            cross_v = np.asarray(
                puck_body.get_linear_velocity(),
                dtype=float,
            )

            cross_speed = float(
                np.linalg.norm(
                    cross_v[:2]
                )
            )

            if (
                abs(y_cross)
                <= goal_center_limit
            ):
                goal_success = True
                goal_event = "GOAL"

            else:
                goal_success = False
                goal_event = "END_LINE_MISS"

            # 穿过端线立刻终止，
            # 不再让 puck 在桌子外继续飞。
            break

        v_now = np.asarray(
            puck_body.get_linear_velocity(),
            dtype=float,
        )

        speed_now = float(
            np.linalg.norm(
                v_now[:2]
            )
        )

        elapsed = (
            rollout_i + 1
        ) * timestep

        if (
            elapsed > 0.25
            and speed_now < 0.03
        ):
            goal_event = "STOP"
            break

        prev_p = p


    final_p = np.asarray(
        puck.get_pose().p,
        dtype=float,
    )

    final_v = np.asarray(
        puck_body.get_linear_velocity(),
        dtype=float,
    )

    # --------------------------------------------------------
    # Goal-aware score
    #
    # 所有合法 GOAL 必须严格高于非 GOAL。
    # GOAL 内部优先越靠近中心越好。
    # --------------------------------------------------------

    progress = (
        min(
            max_x,
            goal_plane_x,
        )
        - float(puck_start[0])
    )

    if goal_success:

        normalized_center_error = (
            abs(y_cross)
            / max(
                goal_center_limit,
                1e-6,
            )
        )

        center_quality = max(
            0.0,
            1.0
            - normalized_center_error,
        )

        score = (
            100.0
            + 10.0 * center_quality
            + 0.25 * (
                cross_speed
                if cross_speed is not None
                else 0.0
            )
        )

    elif goal_event == "END_LINE_MISS":

        outside_error = max(
            0.0,
            abs(y_cross)
            - goal_center_limit,
        )

        score = (
            -20.0
            - 20.0 * outside_error
        )

    else:

        score = (
            4.0 * progress
            - 2.0 * min_goal_distance
            - 1.5 * abs(
                y_at_max_x
            )
        )


    return {
        "valid": True,
        "score": float(score),
        "angle_offset_deg": float(
            angle_offset_deg
        ),
        "hit_speed": float(hit_speed),
        "max_x": float(max_x),
        "y_at_max_x": float(y_at_max_x),
        "min_goal_distance": float(
            min_goal_distance
        ),
        "final_x": float(final_p[0]),
        "final_y": float(final_p[1]),
        "final_vx": float(final_v[0]),
        "final_vy": float(final_v[1]),
        "goal_event": goal_event,
        "goal_success": bool(
            goal_success
        ),
        "goal_plane_x": float(
            goal_plane_x
        ),
        "goal_center_limit": float(
            goal_center_limit
        ),
        "y_cross": (
            None
            if y_cross is None
            else float(y_cross)
        ),
        "cross_time": (
            None
            if cross_time is None
            else float(cross_time)
        ),
        "cross_speed": (
            None
            if cross_speed is None
            else float(cross_speed)
        ),

        "tool_follow_max_error": (
            get_left_tool_follow_max_error()
        ),

    }

def main():
    print("\n==============================================")
    print("AUTO RGB EPISODE V1 / RANDOM INBOUND SPAWN")
    print("==============================================\n")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "random inbound episode seed"
        ),
    )
    parser.add_argument("--config", type=Path,
                        default=ROOT / "env_cfg" / "air_hockey_assets.yml")
    parser.add_argument("--check", action="store_true",
                        help="list fixed assets and planned robot mounts without a graphics device")
    parser.add_argument("--duration", type=float, default=180,
                        help="preview duration in seconds (default: 180; use 0 for no limit)")
    args = parser.parse_args()
    rng = np.random.default_rng(
        args.seed
    )
    if args.duration < 0:
        parser.error("--duration must be nonnegative")

    config = load_asset_config(args.config)
    if args.check:
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        actors = populate_static_preview(scene, config, render=False)
        for key, actor in actors.items():
            print(f"{key}: {actor.get_name()} at {actor.get_pose().p}")
        for key, pose in robot_mount_poses(config).items():
            print(f"planned {key} base: {pose.p}")
        return

    from sapien.utils.viewer import Viewer

    configure_air_hockey_physics(config)
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    scene = engine.create_scene(sapien.SceneConfig())
    scene.set_timestep(config["physics"]["timestep_s"])
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [0.7, 0.7, 0.7], shadow=True)
    # ========================================================
    # Validation-only stronger striker clamp
    # ========================================================
    # 原场景 striker 是独立动态刚体，由 WSG 摩擦夹住。
    # 高速 DEFEND 时临时加强夹持，避免因为夹持模型而干扰
    # 我们对 DEFEND 算法本身的验证。
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
        raise RuntimeError("没有找到 puck 的 PhysxRigidDynamicComponent")

    # ========================================================
    # LEFT UR5 CARTESIAN GOALIE
    # ========================================================
    left_robot = assembly["left_robot"]
    left_striker = assembly["left_striker"]

    robot_cfg = config["robot_preview"]
    arm_joint_names = set(robot_cfg["arm_joint_names"])

    left_links = left_robot.get_links()

    left_ee_index = next(
        i for i, link in enumerate(left_links)
        if link.get_name() == robot_cfg["ee_link_name"]
    )

    left_ee = left_links[left_ee_index].entity
    left_pinocchio = left_robot.create_pinocchio_model()
    left_active_joints = left_robot.get_active_joints()

    left_striker_body = (
        configure_left_kinematic_tool(
            left_striker,
            left_ee,
        )
    )
    configure_episode_recorder(
        scene,
        config,
        assembly,
        left_ee,
    )


    arm_qmask = np.asarray(
        [
            1 if joint.name in arm_joint_names else 0
            for joint in left_active_joints
        ],
        dtype=np.int32,
    )

    left_limits = np.asarray(
        left_robot.get_qlimits(),
        dtype=float,
    )

    # ---------------- DEFEND 参数 ----------------
    table_length = float(config["table"]["length_m"])
    table_width = float(config["table"]["width_m"])
    rail_width = float(config["table"]["rail_width_m"])
    puck_radius = float(config["puck"]["diameter_m"]) / 2.0

    # 左侧防守线：
    # 桌面中心为 x=0，左球门在负 x。
    # 先放在左半场约 70% 位置用于算法验证。
    x_defend = -0.70 * (table_length / 2.0)

    # puck 中心实际能到达的 y 边界
    y_limit = table_width / 2.0 - rail_width - puck_radius

    # striker 比 puck 更大，所以中心允许到达的 Y 范围更小
    striker_radius = float(config["striker"]["diameter_m"]) / 2.0
    striker_y_limit = (
        table_width / 2.0
        - rail_width
        - striker_radius
    )

    print("\n=== DEFEND PREDICTOR CONFIG ===")
    print(f"x_defend = {x_defend:+.4f} m")
    print(f"y_limit  = ±{y_limit:.4f} m")
    print("===============================\n")

    for _ in range(config["grasp"]["settle_steps"]):
        physics_step(scene)

    # ========================================================
    # Cartesian goalie initial state
    # ========================================================
    striker_initial_pos = np.asarray(
        left_striker.get_pose().p,
        dtype=float,
    )

    ee_initial_pose = left_ee.pose
    ee_initial_pos = np.asarray(
        ee_initial_pose.p,
        dtype=float,
    )

    # striker 并不一定和 ee_link 原点完全重合。
    # 后面控制 striker 的世界坐标，因此需要补偿这个偏移。
    striker_ee_offset_xy = (
        striker_initial_pos[:2] - ee_initial_pos[:2]
    )

    defend_ee_z = float(ee_initial_pos[2])
    defend_ee_q = np.asarray(
        ee_initial_pose.q,
        dtype=float,
    ).copy()

    desired_striker_y = 0.0
    latest_t_hit = None
    threat_active = False

    # 独立的 Cartesian command trajectory。
    # 关键：command position 不能每次被实际 striker 位置重置。
    cmd_striker_x = float(striker_initial_pos[0])
    cmd_striker_y = float(striker_initial_pos[1])

    cart_vx = 0.0
    cart_vy = 0.0

    ik_success_count = 0
    ik_fail_count = 0

    # ============================================================
    # DEFEND PHASE MACHINE
    # ============================================================
    defend_phase = "DEFEND"

    # HIT snapshot/restore validation
    hit_snapshot_test_done = False
    hit_search_done = False

    # puck + striker 两圆刚好接触时的中心距离
    contact_distance = puck_radius + striker_radius

    # 给离散仿真留 2 mm 接触容差
    contact_threshold = contact_distance + 0.002

    # 分离判据比接触判据稍宽，避免接触边界抖动
    release_threshold = contact_distance + 0.010

    contact_hold_y = 0.0

    # CONTACT_HOLD 不再要求 puck 必须几何分离。
    # 防守把 puck 直接挡停也是合法结果。
    contact_hold_steps = 0
    post_contact_min_hold_s = 0.08
    post_contact_max_hold_s = 0.20
    post_contact_speed_threshold = 0.03

    pre_contact_pos = None
    pre_contact_vel = None

    print(
        f"[PHASE] contact distance = "
        f"{contact_distance*1000:.1f} mm"
    )

    print("\n=== CARTESIAN GOALIE ENABLED ===")
    print(f"defend line x = {x_defend:+.3f} m")
    print(
        "striker-EE offset = "
        f"({striker_ee_offset_xy[0]:+.4f}, "
        f"{striker_ee_offset_xy[1]:+.4f}) m"
    )
    print("控制方式: striker 沿防守线横向移动")
    print("=================================\n")

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=2.1, y=-1.7, z=3.0)
    viewer.set_camera_rpy(r=0, p=-0.75, y=-2.46)
    controller = install_drag_controller(viewer, assembly, config["robot_preview"])
    install_puck_keyboard_controller(viewer, assembly["puck"], config, controller)
    print("UR5 物理预览：点击冰球选中后，按数字键 1-4/6-9 施加方向冲量；按 Q/E 选左/右机械臂。")
    try:
        scene.update_render()
        viewer.render()
        controller.select("left")
        deadline = time.monotonic() + args.duration if args.duration else float("inf")
        last_frame_time = time.monotonic()
        pending_physics_time = 0.0
        timestep = config["physics"]["timestep_s"]
        max_steps = config["interaction"]["max_physics_steps_per_frame"]

        step_count = 0
        print_every_steps = max(1, int(round(0.05 / timestep)))  # 20 Hz
        control_every_steps = max(1, int(round(0.01 / timestep)))  # 100 Hz IK

        print("\n=== PUCK STATE LOGGER STARTED ===")
        print("每 0.05 秒更新一次 puck / DEFEND prediction")
        print("=================================\n")

        # ========================================================
        # AUTO RANDOM INBOUND EPISODE
        #
        # 当前阶段只验证基础流程，因此主动限制在
        # 无 side-wall bounce 的合法来球。
        # ========================================================

        spawn_x = float(
            rng.uniform(
                0.10,
                0.35,
            )
        )

        spawn_y = float(
            rng.uniform(
                -0.25,
                0.10,
            )
        )

        target_y = float(
            rng.uniform(
                -0.25,
                0.25,
            )
        )

        vx_spawn = -float(
            rng.uniform(
                0.78,
                0.95,
            )
        )

        t_to_defend = (
            float(x_defend)
            - spawn_x
        ) / vx_spawn

        if t_to_defend <= 0.0:
            raise RuntimeError(
                "AUTO SPAWN invalid "
                "t_to_defend"
            )

        vy_spawn = (
            target_y
            - spawn_y
        ) / t_to_defend

        # 避免过大的横向速度，
        # 保证基础流程不碰侧墙。
        vy_spawn = float(
            np.clip(
                vy_spawn,
                -0.45,
                0.45,
            )
        )

        puck_pose_now = (
            puck.get_pose()
        )

        spawn_pose = sapien.Pose(
            [
                spawn_x,
                spawn_y,
                float(
                    puck_pose_now.p[2]
                ),
            ],
            np.asarray(
                puck_pose_now.q,
                dtype=float,
            ),
        )

        puck.set_pose(
            spawn_pose
        )

        puck_body.set_linear_velocity(
            np.asarray(
                [
                    vx_spawn,
                    vy_spawn,
                    0.0,
                ],
                dtype=float,
            )
        )

        puck_body.set_angular_velocity(
            np.zeros(
                3,
                dtype=float,
            )
        )

        print("")
        print(
            "======================================"
        )
        print(
            "[AUTO SPAWN]"
        )
        print(
            f"seed={args.seed}"
        )
        print(
            f"puck=({spawn_x:+.4f}, "
            f"{spawn_y:+.4f})"
        )
        print(
            f"velocity=({vx_spawn:+.4f}, "
            f"{vy_spawn:+.4f}) m/s"
        )
        print(
            f"planned defend target_y="
            f"{target_y:+.4f}"
        )
        print(
            f"ideal t_to_defend="
            f"{t_to_defend:.3f}s"
        )
        print(
            "======================================"
        )
        while not viewer.closed and time.monotonic() < deadline:
            now = time.monotonic()
            pending_physics_time = min(pending_physics_time + now - last_frame_time,
                                       timestep * max_steps)
            last_frame_time = now
            steps = int(pending_physics_time / timestep)
            for _ in range(steps):

                # =================================================
                # 100 Hz Cartesian goalie controller
                # =================================================
                if (step_count % control_every_steps == 0 and not hit_search_done):

                    control_dt = (
                        control_every_steps * timestep
                    )

                    striker_pos = np.asarray(
                        left_striker.get_pose().p,
                        dtype=float,
                    )

                    ee_pos_now = np.asarray(
                        left_ee.pose.p,
                        dtype=float,
                    )

                    # ---------------------------------------------
                    # 根据剩余拦截时间自动决定横向速度
                    # ---------------------------------------------
                    if threat_active and latest_t_hit is not None:

                        remaining_y = abs(
                            desired_striker_y - cmd_striker_y
                        )

                        # 理论上按时到达所需的平均速度。
                        # 乘 1.35 留出加减速和机器人跟踪余量。
                        required_speed = (
                            remaining_y
                            / max(latest_t_hit, 0.05)
                        )

                        vmax_y = float(np.clip(
                            required_speed * 1.35 + 0.15,
                            0.75,
                            2.20,
                        ))

                        if latest_t_hit < 0.25:
                            mode = "EMERGENCY"
                            amax_y = 10.0

                        elif latest_t_hit < 0.50:
                            mode = "URGENT"
                            amax_y = 8.0

                        elif latest_t_hit < 1.00:
                            mode = "FAST"
                            amax_y = 6.0

                        else:
                            mode = "NORMAL"
                            amax_y = 4.0

                    else:
                        if defend_phase in ("CONTACT_HOLD", "HIT_READY"):
                            mode = "HOLD"
                            vmax_y = 0.30
                            amax_y = 2.0
                        else:
                            mode = "IDLE"
                            vmax_y = 0.55
                            amax_y = 2.5

                    # ---------------------------------------------
                    # X 始终收敛到防守线
                    # ---------------------------------------------
                    next_striker_x, cart_vx = cartesian_rate_step(
                        position=cmd_striker_x,
                        velocity=cart_vx,
                        target=float(x_defend),
                        vmax=0.70,
                        amax=3.5,
                        dt=control_dt,
                    )

                    cmd_striker_x = next_striker_x

                    # ---------------------------------------------
                    # Y 根据 predictor 的目标做自适应横向运动
                    # ---------------------------------------------
                    next_striker_y, cart_vy = cartesian_rate_step(
                        position=cmd_striker_y,
                        velocity=cart_vy,
                        target=float(desired_striker_y),
                        vmax=vmax_y,
                        amax=amax_y,
                        dt=control_dt,
                    )

                    cmd_striker_y = next_striker_y

                    # 根据初始 striker -> EE 几何偏移，
                    # 把目标 striker 坐标转换成 EE 坐标。
                    target_ee_x = (
                        next_striker_x
                        - striker_ee_offset_xy[0]
                    )

                    target_ee_y = (
                        next_striker_y
                        - striker_ee_offset_xy[1]
                    )

                    target_world = sapien.Pose(
                        [
                            float(target_ee_x),
                            float(target_ee_y),
                            defend_ee_z,
                        ],
                        defend_ee_q,
                    )

                    target_local = (
                        left_robot.get_root_pose().inv()
                        * target_world
                    )

                    q_now = np.asarray(
                        left_robot.get_qpos(),
                        dtype=float,
                    )

                    q_sol, ik_ok, ik_err = (
                        left_pinocchio.compute_inverse_kinematics(
                            left_ee_index,
                            target_local,
                            initial_qpos=q_now,
                            active_qmask=arm_qmask,
                            max_iterations=80,
                        )
                    )

                    valid = (
                        bool(ik_ok)
                        and np.all(np.isfinite(q_sol))
                        and np.all(
                            q_sol >= left_limits[:, 0] - 1e-5
                        )
                        and np.all(
                            q_sol <= left_limits[:, 1] + 1e-5
                        )
                    )

                    if valid:
                        for joint, q_target in zip(
                            left_active_joints,
                            q_sol,
                        ):
                            if joint.name in arm_joint_names:
                                joint.set_drive_target(
                                    float(q_target)
                                )

                        ik_success_count += 1

                    else:
                        ik_fail_count += 1
                        cart_vx = 0.0
                        cart_vy = 0.0

                    # ---------------------------------------------
                    # 监控 striker 有没有从夹爪里滑动
                    # ---------------------------------------------
                    current_offset = (
                        striker_pos[:2] - ee_pos_now[:2]
                    )

                    slip = float(np.linalg.norm(
                        current_offset
                        - striker_ee_offset_xy
                    ))

                    # 每约 0.2 秒显示一次机器人状态
                    if step_count % max(
                        1,
                        int(round(0.20 / timestep))
                    ) == 0:

                        t_text = (
                            f"{latest_t_hit:.3f}"
                            if latest_t_hit is not None
                            else "---"
                        )

                        tracking_error = float(
                            np.hypot(
                                cmd_striker_x - striker_pos[0],
                                cmd_striker_y - striker_pos[1],
                            )
                        )

                        print(
                            f"[GOALIE] "
                            f"{mode:<9} "
                            f"actual_y={striker_pos[1]:+.3f}  "
                            f"cmd_y={cmd_striker_y:+.3f}  "
                            f"target_y={desired_striker_y:+.3f}  "
                            f"vy_cmd={cart_vy:+.2f}m/s  "
                            f"track={tracking_error*1000:.1f}mm  "
                            f"t={t_text}s  "
                            f"slip={slip*1000:.1f}mm",
                            flush=True,
                        )

                # 保存这一 physics step 之前 puck 的真实状态，
                # 用于第一次检测到接触时记录 pre-contact state。
                puck_pos_before_step = np.asarray(
                    puck.get_pose().p,
                    dtype=float,
                ).copy()

                puck_vel_before_step = np.asarray(
                    puck_body.get_linear_velocity(),
                    dtype=float,
                ).copy()

                physics_step(scene)
                step_count += 1

                # =================================================
                # DEFEND CONTACT DETECTOR
                # =================================================
                puck_pos_after = np.asarray(
                    puck.get_pose().p,
                    dtype=float,
                )

                striker_pos_after = np.asarray(
                    left_striker.get_pose().p,
                    dtype=float,
                )

                center_distance = float(np.linalg.norm(
                    puck_pos_after[:2]
                    - striker_pos_after[:2]
                ))

                if defend_phase == "DEFEND":

                    if center_distance <= contact_threshold:

                        defend_phase = "CONTACT_HOLD"
                        episode_set_phase(
                            "CONTACT_HOLD"
                        )

                        contact_hold_y = float(
                            striker_pos_after[1]
                        )

                        desired_striker_y = contact_hold_y
                        threat_active = False
                        latest_t_hit = None

                        pre_contact_pos = puck_pos_before_step.copy()
                        pre_contact_vel = puck_vel_before_step.copy()

                        contact_hold_steps = 0

                        print(
                            "\n======================================="
                        )
                        print("[CONTACT] DEFEND CONTACT DETECTED")
                        print(
                            f"distance = "
                            f"{center_distance*1000:.1f} mm"
                        )
                        print(
                            f"pre puck: "
                            f"x={pre_contact_pos[0]:+.4f}, "
                            f"y={pre_contact_pos[1]:+.4f}, "
                            f"vx={pre_contact_vel[0]:+.4f}, "
                            f"vy={pre_contact_vel[1]:+.4f}"
                        )
                        print(
                            f"holding striker at "
                            f"y={contact_hold_y:+.4f}"
                        )
                        print(
                            "=======================================\n"
                        )

                elif defend_phase == "CONTACT_HOLD":

                    contact_hold_steps += 1
                    hold_time = contact_hold_steps * timestep

                    post_vel_now = np.asarray(
                        puck_body.get_linear_velocity(),
                        dtype=float,
                    )

                    post_speed_xy = float(np.linalg.norm(
                        post_vel_now[:2]
                    ))

                    separated = (
                        center_distance > release_threshold
                    )

                    settled = (
                        post_speed_xy
                        < post_contact_speed_threshold
                    )

                    minimum_hold_finished = (
                        hold_time >= post_contact_min_hold_s
                    )

                    timeout = (
                        hold_time >= post_contact_max_hold_s
                    )

                    if minimum_hold_finished and (
                        separated or settled or timeout
                    ):

                        post_pos = np.asarray(
                            puck.get_pose().p,
                            dtype=float,
                        ).copy()

                        post_vel = np.asarray(
                            puck_body.get_linear_velocity(),
                            dtype=float,
                        ).copy()

                        if separated:
                            ready_reason = "SEPARATED"
                        elif settled:
                            ready_reason = "SETTLED"
                        else:
                            ready_reason = "HOLD_TIMEOUT"

                        defend_phase = "HIT_READY"
                        episode_set_phase(
                            "HIT_READY"
                        )

                        # 保持 striker 在接触位置。
                        # HIT 模块接管之前不自动回中心。
                        contact_hold_y = float(
                            left_striker.get_pose().p[1]
                        )
                        desired_striker_y = contact_hold_y

                        print(
                            "\n======================================="
                        )
                        print(
                            f"[POST-CONTACT] {ready_reason}"
                        )
                        print(
                            f"hold time = {hold_time:.3f} s"
                        )
                        print(
                            f"distance = "
                            f"{center_distance*1000:.1f} mm"
                        )
                        print(
                            f"post puck: "
                            f"x={post_pos[0]:+.4f}, "
                            f"y={post_pos[1]:+.4f}, "
                            f"vx={post_vel[0]:+.4f}, "
                            f"vy={post_vel[1]:+.4f}"
                        )
                        print(
                            f"post speed = "
                            f"{np.linalg.norm(post_vel[:2]):.4f} m/s"
                        )
                        print("[PHASE] DEFEND -> HIT_READY")
                        if not hit_snapshot_test_done:

                            hit_snapshot_test_done = True
                            episode_pause(
                                "SNAPSHOT_TEST"
                            )

                            print("")
                            print("=======================================")
                            print("[HIT] SNAPSHOT TEST START")

                            saved_state = snapshot_hit_state(
                                assembly
                            )

                            saved_puck_p = np.asarray(
                                saved_state["puck"]["pose"].p,
                                dtype=float,
                            ).copy()

                            saved_puck_v = np.asarray(
                                saved_state["puck"]["linear_velocity"],
                                dtype=float,
                            ).copy()

                            print(
                                "[SAVE] puck "
                                f"p=({saved_puck_p[0]:+.5f}, "
                                f"{saved_puck_p[1]:+.5f}) "
                                f"v=({saved_puck_v[0]:+.5f}, "
                                f"{saved_puck_v[1]:+.5f})"
                            )

                            # ----------------------------------------
                            # 故意破坏 puck 状态
                            # ----------------------------------------

                            puck_body.set_linear_velocity(
                                np.asarray(
                                    [0.90, 0.35, 0.0],
                                    dtype=float,
                                )
                            )

                            test_steps = max(
                                1,
                                int(round(
                                    0.15 / timestep
                                )),
                            )

                            for _ in range(test_steps):
                                physics_step(scene)

                            disturbed_p = np.asarray(
                                puck.get_pose().p,
                                dtype=float,
                            ).copy()

                            print(
                                "[PERTURB] puck "
                                f"p=({disturbed_p[0]:+.5f}, "
                                f"{disturbed_p[1]:+.5f})"
                            )

                            # ----------------------------------------
                            # Restore
                            # ----------------------------------------

                            restore_hit_state(
                                assembly,
                                saved_state,
                            )

                            restored_p = np.asarray(
                                puck.get_pose().p,
                                dtype=float,
                            )

                            restored_v = np.asarray(
                                puck_body.get_linear_velocity(),
                                dtype=float,
                            )

                            pos_error = float(
                                np.linalg.norm(
                                    restored_p - saved_puck_p
                                )
                            )

                            vel_error = float(
                                np.linalg.norm(
                                    restored_v - saved_puck_v
                                )
                            )

                            left_striker_error = rigid_position_error(
                                assembly["left_striker"],
                                saved_state["left_striker"],
                            )

                            right_striker_error = rigid_position_error(
                                assembly["right_striker"],
                                saved_state["right_striker"],
                            )

                            left_q_error = float(
                                np.max(
                                    np.abs(
                                        np.asarray(
                                            assembly["left_robot"].get_qpos(),
                                            dtype=float,
                                        )
                                        - saved_state["left_robot"]["qpos"]
                                    )
                                )
                            )

                            right_q_error = float(
                                np.max(
                                    np.abs(
                                        np.asarray(
                                            assembly["right_robot"].get_qpos(),
                                            dtype=float,
                                        )
                                        - saved_state["right_robot"]["qpos"]
                                    )
                                )
                            )

                            print(
                                f"[RESTORE] puck_pos_err="
                                f"{pos_error:.9f} m"
                            )

                            print(
                                f"[RESTORE] puck_vel_err="
                                f"{vel_error:.9f} m/s"
                            )

                            print(
                                f"[RESTORE] "
                                f"left_striker_snapshot_err="
                                f"{left_striker_error:.9f} m "
                                f"(diagnostic only)"
                            )

                            print(
                                f"[RESTORE] right_striker_err="
                                f"{right_striker_error:.9f} m"
                            )

                            print(
                                f"[RESTORE] left_q_err="
                                f"{left_q_error:.9f} rad"
                            )

                            print(
                                f"[RESTORE] right_q_err="
                                f"{right_q_error:.9f} rad"
                            )

                            # left striker 现在是 kinematic EE tool。
                            # 独立 striker snapshot 与当前 tool pose
                            # 不再是 restore 的 source of truth。
                            #
                            # 真正应检查的是：
                            # restored robot EE -> tool follower 是否正确。
                            tool_restore_error = (
                                measure_left_tool_follow_error()
                            )

                            print(
                                f"[RESTORE] "
                                f"tool_follow_err="
                                f"{tool_restore_error*1000:.3f} mm"
                            )

                            restore_ok = (
                                pos_error < 1e-6
                                and vel_error < 1e-6
                                and right_striker_error < 1e-6
                                and left_q_error < 1e-6
                                and right_q_error < 1e-6
                                and tool_restore_error < 0.002
                            )

                            print(
                                "[RESTORE GATE] "
                                + (
                                    "PASS"
                                    if restore_ok
                                    else "FAIL"
                                )
                            )

                            if not restore_ok:
                                raise RuntimeError(
                                    "KINEMATIC RESTORE GATE FAILED: "
                                    f"puck_pos={pos_error:.9e} m, "
                                    f"puck_vel={vel_error:.9e} m/s, "
                                    f"tool={tool_restore_error*1000:.3f} mm, "
                                    f"right_striker="
                                    f"{right_striker_error:.9e} m, "
                                    f"left_q={left_q_error:.9e} rad, "
                                    f"right_q={right_q_error:.9e} rad"
                                )

                            if restore_ok:
                                print(
                                    "[HIT] SNAPSHOT/RESTORE PASS"
                                )
                                episode_resume(
                                    "HIT_PREP"
                                )
                                if not hit_search_done:

                                    hit_search_done = True

                                    print("")
                                    print("=======================================")
                                    print("[HIT SEARCH] START")
                                    print(
                                        "goal target = opponent center"
                                    )
                                    print("")
                                    print("[HIT SEED] PREPARING SEPARATED STATE")

                                    # 回到真实 HIT_READY 状态
                                    restore_hit_state(
                                        assembly,
                                        saved_state,
                                    )

                                    seed_puck_before = np.asarray(
                                        assembly["puck"].get_pose().p,
                                        dtype=float,
                                    ).copy()

                                    seed_striker_before = np.asarray(
                                        left_striker.get_pose().p,
                                        dtype=float,
                                    ).copy()

                                    print(
                                        f"[HIT SEED] before "
                                        f"puck=({seed_puck_before[0]:+.4f},"
                                        f"{seed_puck_before[1]:+.4f}) "
                                        f"striker=({seed_striker_before[0]:+.4f},"
                                        f"{seed_striker_before[1]:+.4f})"
                                    )

                                    # ------------------------------------------------------------
                                    # striker 朝左撤退 100 mm。
                                    # 左机器人自己的半场就在 -X，所以这是远离 puck。
                                    # ------------------------------------------------------------

                                    print("[HIT SEED V6] retreat mode = LATERAL_Y")

                                    # striker 当前在 puck 的上方或下方，
                                    # 沿 Y 朝离开 puck 的方向移动。
                                    away_sign = (
                                        +1.0
                                        if seed_striker_before[1] >= seed_puck_before[1]
                                        else -1.0
                                    )

                                    if away_sign > 0.0:
                                        available_y = (
                                            striker_y_limit
                                            - seed_striker_before[1]
                                            - 0.010
                                        )
                                    else:
                                        available_y = (
                                            seed_striker_before[1]
                                            + striker_y_limit
                                            - 0.010
                                        )

                                    retreat_distance = min(
                                        0.120,
                                        max(0.0, available_y),
                                    )

                                    if retreat_distance < 0.060:
                                        raise RuntimeError(
                                            f"HIT SEED V6: only "
                                            f"{retreat_distance*1000:.1f} mm "
                                            f"lateral retreat available"
                                        )

                                    retract_target = np.asarray(
                                        [
                                            seed_striker_before[0],
                                            seed_striker_before[1]
                                            + away_sign * retreat_distance,
                                        ],
                                        dtype=float,
                                    )

                                    print(
                                        f"[HIT SEED V6] target="
                                        f"({retract_target[0]:+.4f},"
                                        f"{retract_target[1]:+.4f})  "
                                        f"requested_move="
                                        f"{retreat_distance*1000:.1f} mm"
                                    )

                                    seed_ok = move_left_striker_segment(
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
                                        ee_z=defend_ee_z,
                                        ee_q=defend_ee_q,
                                        target_xy=retract_target,
                                        command_speed=0.35,
                                    )

                                    if not seed_ok:
                                        raise RuntimeError(
                                            "HIT SEARCH SEED retract IK failed"
                                        )

                                    # 释放原来 puck-striker 接触约束
                                    seed_settle_steps = max(
                                        1,
                                        int(round(
                                            0.15 / timestep
                                        )),
                                    )

                                    for _ in range(seed_settle_steps):
                                        physics_step(scene)

                                    seed_puck = np.asarray(
                                        assembly["puck"].get_pose().p,
                                        dtype=float,
                                    )

                                    seed_striker = np.asarray(
                                        left_striker.get_pose().p,
                                        dtype=float,
                                    )

                                    seed_velocity = np.asarray(
                                        puck_body.get_linear_velocity(),
                                        dtype=float,
                                    )

                                    seed_distance = float(
                                        np.linalg.norm(
                                            seed_puck[:2]
                                            - seed_striker[:2]
                                        )
                                    )

                                    seed_speed = float(
                                        np.linalg.norm(
                                            seed_velocity[:2]
                                        )
                                    )

                                    seed_striker_motion = float(
                                        np.linalg.norm(
                                            seed_striker[:2]
                                            - seed_striker_before[:2]
                                        )
                                    )

                                    seed_target_error = float(
                                        np.linalg.norm(
                                            seed_striker[:2]
                                            - retract_target[:2]
                                        )
                                    )

                                    print(
                                        f"[HIT SEED V6] striker_moved="
                                        f"{seed_striker_motion*1000:.1f} mm  "
                                        f"target_error="
                                        f"{seed_target_error*1000:.1f} mm"
                                    )

                                    print(
                                        f"[HIT SEED] separated distance="
                                        f"{seed_distance*1000:.1f} mm  "
                                        f"puck_speed={seed_speed:.5f} m/s"
                                    )

                                    # 现在保存真正供所有 candidate 使用的 state
                                    search_seed_state = snapshot_hit_state(
                                        assembly
                                    )
                                    episode_pause(
                                        "HIT_SEARCH"
                                    )

                                    # ------------------------------------------------------------
                                    # SEARCH SEED 独立稳定性检查
                                    # ------------------------------------------------------------

                                    seed_ref_p = np.asarray(
                                        assembly["puck"].get_pose().p,
                                        dtype=float,
                                    ).copy()

                                    seed_test_steps = max(
                                        1,
                                        int(round(
                                            0.15 / timestep
                                        )),
                                    )

                                    for _ in range(seed_test_steps):
                                        physics_step(scene)

                                    seed_after_p = np.asarray(
                                        assembly["puck"].get_pose().p,
                                        dtype=float,
                                    )

                                    seed_after_v = np.asarray(
                                        puck_body.get_linear_velocity(),
                                        dtype=float,
                                    )

                                    seed_drift = float(
                                        np.linalg.norm(
                                            seed_after_p - seed_ref_p
                                        )
                                    )

                                    seed_after_speed = float(
                                        np.linalg.norm(
                                            seed_after_v[:2]
                                        )
                                    )

                                    print(
                                        f"[HIT SEED] stability "
                                        f"drift={seed_drift*1000:.3f} mm  "
                                        f"speed={seed_after_speed:.5f} m/s"
                                    )

                                    # 稳定性测试完成，重新精确恢复
                                    restore_hit_state(
                                        assembly,
                                        search_seed_state,
                                    )

                                    minimum_safe_distance = (
                                        puck_radius
                                        + striker_radius
                                        + 0.020
                                    )

                                    if (
                                        seed_distance > minimum_safe_distance
                                        and seed_drift < 0.002
                                        and seed_after_speed < 0.02
                                    ):
                                        print("[HIT SEED] STABLE")
                                    else:
                                        raise RuntimeError(
                                            "HIT SEARCH SEED is not dynamically stable"
                                        )

                                    print("")

                                    angle_candidates = [
                                        -20.0,
                                        -10.0,
                                        0.0,
                                        +10.0,
                                        +20.0,
                                    ]

                                    speed_candidates = [
        0.90,
        1.20,
        1.50,
        1.80,
    ]

                                    results = []

                                    candidate_id = 0

                                    for angle_offset in angle_candidates:
                                        for hit_speed in speed_candidates:

                                            candidate_id += 1

                                            result = simulate_hit_candidate(
                                                scene=scene,
                                                timestep=timestep,
                                                assembly=assembly,
                                                saved_state=search_seed_state,
                                                left_robot=left_robot,
                                                left_striker=left_striker,
                                                left_pinocchio=left_pinocchio,
                                                left_ee_index=left_ee_index,
                                                left_active_joints=left_active_joints,
                                                arm_joint_names=arm_joint_names,
                                                arm_qmask=arm_qmask,
                                                left_limits=left_limits,
                                                striker_ee_offset_xy=striker_ee_offset_xy,
                                                ee_z=defend_ee_z,
                                                ee_q=defend_ee_q,
                                                puck_body=puck_body,
                                                table_length=table_length,
                                                rail_width=rail_width,
                                                puck_radius=puck_radius,
                                                striker_radius=striker_radius,
                                                striker_y_limit=striker_y_limit,
                                                goal_width=float(
                                                    config["table"]["goal_width_m"]
                                                ),
                                                angle_offset_deg=angle_offset,
                                                hit_speed=hit_speed,
                                            )

                                            results.append(result)

                                            if result["valid"]:

                                                print(
                                                    f"[CANDIDATE {candidate_id:02d}] "
                                                    f"angle={angle_offset:+.0f}deg  "
                                                    f"speed={hit_speed:.2f}m/s  "
                                                    f"score={result['score']:+.3f}  "
                                                    f"max_x={result['max_x']:+.3f}  "
                                                    f"y@max={result['y_at_max_x']:+.3f}  "
                                                    f"goal_d={result['min_goal_distance']:.3f}"
                    f"  event={result['goal_event']}"
                    f"  tool_err="
                    f"{result.get('tool_follow_max_error', 0.0)*1000:.2f}mm"
                                                )

                                                # 每个 candidate 完成后刷新一次界面
                                                scene.update_render()
                                                viewer.render()

                                            else:

                                                print(
                                                    f"[CANDIDATE {candidate_id:02d}] "
                                                    f"angle={angle_offset:+.0f}deg  "
                                                    f"speed={hit_speed:.2f}m/s  "
                                                    f"INVALID "
                                                    f"{result['reason']}"
                                                )

                                    valid_results = [
                                        r for r in results
                                        if r["valid"]
                                    ]

                                    # 搜索结束后恢复原始 HIT_READY 状态
                                    restore_hit_state(
                                        assembly,
                                        saved_state,
                                    )

                                    # ============================================================
                                    # POST-SEARCH RESTORE
                                    # ============================================================

                                    restore_hit_state(
                                        assembly,
                                        search_seed_state,
                                    )

                                    post_ref_p = np.asarray(
                                        assembly["puck"].get_pose().p,
                                        dtype=float,
                                    ).copy()

                                    post_test_steps = max(
                                        1,
                                        int(round(
                                            0.15 / timestep
                                        )),
                                    )

                                    for _ in range(post_test_steps):
                                        physics_step(scene)

                                    post_p = np.asarray(
                                        assembly["puck"].get_pose().p,
                                        dtype=float,
                                    )

                                    post_v = np.asarray(
                                        puck_body.get_linear_velocity(),
                                        dtype=float,
                                    )

                                    post_drift = float(
                                        np.linalg.norm(
                                            post_p - post_ref_p
                                        )
                                    )

                                    post_speed = float(
                                        np.linalg.norm(
                                            post_v[:2]
                                        )
                                    )

                                    print("")
                                    print(
                                        f"[POST-SEARCH RESTORE] "
                                        f"puck_drift={post_drift*1000:.3f} mm  "
                                        f"speed={post_speed:.5f} m/s"
                                    )

                                    if (
                                        post_drift < 0.002
                                        and post_speed < 0.02
                                    ):
                                        print(
                                            "[POST-SEARCH RESTORE] STABLE"
                                        )
                                    else:
                                        print(
                                            "[POST-SEARCH RESTORE] UNSTABLE"
                                        )

                                    # 再恢复一次，让 viewer 停留在稳定 search seed
                                    restore_hit_state(
                                        assembly,
                                        search_seed_state,
                                    )
                                    if valid_results:

                                        best = max(
                                            valid_results,
                                            key=lambda r: r["score"],
                                        )

                                        print("")
                                        print("---------------------------------------")
                                        print("[HIT SEARCH] BEST")
                                        print("")
                                        print("=======================================")
                                        print("[EXECUTE BEST V3] START")

                                        print(
                                            f"[EXECUTE BEST V3] selected "
                                            f"angle={best['angle_offset_deg']:+.1f} deg  "
                                            f"speed={best['hit_speed']:.2f} m/s"
                                        )

                                        episode_resume(
                                            "HIT_EXECUTE"
                                        )
                                        executed = simulate_hit_candidate(
                                            scene=scene,
                                            timestep=timestep,
                                            assembly=assembly,
                                            saved_state=search_seed_state,
                                            left_robot=left_robot,
                                            left_striker=left_striker,
                                            left_pinocchio=left_pinocchio,
                                            left_ee_index=left_ee_index,
                                            left_active_joints=left_active_joints,
                                            arm_joint_names=arm_joint_names,
                                            arm_qmask=arm_qmask,
                                            left_limits=left_limits,
                                            striker_ee_offset_xy=striker_ee_offset_xy,
                                            ee_z=defend_ee_z,
                                            ee_q=defend_ee_q,
                                            puck_body=puck_body,
                                            table_length=table_length,
                                            rail_width=rail_width,
                                            puck_radius=puck_radius,
                                            striker_radius=striker_radius,
                                            striker_y_limit=striker_y_limit,
                                            goal_width=float(
                                                config["table"]["goal_width_m"]
                                            ),
                                            angle_offset_deg=best[
                                                "angle_offset_deg"
                                            ],
                                            hit_speed=best[
                                                "hit_speed"
                                            ],
                                        )
                                        episode_finish(
                                            executed.get(
                                                "goal_event",
                                                "DONE",
                                            )
                                        )

                                        # Let outer viewer loop terminate naturally.
                                        deadline = time.monotonic()

                                        if not executed["valid"]:
                                            raise RuntimeError(
                                                "[EXECUTE BEST V3] FAILED: "
                                                + str(
                                                    executed.get(
                                                        "reason",
                                                        "UNKNOWN",
                                                    )
                                                )
                                            )

                                        dx = abs(
                                            executed["max_x"]
                                            - best["max_x"]
                                        )

                                        dy = abs(
                                            executed["y_at_max_x"]
                                            - best["y_at_max_x"]
                                        )

                                        dgoal = abs(
                                            executed["min_goal_distance"]
                                            - best["min_goal_distance"]
                                        )

                                        dscore = abs(
                                            executed["score"]
                                            - best["score"]
                                        )

                                        print("")
                                        print(
                                            "[EXECUTE BEST V3] SEARCH VS LIVE REPLAY"
                                        )

                                        print(
                                            f"max_x   search={best['max_x']:+.5f} "
                                            f"live={executed['max_x']:+.5f}"
                                        )

                                        print(
                                            f"y@max   search={best['y_at_max_x']:+.5f} "
                                            f"live={executed['y_at_max_x']:+.5f}"
                                        )

                                        print(
                                            f"goal_d  search={best['min_goal_distance']:.5f} "
                                            f"live={executed['min_goal_distance']:.5f}"
                                        )

                                        print(
                                            f"score   search={best['score']:+.5f} "
                                            f"live={executed['score']:+.5f}"
                                        )

                                        print(
                                            f"[EXECUTE BEST V3] replay_error "
                                            f"dx={dx*1000:.2f} mm  "
                                            f"dy={dy*1000:.2f} mm  "
                                            f"dgoal={dgoal*1000:.2f} mm  "
                                            f"dscore={dscore:.4f}"
                                        )

                                        goal_x = (
                                            table_length / 2.0
                                            - rail_width
                                            - puck_radius
                                        )

                                        print("")
                                        print(
                                            f"[EXECUTE BEST V3] "
                                            f"goal_x={goal_x:+.4f}"
                                        )

                                        print(
                                            f"[EXECUTE BEST V3] "
                                            f"max_x={executed['max_x']:+.4f}  "
                                            f"y@max={executed['y_at_max_x']:+.4f}"
                                        )

                                        if executed["max_x"] >= goal_x - 1e-4:
                                            print(
                                                "[EXECUTE BEST V3] GOAL LINE CROSSED"
                                            )
                                        else:
                                            print(
                                                "[EXECUTE BEST V3] GOAL LINE NOT CROSSED"
                                            )

                                        print(
                                            f"[EXECUTE BEST V3] final puck="
                                            f"({executed['final_x']:+.4f}, "
                                            f"{executed['final_y']:+.4f})"
                                        )

                                        print(
                                            f"[EXECUTE BEST V3] final velocity="
                                            f"({executed['final_vx']:+.4f}, "
                                            f"{executed['final_vy']:+.4f})"
                                        )

                                        print("")
                                        print(
                                            f"[GOAL EVENT] event="
                                            f"{executed['goal_event']}"
                                        )

                                        print(
                                            f"[GOAL EVENT] plane_x="
                                            f"{executed['goal_plane_x']:+.4f} m"
                                        )

                                        print(
                                            f"[GOAL EVENT] legal center |y| <= "
                                            f"{executed['goal_center_limit']:.4f} m"
                                        )

                                        if executed["y_cross"] is not None:

                                            print(
                                                f"[GOAL EVENT] y_cross="
                                                f"{executed['y_cross']:+.4f} m"
                                            )

                                            print(
                                                f"[GOAL EVENT] cross_time="
                                                f"{executed['cross_time']:.4f} s"
                                            )

                                        if executed["goal_success"]:
                                            print(
                                                "[GOAL EVENT] GOAL SUCCESS"
                                            )
                                        else:
                                            print(
                                                "[GOAL EVENT] GOAL MISS"
                                            )

                                        print(
                                            f"[TOOL FOLLOW] "
                                            f"max_error="
                                            f"{executed.get('tool_follow_max_error', 0.0)*1000:.2f} mm"
                                        )

                                        print("[EXECUTE BEST V3] LIVE STATE PRESERVED")
                                        print("=======================================")
                                        print("")
                                        print(
                                            f"angle offset = "
                                            f"{best['angle_offset_deg']:+.1f} deg"
                                        )
                                        print(
                                            f"contact speed = "
                                            f"{best['hit_speed']:.2f} m/s"
                                        )
                                        print(
                                            f"score = {best['score']:+.4f}"
                                        )
                                        print(
                                            f"max puck x = "
                                            f"{best['max_x']:+.4f}"
                                        )
                                        print(
                                            f"y at max x = "
                                            f"{best['y_at_max_x']:+.4f}"
                                        )
                                        print(
                                            f"min goal distance = "
                                            f"{best['min_goal_distance']:.4f} m"
                                        )
                                        print(
                                            f"final puck = "
                                            f"({best['final_x']:+.4f}, "
                                            f"{best['final_y']:+.4f})"
                                        )
                                        print(
                                            f"final velocity = "
                                            f"({best['final_vx']:+.4f}, "
                                            f"{best['final_vy']:+.4f})"
                                        )
                                        print("---------------------------------------")

                                    else:

                                        print(
                                            "[HIT SEARCH] "
                                            "NO VALID CANDIDATES"
                                        )

                                    print("=======================================")
                                    print("")
                            else:
                                print("[HIT] SNAPSHOT/RESTORE NEEDS CHECK")

                            print("=======================================")
                            print("")
                        print(
                            "=======================================\n"
                        )

                if step_count % print_every_steps == 0:
                    pos = np.asarray(puck.get_pose().p, dtype=float)
                    vel = np.asarray(puck_body.get_linear_velocity(), dtype=float)

                    print(
                        f"[PUCK] "
                        f"x={pos[0]:+.4f}  "
                        f"y={pos[1]:+.4f}  "
                        f"vx={vel[0]:+.4f}  "
                        f"vy={vel[1]:+.4f}  "
                        f"speed={(vel[0]**2 + vel[1]**2)**0.5:.4f}",
                        flush=True,
                    )

                    prediction = predict_defend_intercept(
                        x=float(pos[0]),
                        y=float(pos[1]),
                        vx=float(vel[0]),
                        vy=float(vel[1]),
                        x_defend=x_defend,
                        y_limit=y_limit,
                    )

                    if defend_phase == "DEFEND":

                        if prediction is not None:
                            t_hit, y_hit = prediction

                            desired_striker_y = float(np.clip(
                                y_hit,
                                -striker_y_limit,
                                striker_y_limit,
                            ))

                            latest_t_hit = float(t_hit)
                            threat_active = True

                            print(
                                f"[DEFEND] "
                                f"t={t_hit:.3f}s  "
                                f"target=({x_defend:+.3f}, "
                                f"{desired_striker_y:+.3f})",
                                flush=True,
                            )

                        else:
                            latest_t_hit = None
                            threat_active = False
                            desired_striker_y = 0.0

                    else:
                        # 接触以后不允许 goalie 立刻回中心。
                        # 固定在发生接触的位置，等待 puck 分离。
                        latest_t_hit = None
                        threat_active = False
                        desired_striker_y = contact_hold_y

            pending_physics_time -= steps * timestep
            scene.update_render()
            viewer.render()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
