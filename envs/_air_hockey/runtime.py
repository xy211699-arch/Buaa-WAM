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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .assets import (  # noqa: E402
    build_puck,
    build_striker,
    build_table,
    configure_air_hockey_physics,
    load_asset_config,
)


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

    # A straight-down camera is parallel to world Z. Use world Y as image-up
    # in that case, keeping world +X (the right arm) on the right of the image.
    if abs(float(np.dot(world_up, forward))) > 0.999:
        world_up = np.asarray([0.0, 1.0, 0.0])

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
        "format": ("robotwin_air_hockey_native_v1" if state.get("frame_sink")
                   else "air_hockey_episode_rgb_v1"),
        "image_storage": ("native_frame_cache_hdf5_video" if state.get("frame_sink")
                          else "rgb_png"),
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
        "camera": state.get("camera_config"),
        "contacts": state.get("contacts", []),
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
    *, frame_sink=None, output_dir=None,
):
    global _EP_REC

    camera_cfg = config.get("episode_camera", {})
    width = int(camera_cfg.get("width", 640))
    height = int(camera_cfg.get("height", 360))
    fps = float(camera_cfg.get("fps", 30.0))
    fovy_deg = float(camera_cfg.get("fovy_deg", 55.0))
    camera_height = float(camera_cfg.get("height_above_table_m", 2.0))

    camera = scene.add_camera(
        "episode_rgb_overhead",
        width,
        height,
        np.deg2rad(fovy_deg),
        0.05,
        10.0,
    )

    pose = _ep_look_at_pose(
        eye=[
            0.0,
            0.0,
            float(config["table"]["top_height_m"]) + camera_height,
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

    if output_dir is not None:
        out_dir = Path(output_dir)

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
        'frame_sink': frame_sink,
        "scene": scene,
        "camera": camera,
        "assembly": assembly,
        "left_ee": left_ee,
        "right_ee": next(link.entity for link in assembly["right_robot"].get_links()
                         if link.get_name() == config["robot_preview"]["ee_link_name"]),
        "puck_body": puck_body,
        "camera_config": {
            "view": "overhead", "height_above_table_m": camera_height,
            "fovy_deg": fovy_deg, "pose": _ep_pose_dict(pose),
        },
        "contacts": [],

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
        f"RGB: {width}x{height} @ {fps:g} FPS (overhead)"
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

    if state is None or state["paused"]:
        return

    if state["phase"] != phase:
        state["phase"] = phase

        if state["started"]:
            print(
                f"[REC] PHASE -> {phase}"
            )


def episode_mark_contact(side):
    """Timestamp a physical strike on the demonstration clock only."""
    state = _EP_REC
    if state is None or not state["started"] or state["paused"] or state["finished"]:
        return
    if not any(event["arm"] == side for event in state["contacts"]):
        state["contacts"].append({"arm": side, "episode_time": float(state["episode_time"])})


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

    if state.get("frame_sink") is None:
        iio.imwrite(state["rgb_dir"] / rgb_name, rgb)

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
        "terminal": bool(state.get("terminal_capture", False)),

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
        "right_tcp": _ep_pose_dict(state["right_ee"].pose),

        "action": {
            "left_joint_drive_target": [
                _ep_joint_drive_target(
                    joint
                )
                for joint in left_joints
            ],
            "right_joint_drive_target": [
                _ep_joint_drive_target(joint) for joint in right_robot.get_active_joints()
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
        "right_striker_pose": _ep_pose_dict(assembly["right_striker"].get_pose()),
    }

    if state.get("frame_sink") is not None:
        record["rgb"] = None  # Images are in the native cache, not duplicate PNGs.
        state["frame_sink"](state, record, rgb)

    state["frames_fp"].write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )

    state["frames_fp"].flush()

    state["frame_index"] += 1
    state["last_capture_time"] = float(state["episode_time"])
    if state.get("frame_sink") is not None:
        state["last_record"] = record
        state["last_rgb"] = rgb

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

    # A scheduled capture may already be exactly at contact + 1 second.
    # Do not create a zero-duration state/action transition at termination.
    state["terminal_capture"] = True
    if abs(state["episode_time"] - state.get("last_capture_time", -1.0)) > 1e-9:
        _ep_capture_frame(force=True)
    elif state.get("frame_sink") is not None:
        # The high-rate capture grid can end between two state-grid samples.
        # Reuse the exact terminal image/state; the sink deduplicates each grid.
        record = dict(state["last_record"], terminal=True)
        state["frame_sink"](state, record, state["last_rgb"])

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
_TOOL_FOLLOWERS = {}


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
    _TOOL_FOLLOWERS.clear()
    return configure_kinematic_tool(left_striker, left_ee, side="left")


def configure_kinematic_tool(left_striker, left_ee, *, side):
    """Attach either striker to its own EE; preserve the legacy left API."""
    global _LEFT_TOOL_FOLLOWER
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")

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

    follower = {
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
    _TOOL_FOLLOWERS[side] = follower
    if side == "left":
        _LEFT_TOOL_FOLLOWER = follower

    print("")
    print(
        f"=== {side.upper()} KINEMATIC STRIKER TOOL ENABLED ==="
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


def restore_tool_from_robot(side):
    state = _TOOL_FOLLOWERS.get(side)
    if state is None:
        return
    pose = state["ee"].pose * state["local_pose"]
    state["entity"].set_pose(_copy_tool_pose(pose))
    state["body"].set_kinematic_target(_copy_tool_pose(pose))
    state["max_error"] = state["last_error"] = 0.0


def get_tool_follow_max_error(side):
    state = _TOOL_FOLLOWERS.get(side)
    return 0.0 if state is None else float(state["max_error"])


def reset_tool_follow_stats(side):
    state = _TOOL_FOLLOWERS.get(side)
    if state is not None:
        state.update(max_error=0.0, last_error=0.0, step_count=0)


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
    right = _TOOL_FOLLOWERS.get("right")
    if right is not None:
        right["body"].set_kinematic_target(right["ee"].pose * right["local_pose"])

    scene.step()

    error = (
        measure_left_tool_follow_error()
    )

    sync_left_tool_target()
    if right is not None:
        expected = right["ee"].pose * right["local_pose"]
        right_error = float(np.linalg.norm(right["entity"].pose.p - expected.p))
        right["last_error"] = right_error
        right["max_error"] = max(right["max_error"], right_error)
        right["step_count"] += 1
        right["body"].set_kinematic_target(expected)

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


def snapshot_robot(robot, *, preserve_drive_targets=False):
    result = {
        "qpos": np.asarray(
            robot.get_qpos(),
            dtype=float,
        ).copy(),

        "qvel": np.asarray(
            robot.get_qvel(),
            dtype=float,
        ).copy(),
    }
    if preserve_drive_targets:
        result["drive_targets"] = [_ep_joint_drive_target(j) for j in robot.get_active_joints()]
    return result


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
    targets = state.get("drive_targets", qpos)
    for joint, q in zip(
        robot.get_active_joints(),
        targets,
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


def snapshot_hit_state(assembly, *, preserve_drive_targets=False):
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
            assembly["left_robot"], preserve_drive_targets=preserve_drive_targets,
        ),

        "right_robot": snapshot_robot(
            assembly["right_robot"], preserve_drive_targets=preserve_drive_targets,
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

    if "right" in _TOOL_FOLLOWERS:
        restore_tool_from_robot("right")
    else:
        restore_rigid(assembly["right_striker"], state["right_striker"])



def reset_runtime():
    """Release scene-owning globals between independent official episodes."""
    global _EP_REC, _LEFT_TOOL_FOLLOWER
    if _EP_REC is not None:
        _ep_close()
    _EP_REC = None
    _LEFT_TOOL_FOLLOWER = None
    _TOOL_FOLLOWERS.clear()
