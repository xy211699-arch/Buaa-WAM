"""Open the air-hockey scene with two interactive UR5-WSG arms.

Run from the repository root with ``python scripts/preview_air_hockey_assets.py``.
Use ``--check`` to inspect the scene without a graphics device.
"""

import argparse
import sys
import time
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "env_cfg" / "air_hockey_assets.yml")
    parser.add_argument("--check", action="store_true",
                        help="list fixed assets and planned robot mounts without a graphics device")
    parser.add_argument("--duration", type=float, default=180,
                        help="preview duration in seconds (default: 180; use 0 for no limit)")
    args = parser.parse_args()
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
    assembly = populate_robot_preview(scene, config)
    puck = assembly["puck"]

    # ============================================================
    # LEFT UR5 AUTO DEFEND
    # ============================================================
    left_robot = assembly["left_robot"]

    ee_link_name = config["robot_preview"]["ee_link_name"]
    arm_joint_names = set(config["robot_preview"]["arm_joint_names"])

    left_links = left_robot.get_links()
    left_ee_index = next(
        i for i, link in enumerate(left_links)
        if link.get_name() == ee_link_name
    )
    left_ee = left_links[left_ee_index].entity

    left_pinocchio = left_robot.create_pinocchio_model()

    left_active_joints = left_robot.get_active_joints()

    arm_qmask = np.asarray(
        [
            1 if joint.name in arm_joint_names else 0
            for joint in left_active_joints
        ],
        dtype=np.int32,
    )

    # 保留当前末端高度和姿态，只让防守器在桌面 XY 上移动
    defend_ee_z = float(left_ee.pose.p[2])
    defend_ee_q = np.asarray(left_ee.pose.q, dtype=float).copy()

    last_defend_target_y = None
    ik_success_count = 0
    ik_fail_count = 0

    # ============================================================
    # SMOOTH DEFEND CONTROL
    # ============================================================
    # 不再把 IK 解瞬间塞给机器人，而是保存为 desired_q，
    # 每个物理 step 只允许有限的关节变化。
    desired_q = np.asarray(left_robot.get_qpos(), dtype=float).copy()
    command_q = desired_q.copy()

    # 第一版故意设得比较保守，先保证 striker 不被甩飞。
    MAX_ARM_SPEED_RAD_S = 0.60
    MAX_ARM_ACCEL_RAD_S2 = 2.0

    command_dq = np.zeros_like(command_q)

    print(
        f"[SMOOTH] max joint speed = {MAX_ARM_SPEED_RAD_S:.2f} rad/s, "
        f"max accel = {MAX_ARM_ACCEL_RAD_S2:.2f} rad/s^2"
    )

    puck_body = puck.find_component_by_type(
        sapien.physx.PhysxRigidDynamicComponent
    )
    if puck_body is None:
        raise RuntimeError("没有找到 puck 的 PhysxRigidDynamicComponent")

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

    striker_radius = float(config["striker"]["diameter_m"]) / 2.0

    # striker 本身比 puck 大，因此其中心不能靠 rail 太近
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
        scene.step()

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
        print_every_steps = max(1, int(round(0.1 / timestep)))  # 10 Hz

        print("\n=== PUCK STATE LOGGER STARTED ===")
        print("每 0.1 秒输出一次 x, y, vx, vy")
        print("=================================\n")

        while not viewer.closed and time.monotonic() < deadline:
            now = time.monotonic()
            pending_physics_time = min(pending_physics_time + now - last_frame_time,
                                       timestep * max_steps)
            last_frame_time = now
            steps = int(pending_physics_time / timestep)
            for _ in range(steps):

                # =================================================
                # Rate-limited joint command
                # =================================================
                # 先根据当前位置到 desired_q 的误差得到期望速度
                q_error = desired_q - command_q

                desired_dq = np.clip(
                    q_error / max(timestep, 1e-6),
                    -MAX_ARM_SPEED_RAD_S,
                    MAX_ARM_SPEED_RAD_S,
                )

                # 再限制加速度，防止速度瞬变
                max_dv = MAX_ARM_ACCEL_RAD_S2 * timestep
                command_dq += np.clip(
                    desired_dq - command_dq,
                    -max_dv,
                    max_dv,
                )

                # 积分得到本 step 的关节目标
                command_q += command_dq * timestep

                # 防止越过最终目标
                crossed = (
                    ((desired_q - command_q) * q_error) < 0
                )
                command_q[crossed] = desired_q[crossed]
                command_dq[crossed] = 0.0

                # 只控制 UR5 arm joints，不碰夹爪
                for idx, joint in enumerate(left_active_joints):
                    if joint.name in arm_joint_names:
                        joint.set_drive_target(
                            float(command_q[idx])
                        )

                scene.step()
                step_count += 1

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

                    if prediction is not None:
                        t_hit, y_hit = prediction

                        print(
                            f"[DEFEND] "
                            f"预计 {t_hit:.3f} s 后穿过防守线  "
                            f"x={x_defend:+.3f}, "
                            f"y={y_hit:+.3f}",
                            flush=True,
                        )

                        # -----------------------------------------
                        # AUTO DEFEND
                        # -----------------------------------------
                        # 不让 striker 中心撞进 rail
                        target_y = float(np.clip(
                            y_hit,
                            -striker_y_limit,
                            striker_y_limit,
                        ))

                        # 只有预测仍在未来时才更新机器人
                        if t_hit > 0.03:
                            target_world = sapien.Pose(
                                [
                                    float(x_defend),
                                    target_y,
                                    defend_ee_z,
                                ],
                                defend_ee_q,
                            )

                            # Pinocchio IK 目标使用机器人 base frame
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
                                    max_iterations=200,
                                )
                            )

                            limits = np.asarray(
                                left_robot.get_qlimits(),
                                dtype=float,
                            )

                            valid = (
                                bool(ik_ok)
                                and np.all(np.isfinite(q_sol))
                                and np.all(q_sol >= limits[:, 0] - 1e-5)
                                and np.all(q_sol <= limits[:, 1] + 1e-5)
                            )

                            if valid:
                                # 这里只更新目标关节角。
                                # 真正发送给关节的命令在每个 physics step
                                # 中经过速度/加速度限制后执行。
                                desired_q[:] = q_sol

                                ik_success_count += 1

                                if (
                                    last_defend_target_y is None
                                    or abs(
                                        target_y
                                        - last_defend_target_y
                                    ) > 0.02
                                ):
                                    print(
                                        f"[AUTO] LEFT UR5 -> "
                                        f"x={x_defend:+.3f}, "
                                        f"y={target_y:+.3f}, "
                                        f"t={t_hit:.3f}s  IK=OK",
                                        flush=True,
                                    )

                                last_defend_target_y = target_y

                            else:
                                ik_fail_count += 1
                                print(
                                    f"[AUTO] IK FAIL "
                                    f"x={x_defend:+.3f}, "
                                    f"y={target_y:+.3f}",
                                    flush=True,
                                )

            pending_physics_time -= steps * timestep
            scene.update_render()
            viewer.render()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
