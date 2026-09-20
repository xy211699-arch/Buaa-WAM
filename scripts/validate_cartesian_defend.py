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
        scene.step()

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

    cart_vx = 0.0
    cart_vy = 0.0

    ik_success_count = 0
    ik_fail_count = 0

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
                if step_count % control_every_steps == 0:

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

                        if latest_t_hit < 0.30:
                            mode = "EMERGENCY"
                            vmax_y = 1.90
                            amax_y = 8.0

                        elif latest_t_hit < 0.60:
                            mode = "URGENT"
                            vmax_y = 1.60
                            amax_y = 6.0

                        elif latest_t_hit < 1.20:
                            mode = "FAST"
                            vmax_y = 1.20
                            amax_y = 4.5

                        else:
                            mode = "NORMAL"
                            vmax_y = 0.85
                            amax_y = 3.0

                    else:
                        mode = "IDLE"
                        vmax_y = 0.55
                        amax_y = 2.5

                    # ---------------------------------------------
                    # X 始终收敛到防守线
                    # ---------------------------------------------
                    next_striker_x, cart_vx = cartesian_rate_step(
                        position=float(striker_pos[0]),
                        velocity=cart_vx,
                        target=float(x_defend),
                        vmax=0.70,
                        amax=3.5,
                        dt=control_dt,
                    )

                    # ---------------------------------------------
                    # Y 根据 predictor 的目标做自适应横向运动
                    # ---------------------------------------------
                    next_striker_y, cart_vy = cartesian_rate_step(
                        position=float(striker_pos[1]),
                        velocity=cart_vy,
                        target=float(desired_striker_y),
                        vmax=vmax_y,
                        amax=amax_y,
                        dt=control_dt,
                    )

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

                        print(
                            f"[GOALIE] "
                            f"{mode:<9} "
                            f"striker=({striker_pos[0]:+.3f},"
                            f"{striker_pos[1]:+.3f})  "
                            f"target_y={desired_striker_y:+.3f}  "
                            f"vy={cart_vy:+.2f}m/s  "
                            f"t={t_text}s  "
                            f"slip={slip*1000:.1f}mm",
                            flush=True,
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
                        # puck 不再朝左侧防守线运动。
                        # 没有威胁时缓慢回到中央待机位置。
                        latest_t_hit = None
                        threat_active = False
                        desired_striker_y = 0.0

            pending_physics_time -= steps * timestep
            scene.update_render()
            viewer.render()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
