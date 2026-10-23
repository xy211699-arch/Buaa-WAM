"""Actual PhysX environment, deliberately independent of the scripted expert.

The inactive arm holds its initial joint drive targets throughout a trial.
Only scoring and preflight can read puck truth. The policy sees overhead RGB
and robot proprioception; it cannot invoke snapshots or candidate searches.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .metrics import FrameState
from .schema import BenchmarkConfig, Geometry


def _imports():
    try:
        import sapien.core as sapien
        from envs._air_hockey import assets, runtime
    except ImportError as error:
        raise RuntimeError("SAPIEN/RoboTwin runtime is required only for prepare/run; use the project's simulation environment") from error
    return sapien, assets, runtime


def _contact(scene, a, b):
    names = {a.get_name(), b.get_name()}
    for contact in scene.get_contacts():
        if {body.entity.name for body in contact.bodies} == names:
            if any(np.linalg.norm(point.impulse) > 1e-9 for point in contact.points):
                return True
    return False


class PhysXReference:
    def __init__(self, assets: dict, config: BenchmarkConfig):
        self.sapien, self.assets_module, _ = _imports()
        self.assets, self.config = copy.deepcopy(assets), config
        self.geometry = Geometry.from_assets(assets)
        self._old_threshold = float(self.sapien.physx.get_scene_config().bounce_threshold)
        self.assets_module.configure_air_hockey_physics(self.assets)

    def rollout(self, case: dict, *, perturbed: bool = False) -> dict:
        s = self.sapien
        scene = s.Scene([s.physx.PhysxCpuSystem()])
        scene.set_timestep(self.config.physics_dt_s)
        self.assets_module.build_table(scene, self.assets, render=False)
        puck = self.assets_module.build_puck(scene, self.assets, render=False)
        body = puck.find_component_by_type(s.physx.PhysxRigidDynamicComponent)
        for _ in range(int(self.assets.get("grasp", {}).get("settle_steps", 400))):
            scene.step()
        puck.set_pose(s.Pose([*case["position_m"], float(puck.pose.p[2])]))
        body.set_linear_velocity([*case["velocity_m_s"], 0.0])
        body.set_angular_velocity([0.0, 0.0, case["spin_rad_s"]])
        dt, sign = self.config.physics_dt_s, self.config.direction
        last_p = np.asarray(puck.pose.p[:2], dtype=float).copy()
        last_v = np.asarray(body.linear_velocity[:2], dtype=float).copy()
        bounces, applied = 0, False
        for step in range(1, math.ceil(self.config.episode_timeout_s / dt) + 1):
            before = (step - 1) * dt
            pt = case.get("perturb_time_s")
            if perturbed and pt is not None and not applied and before + 1e-9 >= pt:
                velocity = np.asarray(body.linear_velocity, dtype=float).copy()
                velocity[:2] += case["delta_velocity_m_s"]
                body.set_linear_velocity(velocity)
                applied = True
            scene.step()
            p = np.asarray(puck.pose.p[:2], dtype=float).copy()
            v = np.asarray(body.linear_velocity[:2], dtype=float).copy()
            if last_v[1] * v[1] < -1e-8 and abs(p[1]) > self.geometry.y_limit_m - 0.04:
                bounces += 1
            x0, x1 = sign * last_p[0], sign * p[0]
            line = self.config.intercept_line_x_m
            if x0 > line >= x1:
                fraction = (line - x0) / (x1 - x0)
                ttc = before + fraction * dt
                return {"valid": True, "backend": "sapien-physx", "ttc_s": ttc,
                        "side_bounces": bounces, "intercept_y_m": float(last_p[1] + fraction * (p[1] - last_p[1])),
                        "intercept_speed_m_s": float(np.linalg.norm(v)), "perturbation_applied": applied}
            if abs(p[0]) > self.geometry.x_limit_m + 0.15 or abs(p[1]) > self.geometry.y_limit_m + 0.10:
                break
            last_p, last_v = p, v
        return {"valid": False, "backend": "sapien-physx", "reason": "no_intercept_crossing"}

    def close(self):
        scene_config = self.sapien.physx.get_scene_config()
        scene_config.bounce_threshold = self._old_threshold
        self.sapien.physx.set_scene_config(scene_config)


class SapienBackend:
    def __init__(self, assets: dict, config: BenchmarkConfig):
        self.sapien, self.assets_module, self.runtime = _imports()
        self.assets, self.config = copy.deepcopy(assets), config
        self.geometry = Geometry.from_assets(assets)
        self.y_limit_m = self.geometry.y_limit_m
        self._old_threshold = float(self.sapien.physx.get_scene_config().bounce_threshold)
        self.engine, self.renderer, self.scene = None, None, None

    def reset(self, case: dict):
        self.runtime.reset_runtime()
        self.assets_module.configure_air_hockey_physics(self.assets)
        s = self.sapien
        if self.engine is None:
            self.engine = s.Engine()
            self.renderer = s.SapienRenderer()
            self.engine.set_renderer(self.renderer)
        scene_config = s.SceneConfig()
        scene_config.bounce_threshold = self.assets["physics"]["bounce_threshold_m_s"]
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(self.config.physics_dt_s)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, .5, -1], [.7, .7, .7], shadow=True)
        # Match the established expert scene's actual grasp settings.
        scene_assets = copy.deepcopy(self.assets)
        scene_assets["grasp"].update(finger_force_limit_N=150.0, finger_drive_stiffness_N_m=2200.0,
                                     finger_drive_damping_Ns_m=180.0, finger_static_friction=4.0,
                                     finger_dynamic_friction=3.0)
        self.assembly = self.runtime.populate_robot_preview(self.scene, scene_assets)
        self.arm_indices, self.finger_indices, self.ee, self.arm_joint_names = {}, {}, {}, {}
        for side in ("left", "right"):
            robot = self.assembly[side + "_robot"]
            joints = robot.get_active_joints()
            arm_names = self.assets["robot_preview"]["arm_joint_names"]
            self.arm_indices[side] = [next(i for i, j in enumerate(joints) if j.name == n) for n in arm_names]
            self.finger_indices[side] = [i for i in range(len(joints)) if i not in self.arm_indices[side]]
            if len(self.finger_indices[side]) != 2:
                raise ValueError("benchmark expects the actual UR5-WSG 6+2 joint layout")
            self.arm_joint_names[side] = list(arm_names)
            self.ee[side] = next(link.entity for link in robot.get_links()
                                 if link.get_name() == self.assets["robot_preview"]["ee_link_name"])
            self.runtime.configure_kinematic_tool(self.assembly[side + "_striker"], self.ee[side], side=side)
        for _ in range(int(self.assets["grasp"]["settle_steps"])):
            self.runtime.physics_step(self.scene)
        self.puck = self.assembly["puck"]
        self.body = self.puck.find_component_by_type(s.physx.PhysxRigidDynamicComponent)
        z = float(self.puck.pose.p[2])
        self.puck.set_pose(s.Pose([*case["position_m"], z]))
        self.body.set_linear_velocity([*case["velocity_m_s"], 0.0])
        self.body.set_angular_velocity([0.0, 0.0, case["spin_rad_s"]])
        self.robot = self.assembly[self.config.controlled_side + "_robot"]
        self.indices = self.arm_indices[self.config.controlled_side]
        self.limits = np.asarray(self.robot.get_qlimits(), dtype=float)[self.indices]
        self.t = 0.0
        self.last_command = np.asarray(self.robot.get_qpos(), dtype=float)[self.indices].copy()
        self.last_command_time = -1.0 / self.config.action_hz
        self._violations = set()
        self._robot_entities = {link.entity.name for link in self.robot.get_links()}
        self.camera = self.scene.add_camera("benchmark_overhead", self.config.camera_width,
                                            self.config.camera_height, math.radians(self.config.camera_fovy_deg), 0.01, 100)
        top = self.geometry.table_height_m
        self.camera.set_pose(self.runtime._ep_look_at_pose([0, 0, top + self.config.camera_height_m], [0, 0, top]))
        # No evaluation run should begin with a physical overlap at the held striker.
        combined_radius = self.geometry.puck_radius_m + self.assets["striker"]["diameter_m"] / 2
        for side in ("left", "right"):
            if np.linalg.norm(np.asarray(self.puck.pose.p[:2]) - self.assembly[side + "_striker"].pose.p[:2]) <= combined_radius:
                raise ValueError("prepared serve overlaps a held striker; fix the suite before evaluating any method")

    def initial_command(self):
        return self.last_command.copy()

    def step(self):
        self.runtime.physics_step(self.scene)
        self.t += self.config.physics_dt_s

    def apply(self, command):
        command = np.asarray(command, dtype=float)
        violations = set()
        if command.shape != (6,) or not np.isfinite(command).all():
            violations.add("invalid_command")
        else:
            if np.any(command < self.limits[:, 0] - 1e-6) or np.any(command > self.limits[:, 1] + 1e-6):
                violations.add("command_joint_position")
            interval = max(self.t - self.last_command_time, 1.0 / self.config.action_hz)
            if np.any(abs(command - self.last_command) / interval > np.asarray(self.config.max_command_speed_rad_s) + 1e-6):
                violations.add("command_joint_velocity")
        if not violations:
            joints = self.robot.get_active_joints()
            for index, target in zip(self.indices, command):
                joints[index].set_drive_target(float(target))
            self.last_command = command.copy()
            self.last_command_time = self.t
        self._violations.update(violations)
        return tuple(sorted(violations))

    def perturb(self, delta):
        velocity = np.asarray(self.body.linear_velocity, dtype=float).copy()
        velocity[:2] += delta
        self.body.set_linear_velocity(velocity)

    def frame(self):
        side = self.config.controlled_side
        q = np.asarray(self.robot.get_qpos(), dtype=float)[self.indices]
        dq = np.asarray(self.robot.get_qvel(), dtype=float)[self.indices]
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            self._violations.add("nonfinite_robot_state")
        if np.any(q < self.limits[:, 0] - 1e-6) or np.any(q > self.limits[:, 1] + 1e-6):
            self._violations.add("joint_position")
        if np.any(abs(dq) > np.asarray(self.config.max_command_speed_rad_s) + 1e-6):
            self._violations.add("joint_velocity")
        striker_p = np.asarray(self.assembly[side + "_striker"].pose.p, dtype=float)
        x, y = self.config.direction * striker_p[0], striker_p[1]
        if not self.config.workspace_x_m[0] <= x <= self.config.workspace_x_m[1] or not self.config.workspace_y_m[0] <= y <= self.config.workspace_y_m[1]:
            self._violations.add("striker_workspace")
        if abs(striker_p[2] - self.geometry.table_height_m) > self.config.striker_height_tolerance_m:
            self._violations.add("striker_height")
        for contact in self.scene.get_contacts():
            names = {body.entity.name for body in contact.bodies}
            if "air_hockey_table" in names and names & self._robot_entities:
                if any(np.linalg.norm(point.impulse) > 1e-9 for point in contact.points):
                    self._violations.add("robot_table_collision")
        position = np.asarray(self.puck.pose.p, dtype=float)
        other = "right" if side == "left" else "left"
        escaped = (abs(position[1]) > self.geometry.y_limit_m + .10
                   or abs(position[0]) > self.geometry.x_limit_m + .20
                   or abs(position[2] - self.geometry.table_height_m - self.geometry.puck_height_m / 2) > .10)
        return FrameState(float(self.t), tuple(float(v) for v in position[:2]),
                          tuple(float(v) for v in self.body.linear_velocity[:2]),
                          contact=_contact(self.scene, self.puck, self.assembly[side + "_striker"]),
                          violations=tuple(sorted(self._violations)),
                          opponent_contact=_contact(self.scene, self.puck, self.assembly[other + "_striker"]),
                          out_of_bounds=bool(escaped))

    def observe(self):
        self.scene.update_render()
        self.camera.take_picture()
        rgb = np.clip(self.camera.get_picture("Color")[..., :3] * 255, 0, 255).astype(np.uint8)
        state = {}
        for side in ("left", "right"):
            robot = self.assembly[side + "_robot"]
            q, dq = np.asarray(robot.get_qpos()), np.asarray(robot.get_qvel())
            state[side + "_arm_joint_state"] = q[self.arm_indices[side]].astype(np.float32)
            state[side + "_arm_joint_velocity"] = dq[self.arm_indices[side]].astype(np.float32)
            state[side + "_ee_joint_state"] = q[self.finger_indices[side]].astype(np.float32)
            pose = self.ee[side].pose
            state[side + "_ee_pose"] = np.concatenate([pose.p, pose.q]).astype(np.float32)
            state[side + "_tcp_pose"] = state[side + "_ee_pose"].copy()
        return {"timestamp_s": float(self.t), "state": state,
                "vision": {"cam_head": {"color": rgb,
                           "intrinsic_matrix": self.camera.get_intrinsic_matrix(),
                           "extrinsics_matrix": self.camera.get_model_matrix()}}}

    def close(self):
        self.runtime.reset_runtime()
        self.scene = None
        self.engine = self.renderer = None
        scene_config = self.sapien.physx.get_scene_config()
        scene_config.bounce_threshold = self._old_threshold
        self.sapien.physx.set_scene_config(scene_config)
