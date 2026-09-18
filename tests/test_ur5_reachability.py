"""Checks for the interactive UR5 end-effector reachability preview."""

import unittest
import os
from pathlib import Path

import numpy as np


class ReachabilityTest(unittest.TestCase):
    def test_rejects_ik_failure_and_joint_limit_violations(self):
        from scripts.ur5_reachability import valid_ik_solution

        limits = np.array([[-1.0, 1.0], [-0.5, 0.5]])
        self.assertFalse(valid_ik_solution(False, [0, 0], limits))
        self.assertFalse(valid_ik_solution(True, [1.1, 0], limits))
        self.assertFalse(valid_ik_solution(True, [0, float("nan")], limits))
        self.assertTrue(valid_ik_solution(True, [0.4, -0.4], limits))

    @unittest.skipUnless(os.getenv("ROBOTWIN_GPU_TEST") == "1", "requires Vulkan URDF loader")
    def test_goal_side_lateral_targets_use_full_ur5_pan_range(self):
        import sapien.core as sapien

        from air_hockey_assets import load_asset_config
        from scripts.preview_air_hockey_assets import populate_robot_preview
        from scripts.ur5_reachability import valid_ik_solution

        root = Path(__file__).resolve().parents[1]
        config = load_asset_config(root / "env_cfg" / "air_hockey_assets.yml")
        scene = sapien.Scene()
        scene.set_timestep(config["physics"]["timestep_s"])
        assembly = populate_robot_preview(scene, config)
        for side, y in (("left", -0.3), ("right", 0.3)):
            robot = assembly[f"{side}_robot"]
            ee_idx = next(i for i, link in enumerate(robot.get_links())
                          if link.get_name() == "ee_link")
            ee_pose = robot.get_links()[ee_idx].entity.pose
            target = sapien.Pose([ee_pose.p[0], y, ee_pose.p[2]], ee_pose.q)
            solution, success, _ = robot.create_pinocchio_model().compute_inverse_kinematics(
                ee_idx, robot.pose.inv() * target, initial_qpos=robot.get_qpos(),
                active_qmask=[1, 1, 1, 1, 1, 1, 0, 0], max_iterations=100)
            self.assertTrue(valid_ik_solution(success, solution, robot.get_qlimits()), side)

    @unittest.skipUnless(os.getenv("ROBOTWIN_GPU_TEST") == "1", "requires Vulkan viewer")
    def test_gizmo_target_moves_ur5_and_preserves_gripper_joints(self):
        import sapien.core as sapien
        from sapien.utils.viewer import Viewer

        from air_hockey_assets import load_asset_config
        from scripts.preview_air_hockey_assets import populate_robot_preview
        from scripts.ur5_reachability import UR5DragController, install_drag_controller

        root = Path(__file__).resolve().parents[1]
        config = load_asset_config(root / "env_cfg" / "air_hockey_assets.yml")
        scene = sapien.Scene()
        assembly = populate_robot_preview(scene, config)
        viewer = Viewer()
        viewer.set_scene(scene)
        controller = install_drag_controller(viewer, assembly, config["robot_preview"])
        try:
            scene.update_render()
            viewer.render()
            controller.select("left")
            self.assertEqual(controller.transform.move_group_selection,
                             [True, True, True, True, True, True, False, False])
            robot = assembly["left_robot"]
            before = robot.get_qpos().copy()
            target = controller.ee_entities["left"].pose.to_transformation_matrix()
            target[0, 3] += 0.02
            controller.transform.gizmo_matrix = target
            controller.apply_gizmo_target()
            drive_targets = np.array([joint.get_drive_target()
                                      for joint in robot.get_active_joints()])
            self.assertFalse(np.allclose(before[:6], drive_targets[:6]))
            self.assertFalse(np.allclose(before[6:], drive_targets[6:]))
            for _ in range(200):
                scene.step()
            after = robot.get_qpos()
            self.assertFalse(np.allclose(before[:6], after[:6]))
            self.assertTrue(controller.status.startswith("Reachable"))

            far_target = target.copy()
            far_target[0, 3] += 3.0
            controller.transform.gizmo_matrix = far_target
            controller.apply_gizmo_target()
            np.testing.assert_allclose(
                [joint.get_drive_target() for joint in robot.get_active_joints()],
                drive_targets)
            self.assertTrue(controller.status.startswith("Unreachable"))

            controller.select("right")
            right_robot = assembly["right_robot"]
            right_before = right_robot.get_qpos().copy()
            right_target = controller.ee_entities["right"].pose.to_transformation_matrix()
            right_target[0, 3] -= 0.02
            controller.transform.gizmo_matrix = right_target
            controller.apply_gizmo_target()
            right_drive_targets = np.array([joint.get_drive_target()
                                            for joint in right_robot.get_active_joints()])
            self.assertFalse(np.allclose(right_before[:6], right_drive_targets[:6]))
            np.testing.assert_allclose(robot.get_qpos(), after)

            right_y_target = controller.ee_entities["right"].pose.to_transformation_matrix()
            right_y_target[1, 3] = 0.3
            controller.transform.gizmo_matrix = right_y_target
            controller.apply_gizmo_target()
            self.assertTrue(controller.status.startswith("Reachable"))
            for _ in range(350):
                scene.step()
            self.assertAlmostEqual(controller.ee_entities["right"].pose.p[1], 0.3, delta=0.06)
            self.assertGreater(assembly["right_striker"].pose.p[1], 0.24)
        finally:
            viewer.close()

        fresh_viewer = Viewer()
        try:
            self.assertFalse(any(isinstance(plugin, UR5DragController)
                                 for plugin in fresh_viewer.plugins))
        finally:
            fresh_viewer.close()


if __name__ == "__main__":
    unittest.main()
