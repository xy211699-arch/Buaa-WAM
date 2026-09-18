"""Checks for physical gripper contact and directional puck input."""

import unittest
import os

import numpy as np
import sapien.core as sapien


class AirHockeyInteractionTest(unittest.TestCase):
    def test_directional_impulse_moves_puck_and_can_hit_disc(self):
        from air_hockey_assets import build_puck, build_striker, build_table, load_asset_config
        from scripts.puck_keyboard_control import apply_directional_impulse
        from pathlib import Path

        cfg = load_asset_config(Path(__file__).resolve().parents[1] / "env_cfg/air_hockey_assets.yml")
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        scene.set_timestep(cfg["physics"]["timestep_s"])
        build_table(scene, cfg, render=False)
        puck = build_puck(scene, cfg, render=False)
        striker = build_striker(scene, cfg, render=False, x=0.25, y=0)
        striker.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).set_kinematic(True)
        apply_directional_impulse(puck, "6", impulse_ns=0.016)
        body = puck.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        scene.step()
        self.assertGreater(body.linear_velocity[0], 1.0)
        for _ in range(260):
            scene.step()
        self.assertLess(body.linear_velocity[0], 0.0)

    def test_keypad_directions_are_normalized_and_keypress_is_single_shot(self):
        from pathlib import Path
        from types import SimpleNamespace
        from air_hockey_assets import build_puck, load_asset_config
        from scripts.puck_keyboard_control import PuckKeyboardController, direction_for_key

        cfg = load_asset_config(Path(__file__).resolve().parents[1] / "env_cfg" / "air_hockey_assets.yml")
        np.testing.assert_allclose(direction_for_key("8"), [0, 1])
        np.testing.assert_allclose(direction_for_key("3"), [2 ** -0.5, -2 ** -0.5])
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        scene.set_timestep(cfg["physics"]["timestep_s"])
        puck = build_puck(scene, cfg, render=False)
        keys = {"8": False}
        controller = PuckKeyboardController(puck, cfg)
        controller.viewer = SimpleNamespace(closed=False, window=SimpleNamespace(
            key_down=lambda key: keys.get(key, False)), selected_entity=puck)
        keys["8"] = True
        controller.after_render()
        scene.step()
        body = puck.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        self.assertAlmostEqual(body.linear_velocity[1], 0.0, delta=1e-5)
        keys["8"] = False
        controller.after_render()
        controller.selected = True
        keys["8"] = True
        controller.after_render()
        scene.step()
        velocity = body.linear_velocity
        self.assertAlmostEqual(velocity[1], cfg["interaction"]["keyboard_impulse_Ns"] /
                               cfg["puck"]["mass_kg"], delta=0.05)
        keys["8"] = False
        controller.after_render()
        keys["8"] = True
        controller.after_render()
        scene.step()
        self.assertAlmostEqual(body.linear_velocity[1],
                               2 * cfg["interaction"]["keyboard_impulse_Ns"] /
                               cfg["puck"]["mass_kg"], delta=0.05)

    @unittest.skipUnless(os.getenv("ROBOTWIN_GPU_TEST") == "1", "requires Vulkan URDF loader")
    def test_real_ur5_finger_force_lifts_disc_without_fixed_joint(self):
        from pathlib import Path
        from air_hockey_assets import configure_air_hockey_physics, load_asset_config
        from scripts.preview_air_hockey_assets import populate_robot_preview

        cfg = load_asset_config(Path(__file__).resolve().parents[1] / "env_cfg" / "air_hockey_assets.yml")
        prior = sapien.physx.get_scene_config()
        threshold = prior.bounce_threshold
        try:
            configure_air_hockey_physics(cfg)
            scene = sapien.Scene()
            scene.set_timestep(cfg["physics"]["timestep_s"])
            assembly = populate_robot_preview(scene, cfg, render=False)
            self.assertNotIn("left_striker_joint", assembly)
            self.assertNotIn("right_striker_joint", assembly)
            for side in ("left", "right"):
                self.assertIsNone(assembly[f"{side}_striker"].find_component_by_type(
                    sapien.physx.PhysxDriveComponent))
            for _ in range(400):
                scene.step()
            for side in ("left", "right"):
                robot = assembly[f"{side}_robot"]
                ee_idx = next(i for i, link in enumerate(robot.get_links())
                              if link.get_name() == "ee_link")
                ee = robot.get_links()[ee_idx].entity.pose
                target = sapien.Pose([ee.p[0], ee.p[1], ee.p[2] + 0.05], ee.q)
                solution, success, _ = robot.create_pinocchio_model().compute_inverse_kinematics(
                    ee_idx, robot.pose.inv() * target, initial_qpos=robot.get_qpos(),
                    active_qmask=[1] * 6 + [0, 0], max_iterations=100)
                self.assertTrue(success)
                for joint, qpos in zip(robot.get_active_joints()[:6], solution[:6]):
                    joint.set_drive_target(float(qpos))
            for _ in range(700):
                scene.step()
            for side in ("left", "right"):
                self.assertGreater(assembly[f"{side}_striker"].pose.p[2],
                                   cfg["table"]["top_height_m"] + 0.035)
        finally:
            prior.bounce_threshold = threshold
            sapien.physx.set_scene_config(prior)

    @unittest.skipUnless(os.getenv("ROBOTWIN_GPU_TEST") == "1", "requires Vulkan URDF loader")
    def test_gripped_striker_rebounds_puck_without_slipping_out(self):
        from pathlib import Path
        from air_hockey_assets import configure_air_hockey_physics, load_asset_config
        from scripts.preview_air_hockey_assets import populate_robot_preview
        from scripts.puck_keyboard_control import apply_directional_impulse

        cfg = load_asset_config(Path(__file__).resolve().parents[1] / "env_cfg" / "air_hockey_assets.yml")
        prior = sapien.physx.get_scene_config()
        threshold = prior.bounce_threshold
        try:
            configure_air_hockey_physics(cfg)
            scene = sapien.Scene()
            scene.set_timestep(cfg["physics"]["timestep_s"])
            assembly = populate_robot_preview(scene, cfg, render=False)
            for _ in range(cfg["grasp"]["settle_steps"]):
                scene.step()
            robot = assembly["right_robot"]
            ee = next(link.entity for link in robot.get_links()
                      if link.get_name() == "ee_link")
            striker = assembly["right_striker"]
            before = (ee.pose.inv() * striker.pose).p.copy()
            puck = assembly["puck"]
            apply_directional_impulse(puck, "6", impulse_ns=0.02)
            for _ in range(650):
                scene.step()
            puck_velocity = puck.find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent).linear_velocity
            self.assertLess(puck_velocity[0], -0.2)
            after = (ee.pose.inv() * striker.pose).p
            self.assertLess(np.linalg.norm(after - before), 0.01)
        finally:
            prior.bounce_threshold = threshold
            sapien.physx.set_scene_config(prior)

    @unittest.skipUnless(os.getenv("ROBOTWIN_GPU_TEST") == "1", "requires Vulkan viewer")
    def test_click_selects_and_highlights_puck_then_arm_selection_clears_it(self):
        from pathlib import Path
        from sapien.utils.viewer import Viewer
        from air_hockey_assets import load_asset_config
        from scripts.preview_air_hockey_assets import populate_robot_preview
        from scripts.puck_keyboard_control import install_puck_keyboard_controller
        from scripts.ur5_reachability import install_drag_controller

        cfg = load_asset_config(Path(__file__).resolve().parents[1] / "env_cfg" / "air_hockey_assets.yml")
        scene = sapien.Scene()
        assembly = populate_robot_preview(scene, cfg)
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.set_camera_xyz(x=2.1, y=-1.7, z=3)
        viewer.set_camera_rpy(r=0, p=-0.75, y=-2.46)
        arm = install_drag_controller(viewer, assembly, cfg["robot_preview"])
        puck = assembly["puck"]
        controller = install_puck_keyboard_controller(viewer, puck, cfg, arm)
        material = puck.find_component_by_type(sapien.render.RenderBodyComponent).render_shapes[0].material
        original = np.asarray(material.base_color).copy()
        try:
            scene.update_render()
            viewer.render()
            arm.select("left")
            scene.update_render()
            viewer.render()
            picture = viewer.window.get_picture("Segmentation")
            ys, xs = np.where((picture[:, :, 1] == puck.per_scene_id)
                              & (picture[:, :, 2] == puck.scene.id))
            self.assertGreater(len(xs), 0)
            self.assertTrue(controller.on_click(viewer, int(np.median(xs)), int(np.median(ys))))
            self.assertIs(viewer.selected_entity, puck)
            self.assertTrue(controller.selected)
            self.assertGreater(material.base_color[1], original[1])
            arm.select("right")
            controller.after_render()
            self.assertFalse(controller.selected)
            np.testing.assert_allclose(material.base_color, original)
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
