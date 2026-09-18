"""Physics-only checks for the procedural air hockey assets."""

import unittest
import os
from copy import deepcopy
from pathlib import Path

import sapien.core as sapien

from air_hockey_assets import (
    build_puck,
    build_striker,
    build_table,
    load_asset_config,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "env_cfg" / "air_hockey_assets.yml"


class AirHockeyAssetsTest(unittest.TestCase):
    def setUp(self):
        self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        self.scene.set_timestep(0.001)
        self.config = load_asset_config(CONFIG_PATH)

    def test_builders_create_a_supported_puck_and_grippable_striker(self):
        table = build_table(self.scene, self.config, render=False)
        puck = build_puck(self.scene, self.config, render=False)
        striker = build_striker(self.scene, self.config, render=False)

        self.assertEqual(table.get_name(), "air_hockey_table")
        self.assertEqual(puck.get_name(), "air_hockey_puck")
        self.assertEqual(striker.get_name(), "air_hockey_striker")
        puck_body = puck.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        striker_body = striker.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        self.assertIsNotNone(puck_body)
        self.assertIsNotNone(striker_body)
        self.assertAlmostEqual(puck_body.mass, self.config["puck"]["mass_kg"])
        self.assertAlmostEqual(striker_body.mass, self.config["striker"]["mass_kg"])
        self.assertGreater(striker_body.inertia[2], 0)

    def test_round_assets_rest_flat_on_the_table(self):
        build_table(self.scene, self.config, render=False)
        puck = build_puck(self.scene, self.config, render=False, x=-0.4)
        striker = build_striker(self.scene, self.config, render=False, x=0.4)
        for _ in range(1000):
            self.scene.step()

        table_top = self.config["table"]["top_height_m"]
        puck_center = table_top + self.config["puck"]["thickness_m"] / 2
        self.assertAlmostEqual(puck.get_pose().p[2], puck_center, delta=0.003)
        self.assertAlmostEqual(striker.get_pose().p[2], table_top, delta=0.006)

    def test_low_speed_puck_rebounds_from_side_rail(self):
        from air_hockey_assets import configure_air_hockey_physics

        old_config = sapien.physx.get_scene_config()
        original_threshold = old_config.bounce_threshold
        try:
            configure_air_hockey_physics(self.config)
            scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
            scene.set_timestep(0.001)
            build_table(scene, self.config, render=False)
            puck = build_puck(scene, self.config, render=False, y=0.35)
            body = puck.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            body.linear_velocity = [0, 1, 0]
            for _ in range(250):
                scene.step()
            self.assertLess(body.linear_velocity[1], -0.1)
        finally:
            old_config.bounce_threshold = original_threshold
            sapien.physx.set_scene_config(old_config)

    def test_two_strikers_have_separate_robot_side_names(self):
        left = build_striker(self.scene, self.config, render=False,
                             x=-0.4, y=0.0, name="left_striker")
        right = build_striker(self.scene, self.config, render=False,
                              x=0.4, y=0.0, name="right_striker")
        self.assertEqual((left.get_name(), right.get_name()),
                         ("left_striker", "right_striker"))

    def test_striker_is_a_single_disc_without_handle_geometry(self):
        striker = build_striker(self.scene, self.config, render=False)
        body = striker.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        self.assertEqual(len(body.collision_shapes), 1)
        self.assertNotIn("handle_length_m", self.config["striker"])
        radius = self.config["striker"]["diameter_m"] / 2
        expected_yaw_inertia = self.config["striker"]["mass_kg"] * radius ** 2 / 2
        self.assertAlmostEqual(body.inertia[2], expected_yaw_inertia, places=7)

    def test_preview_contains_only_fixed_table_puck_and_two_strikers(self):
        from scripts.preview_air_hockey_assets import populate_static_preview

        actors = populate_static_preview(self.scene, self.config, render=False)
        self.assertEqual(set(actors), {"table", "puck", "left_striker", "right_striker"})
        self.assertEqual(len(self.scene.get_all_actors()), 4)
        for actor in actors.values():
            self.assertIsNotNone(actor.find_component_by_type(sapien.physx.PhysxRigidStaticComponent))
            self.assertIsNone(actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent))

    def test_robot_mounts_are_level_with_table_and_face_goal_centers(self):
        from scripts.preview_air_hockey_assets import robot_mount_poses

        config = deepcopy(self.config)
        config["table"]["length_m"] = 2.0
        config["table"]["top_height_m"] = 0.8
        config["robot_preview"] = {"base_offset_from_end_m": 0.2}
        poses = robot_mount_poses(config)
        self.assertEqual(set(poses), {"left_robot", "right_robot"})
        self.assertAlmostEqual(poses["left_robot"].p[0], -1.2)
        self.assertAlmostEqual(poses["right_robot"].p[0], 1.2)
        for pose in poses.values():
            self.assertAlmostEqual(pose.p[1], 0)
            self.assertAlmostEqual(pose.p[2], 0.8)
        self.assertAlmostEqual(poses["left_robot"].q[0], 1)
        self.assertAlmostEqual(abs(poses["right_robot"].q[3]), 1)

    @unittest.skipUnless(os.getenv("ROBOTWIN_GPU_TEST") == "1", "requires Vulkan renderer")
    def test_robot_preview_loads_two_ur5_wsg_models_on_supports(self):
        from scripts.preview_air_hockey_assets import populate_robot_preview

        scene = sapien.Scene()
        result = populate_robot_preview(scene, self.config)
        self.assertEqual(set(result), {"table", "puck", "left_striker", "right_striker",
                                       "left_stand", "right_stand", "left_robot", "right_robot"})
        self.assertIsNotNone(result["puck"].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent))
        self.assertEqual(len(scene.get_all_articulations()), 2)
        for side in ("left", "right"):
            stand = result[f"{side}_stand"]
            robot = result[f"{side}_robot"]
            self.assertIsNotNone(stand.find_component_by_type(sapien.physx.PhysxRigidStaticComponent))
            self.assertEqual(len(robot.get_qpos()), 8)
            self.assertAlmostEqual(robot.get_root_pose().p[2], self.config["table"]["top_height_m"])
            joint_names = [joint.name for joint in robot.get_active_joints()]
            self.assertEqual(joint_names[:6], ["shoulder_pan_joint", "shoulder_lift_joint",
                                               "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                                               "wrist_3_joint"])
            self.assertIn("ee_link", [link.get_name() for link in robot.get_links()])
            fingers = {link.get_name(): link.entity.pose.p for link in robot.get_links()
                       if link.get_name() in ("finger_left", "finger_right")}
            separation = fingers["finger_right"] - fingers["finger_left"]
            self.assertGreater(abs(separation[1]), 0.09)
            self.assertLess(abs(separation[0]), 0.01)
            self.assertIsNotNone(result[f"{side}_striker"].find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent))
            limits = robot.get_qlimits()
            qpos = robot.get_qpos()
            self.assertTrue(((qpos >= limits[:, 0]) & (qpos <= limits[:, 1])).all())
            ee = next(link.entity for link in robot.get_links() if link.get_name() == "ee_link")
            self.assertAlmostEqual(ee.pose.p[0], -0.85 if side == "left" else 0.85, delta=0.01)
            self.assertAlmostEqual(ee.pose.p[1], 0, delta=0.01)
            self.assertAlmostEqual(ee.pose.p[2], self.config["table"]["top_height_m"] + 0.16,
                                   delta=0.01)


if __name__ == "__main__":
    unittest.main()
