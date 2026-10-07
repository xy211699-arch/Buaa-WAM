"""Regression checks for the Stage-5 dual-arm continuation."""
import unittest

import numpy as np
import sapien.core as sapien

from scripts import collect_episode_rgb_auto_v1 as collector
from scripts import plan_direct_counterattack_stage5_v1_5 as planner


class DualReturnTests(unittest.TestCase):
    def tearDown(self):
        collector._TOOL_FOLLOWERS.clear()
        collector._LEFT_TOOL_FOLLOWER = None

    def test_overhead_camera_has_finite_orthonormal_pose(self):
        pose = collector._ep_look_at_pose([0, 0, 2.7], [0, 0, 0.74])
        matrix = pose.to_transformation_matrix()
        self.assertTrue(np.isfinite(matrix).all())
        np.testing.assert_allclose(matrix[:3, 0], [0, 0, -1], atol=1e-6)
        np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        # Camera image-right is world +X: preserve the arms' screen identities.
        np.testing.assert_allclose(-matrix[:3, 1], [1, 0, 0], atol=1e-6)

    def test_close_but_noncontact_objects_do_not_count_as_a_hit(self):
        scene = sapien.Scene()
        scene.set_timestep(0.001)
        objects = []
        for name, x, kind in (("puck", 0.0, "dynamic"), ("striker", 0.104, "kinematic")):
            builder = scene.create_actor_builder()
            builder.add_sphere_collision(radius=0.05)
            builder.set_initial_pose(sapien.Pose([x, 0, 1]))
            objects.append(builder.build(name) if kind == "dynamic" else builder.build_kinematic(name))
        scene.step()
        self.assertFalse(planner.has_physical_contact(scene, *objects))
        objects[1].set_pose(sapien.Pose([0.095, 0, 1]))
        scene.step()
        self.assertTrue(planner.has_physical_contact(scene, *objects))

    def test_right_tool_follows_ee_and_restores_without_touching_left(self):
        scene = sapien.Scene()
        pairs = []
        for side, x in (("left", -0.5), ("right", 0.5)):
            ee = scene.create_actor_builder().build_static(side + "_ee")
            ee.set_pose(sapien.Pose([x, 0, 1]))
            builder = scene.create_actor_builder()
            builder.add_sphere_collision(radius=0.02)
            builder.set_initial_pose(sapien.Pose([x, 0.1, 1]))
            tool = builder.build(side + "_tool")
            collector.configure_kinematic_tool(tool, ee, side=side)
            pairs.append((ee, tool))
        pairs[1][0].set_pose(sapien.Pose([0.6, 0.2, 1]))
        collector.restore_tool_from_robot("right")
        np.testing.assert_allclose(pairs[1][1].pose.p, [0.6, 0.3, 1], atol=1e-6)
        np.testing.assert_allclose(pairs[0][1].pose.p, [-0.5, 0.1, 1], atol=1e-6)

    def test_mirrored_post_hit_rollout_continues_after_crossing_goal(self):
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        scene.set_timestep(0.001)
        builder = scene.create_actor_builder()
        builder.add_sphere_collision(radius=0.03)
        builder.set_initial_pose(sapien.Pose([-0.98, 0, 1]))
        puck = builder.build("puck")
        body = puck.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        body.set_linear_velocity([-0.1, 0, 0])
        result = planner.rollout_candidate(
            scene=scene, timestep=0.001, puck=puck, puck_body=body,
            table_length=2.128, rail_width=0.045, puck_radius=0.03, goal_width=0.40,
            contact_time=0.0, contact_x=-0.5, prep_avg_speed=0.0, hit_speed=1.0,
            goal_target_y=0.0, desired_out_speed=0.1, seconds=1.0,
            frame_sign=-1.0, fixed_horizon=True,
        )
        self.assertEqual(result["event"], "GOAL")
        self.assertAlmostEqual(float(puck.pose.p[0]), -1.08, delta=0.001)
        self.assertAlmostEqual(result["final_x"], 1.08, delta=0.001)


if __name__ == "__main__":
    unittest.main()
