import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from world_model.pipeline import rgb_reference, execution_candidates
from world_model.policy import BehaviorPolicy, decode_intention, expert_intention


class BehaviorPolicyTests(unittest.TestCase):
    def test_bc_selection_retains_base_action_and_rejects_missing_base(self):
        ranked = [{'candidate_index': 4, 'score': 100.}, {'candidate_index': 0, 'score': 0.}]
        self.assertEqual(execution_candidates(ranked, 'bc'), [ranked[1]])
        self.assertEqual(execution_candidates(ranked[:1], 'bc'), [])
        self.assertEqual(execution_candidates(ranked, 'world-model'), ranked)
        self.assertEqual(ranked[0]['candidate_index'], 4)
        with self.assertRaises(ValueError):
            execution_candidates(ranked, 'unknown')

    def test_dry_planning_never_executes_motion_or_reads_puck_truth(self):
        from envs._air_hockey import expert
        reference = {'t': np.array([0., 1.]),
                     'xy': np.array([[-.2, 0.], [-.8, 0.]]),
                     'vxy': np.array([[-.6, 0.], [-.6, 0.]])}
        context = dict(scene=None, timestep=.001, assembly={'puck': object()},
            initial_state={}, left_robot=SimpleNamespace(get_qpos=lambda: np.zeros(8)),
            left_striker=SimpleNamespace(get_pose=lambda: SimpleNamespace(p=np.array([-.65, 0., .77]))),
            left_pinocchio=None, left_ee_index=0, left_active_joints=[],
            arm_joint_names=[], arm_qmask=None, left_limits=None,
            striker_ee_offset_xy=None, ee_z=.9, ee_q=None, puck_body=None,
            table_length=2.128, table_width=1.218, rail_width=.045,
            puck_radius=.03165, striker_radius=.04815, striker_y_limit=.515,
            goal_width=.4, serve={}, puck_reference=reference,
            contact_time=.5, goal_target_y=0., desired_out_speed=1.35)
        with patch.object(expert, 'restore_hit_state'), patch.object(
                expert, 'solve_striker_xy_ik', return_value=(True, np.zeros(8))), patch.object(
                expert, 'timed_cartesian_segment', side_effect=AssertionError('Motion during planning')):
            plan = expert.evaluate_candidate(**context, plan_only=True)
        self.assertTrue(plan['valid'])
        self.assertEqual(plan['reason'], 'PLAN_ONLY')
        self.assertAlmostEqual(plan['contact_x'], -.5)
        self.assertNotIn('physical_contact_time', plan)

        context['puck_body'] = SimpleNamespace(get_linear_velocity=lambda: np.zeros(3))
        context['after_strike_handler'] = lambda result: self.fail('Right arm triggered without physical contact')
        prep = {'ok': True, 'contact': False, 'elapsed_seconds': .3}
        strike = {'ok': True, 'contact': True, 'elapsed_seconds': .12,
                  'physical_contact_local_time': None, 'min_distance': .08}
        with patch.object(expert, 'restore_hit_state'), patch.object(
                expert, 'solve_striker_xy_ik', return_value=(True, np.zeros(8))), patch.object(
                expert, 'timed_cartesian_segment', side_effect=[prep, strike]):
            actual = expert.evaluate_candidate(**context)
        self.assertFalse(actual['valid'])
        self.assertEqual(actual['reason'], 'NO_PHYSICAL_CONTACT')
        self.assertIsNone(actual['physical_contact_time'])

    def test_labels_ignore_outcomes_and_decoding_bounds_commands(self):
        plan = dict(contact_time=2., goal_target_y=.1, desired_out_speed=1.35,
                    direction_correction_deg=6., speed_scale=1., command_lead_s=.02)
        expected, _ = expert_intention(plan, 1.5)
        plan.update(actual_contact_time=-99., score=999., incoming_vx=999.)
        actual, _ = expert_intention(plan, 1.5)
        np.testing.assert_array_equal(actual, expected)
        decoded = decode_intention([10., 1., 9., 9., 9., 9.], -1.)
        self.assertEqual(decoded['contact_time'], 1.)
        self.assertEqual(decoded['attack_mode'], 'DIRECT')
        self.assertEqual(decoded['command_lead_s'], .1)
        with self.assertRaises(ValueError):
            decode_intention([np.nan] * 6, 0.)

    def test_reference_uses_only_forecast_and_its_relative_clock(self):
        history = np.zeros((9, 2))
        history[-1] = [0., .1]
        t = np.arange(1, 21) * .05
        forecast = np.column_stack((-t, np.full(20, .1)))
        reference = rgb_reference(np.concatenate((history.ravel(), forecast.ravel())),
                                  {'history_frames': 9, 'xy_scale_m': [1., 1.], 'dt_s': .05})
        np.testing.assert_allclose(reference['vxy'], np.tile([-1., 0.], (21, 1)), atol=1e-12)
        self.assertEqual(reference['t'][0], 0.)
        self.assertEqual(reference['t'][-1], 1.)

    def test_policy_can_learn_from_forecast_features(self):
        torch.manual_seed(42)
        model = BehaviorPolicy()
        features = torch.randn(4, 58)
        robot = torch.randn(4, 18)
        values, bank = model(features, robot)
        (values.square().mean() + bank.square().mean()).backward()
        self.assertEqual(values.shape, (4, 6))
        self.assertTrue(torch.isfinite(model.net[0].weight.grad).all())
        self.assertGreater(float(model.net[0].weight.grad.abs().sum()), 0.)
