import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from world_model.joint_data import future_targets, load_branches
from world_model.joint_model import JointHitModel, masked_xy_loss
from world_model.model import SpatialRGBPuckPredictor


class JointModelTests(unittest.TestCase):
    def test_failed_strike_with_later_collision_keeps_real_future_and_failure_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'manifest.json').write_text(json.dumps({'split': 'train', 'files': ['episode.hdf5']}))
            times = np.arange(1, 1001) * .001
            trajectory = np.column_stack((times, -.1 - times * .3, np.full(1000, .1)))
            names = [f'joint_{index}' for index in range(6)]
            plan = dict(contact_time=.3, prep_time=.23, goal_target_y=0., desired_out_speed=1.35,
                        direction_correction_deg=0., speed_scale=1., command_lead_s=0., attack_mode='DIRECT')
            np.savez_compressed(root / 'execution_0.npz', rgb=np.zeros((9, 8, 16, 3), dtype=np.uint8),
                source_filename='episode.hdf5', plan_json=json.dumps(plan), decision_s=0.,
                actual_trajectory=trajectory, contact=False, actual_contact_s=np.nan, observed_contact_s=.6,
                q=np.zeros(6), dq=np.zeros(6), target=np.zeros(6), joint_names=names)
            config = dict(horizon_steps=20, dt_s=.05, contact_guard_s=.02, image_width=16,
                          image_height=8, xy_scale_m=[1., 1.], arm_joint_names=names)
            row = load_branches(root, ['episode.hdf5'], 'train', config)[0]
            self.assertEqual(float(row['contact']), 0.)
            self.assertFalse(bool(row['expert']))
            self.assertEqual(int(row['mask'].sum()), 20)
            self.assertGreater(float(row['future_xy'].abs().sum()), 0.)
            with self.assertRaises(ValueError):
                load_branches(root, [], 'train', config)

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model = JointHitModel({'context_width': 32, 'context_layers': 1, 'history_frames': 9})
        self.visual = torch.randn(2, 9, 28)
        self.robot = torch.randn(2, 18)

    def test_future_loss_updates_present_encoder_and_detach_prevents_it(self):
        for detached in (False, True):
            self.model.zero_grad(set_to_none=True)
            z = self.model.encode_present(self.visual, self.robot)
            result = self.model.future(z, self.visual[:, -1, 24:26], torch.tensor([.05, .1]), detach_context=detached)
            result['xy'].square().mean().backward()
            grad = self.model.visual_projection.weight.grad
            if detached:
                self.assertIsNone(grad)
            else:
                self.assertGreater(float(grad.abs().sum()), 0.)

    def test_action_loss_updates_present_encoder_without_future_access(self):
        noise = torch.randn(2, 6)
        with patch.object(self.model, 'future', side_effect=AssertionError('Future accessed by policy')):
            action, _ = self.model.policy(self.visual, self.robot, noise)
        action.square().mean().backward()
        self.assertGreater(float(self.model.visual_projection.weight.grad.abs().sum()), 0.)
        self.model.eval()
        first = self.model.policy(self.visual, self.robot, noise)[0]
        with torch.no_grad():
            for parameter in self.model.future_head.parameters():
                parameter.add_(1000.)
        second = self.model.policy(self.visual, self.robot, noise)[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_exact_states_can_cover_contact_but_interpolation_cannot_cross_it(self):
        t = np.array([0., .05, .1, .15]); xy = np.column_stack((t, -t))
        target, mask = future_targets(t, xy, 0., [.05, .075, .1, .2], contacts=[.08])
        np.testing.assert_array_equal(mask, [True, False, True, False])
        np.testing.assert_allclose(target[0], [.05, -.05])
        target, mask = future_targets(t, xy, 0., [.05, .1], upper_s=.1)
        np.testing.assert_array_equal(mask, [True, False])

    def test_failed_motion_can_have_real_future_and_masked_points_have_no_gradient(self):
        t = np.array([.001, .05, .1]); xy = np.array([[.2, .3], [.15, .3], [.1, .3]])
        target, mask = future_targets(t, xy, 0., [.05, .1, .15])
        self.assertTrue(np.any(target[mask] != 0))
        prediction = torch.zeros(1, 3, 2, requires_grad=True)
        masked_xy_loss(prediction, torch.tensor(target)[None], torch.tensor(mask)[None]).backward()
        self.assertEqual(float(prediction.grad[:, 2].abs().sum()), 0.)

    def test_observation_encoder_never_runs_free_flight_head(self):
        vision = SpatialRGBPuckPredictor(np.eye(3).tolist())
        rgb = torch.zeros(1, 9, 3, 36, 64, dtype=torch.uint8)
        with patch.object(vision.temporal, 'forward', side_effect=AssertionError('Forecast invoked')):
            result = vision.encode_observation(rgb)
        self.assertEqual(result['tokens'].shape, (1, 9, 28))
