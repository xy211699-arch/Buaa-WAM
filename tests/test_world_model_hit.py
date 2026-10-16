"""Leakage, contact-boundary, and action conditioning regression checks."""

import unittest
import json
from pathlib import Path
import tempfile
import h5py
import numpy as np
import torch

from world_model.hit_data import HitRGBDataset, counterfactual_samples, planned_action, posthit_targets
from world_model.hit_model import HitPredictor, hit_loss
from world_model.data import camera_plane_projection
from tests import test_world_model_rgb


class HitModelTests(unittest.TestCase):
    def test_counterfactual_split_cannot_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({'split': 'test', 'files': ['heldout.hdf5']}))
            with self.assertRaises(ValueError):
                counterfactual_samples(root, 'train', ['train.hdf5'], {})

    def test_dataset_histories_precede_fast_stroke_and_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.hdf5'
            test_world_model_rgb.WorldModelTests.make_episode(path)
            with h5py.File(path, 'a') as file:
                plan = {'arm': 'left', 'coordinate_frame': 'world', 'contact_time': 1.15,
                        'contact_x': -.6, 'contact_y': .1, 'nominal_theta_deg': 20.,
                        'direction_correction_deg': 5., 'hit_speed': 1.7, 'prep_time': .9}
                file.create_dataset('episode/expert_summary_json',
                                    data=json.dumps({'final_best': plan}))
                _, calibration = camera_plane_projection(file, .75, [1., 1.])
            config = {'dt_s': .05, 'history_frames': 9, 'horizon_steps': 20,
                      'contact_guard_s': .02, 'decision_offsets_s': [0., .05],
                      'xy_scale_m': [1., 1.], 'image_width': 32, 'image_height': 24,
                      'puck_plane_z_m': .75, 'image_to_world': calibration.tolist()}
            dataset = HitRGBDataset(directory, ['sample.hdf5'], config)
            self.assertEqual(len(dataset), 2)
            for sample in dataset:
                self.assertLessEqual(sample['last_rgb_s'], .88)
                self.assertLess(sample['last_rgb_s'], sample['actual_contact_s'])
                self.assertEqual(sample['rgb'].shape, (9, 3, 24, 32))
                self.assertFalse(sample['mask'][-1])

    def test_command_whitelist_ignores_future_outcomes(self):
        plan = {'arm': 'left', 'coordinate_frame': 'world', 'contact_time': 1.2,
                'contact_x': -.6, 'contact_y': .1, 'nominal_theta_deg': 20.,
                'direction_correction_deg': 5., 'hit_speed': 1.7}
        original = planned_action(plan, 1., [1., 1.])
        plan.update(physical_contact_time=999., post_strike_vx_world=999.,
                    score=999., goal_success=False, incoming_vx=999.)
        np.testing.assert_array_equal(original, planned_action(plan, 1., [1., 1.]))
        np.testing.assert_allclose(original[3:5], [np.cos(np.deg2rad(25)), np.sin(np.deg2rad(25))])
        plan['coordinate_frame'] = 'world_rotated_180deg_about_z'
        with self.assertRaises(ValueError):
            planned_action(plan, 1., [1., 1.])

    def test_targets_do_not_interpolate_across_contacts(self):
        times = np.arange(21) * .05
        xy = np.column_stack((times, times * 2))
        target, mask = posthit_targets(times, xy, .211, .62, np.arange(1, 21) * .05, .02)
        queries = .211 + np.arange(1, 21) * .05
        self.assertTrue(np.all(queries[mask] > .211))
        self.assertTrue(np.all(queries[mask] < .60))
        np.testing.assert_allclose(target[mask, 0], queries[mask], atol=1e-6)
        self.assertFalse(mask[-1])
        self.assertTrue(np.all(target[~mask] == 0))

    def test_actions_change_forecast_and_backbone_stays_frozen(self):
        torch.set_num_threads(2)
        config = {'backbone_config': {'model_type': 'spatial', 'image_to_world': np.eye(3).tolist(),
                                     'hidden_dim': 64, 'dt_s': .05, 'horizon_steps': 20},
                  'dt_s': .05, 'history_frames': 9}
        model = HitPredictor(config)
        features = torch.randn(2, 58)
        action = torch.zeros(2, 6)
        times = torch.arange(1, 21) * .05
        result = model.forecast_features(features, action, times)
        changed = model.forecast_features(features, action + 1., times)
        self.assertFalse(torch.allclose(result['future_xy'], changed['future_xy']))
        mask = torch.zeros(2, 20, dtype=torch.bool)
        mask[:, :5] = True
        target = torch.zeros(2, 20, 2)
        loss = hit_loss(result, target, mask, torch.zeros(2))
        result['future_xy'].retain_grad()
        loss.backward()
        self.assertTrue(torch.all(result['future_xy'].grad[:, 5:] == 0))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        model.use_action = False
        a = model.forecast_features(features, action, times)
        b = model.forecast_features(features, action + 1., times)
        self.assertTrue(torch.equal(a['future_xy'], b['future_xy']))


if __name__ == '__main__':
    unittest.main()
