"""Regression tests for contact isolation, episode splits, and RGB supervision."""

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from data.decode_image_bit import encode_image_bit
from world_model.data import PuckRGBDataset, camera_plane_projection, precontact_windows, split_episodes
from world_model.model import RGBPuckPredictor, SpatialRGBPuckPredictor, trajectory_loss
from scripts.train_world_model_rgb import evaluate


class WorldModelTests(unittest.TestCase):
    def test_horizon_weighting_and_mask(self):
        prediction = {'future_xy': torch.tensor([[[1., 0.], [2., 0.], [99., 0.]]]),
                      'history_xy': torch.zeros(1, 2, 2)}
        batch = {'future_xy': torch.zeros(1, 3, 2),
                 'history_xy': torch.zeros(1, 2, 2),
                 'mask': torch.tensor([[True, True, False]])}
        self.assertAlmostEqual(float(trajectory_loss(prediction, batch)), 2.5)
        self.assertAlmostEqual(float(trajectory_loss(
            prediction, batch, horizon_weight_growth=2.)), 3.)

    @staticmethod
    def make_episode(path, seed=0):
        times = np.arange(41) * .05
        with h5py.File(path, 'w') as file:
            file.attrs['seed'] = seed
            file.create_dataset('task_state/timestamps', data=times[:-1])
            file.create_dataset('observations/timestamps', data=times)
            pose = np.zeros((40, 7))
            pose[:, 0], pose[:, 1], pose[:, 3] = times[:-1], .1, 1
            file.create_dataset('task_state/puck_pose', data=pose)
            file.create_dataset('task_state/puck_linear_velocity', data=np.tile([1, 0, 0], (40, 1)))
            bits = encode_image_bit(np.full((24, 32, 3), [240, 20, 10], dtype=np.uint8))
            file.create_dataset('vision/cam_head/colors', data=np.array([bits] * 40, dtype=f'S{len(bits)}'))
            file.create_dataset('vision/cam_head/shape', data=[24, 32, 3])
            intrinsic = np.array([[23., 0., 16.], [0., 23., 12.], [0., 0., 1.]])
            file.create_dataset('vision/cam_head/intrinsic_matrix', data=np.tile(intrinsic, (40, 1, 1)))
            file.create_dataset('episode/metadata_json', data=json.dumps({
                'camera': {'pose': {'p': [0, 0, 2.74], 'q': [-.5, .5, -.5, -.5]}},
                'contacts': [{'arm': 'left', 'episode_time': 1.2},
                             {'arm': 'right', 'episode_time': 1.7}]}))

    def test_future_and_history_never_cross_first_contact(self):
        config = {'history_frames': 9, 'horizon_steps': 20, 'minimum_future_steps': 2,
                  'dt_s': .05, 'contact_guard_s': .02, 'image_width': 32,
                  'image_height': 24, 'xy_scale_m': [1., 1.]}
        with tempfile.TemporaryDirectory() as directory:
            self.make_episode(Path(directory) / 'episode.hdf5')
            dataset = PuckRGBDataset(directory, ['episode.hdf5'], config)
            self.assertEqual(len(dataset), 14)
            for item in range(len(dataset)):
                sample = dataset[item]
                valid = sample['mask']
                self.assertGreaterEqual(int(valid.sum()), 2)
                self.assertTrue(torch.all(sample['future_xy'][valid, 0] < 1.18))
                self.assertTrue(torch.all(sample['history_xy'][:, 0] < 1.18))
                self.assertEqual(sample['rgb'].dtype, torch.uint8)
                self.assertGreater(int(sample['rgb'][0, 0].float().mean()), 200)
                self.assertLess(int(sample['rgb'][0, 2].float().mean()), 30)
            self.assertEqual(int(dataset[-1]['mask'].sum()), 2)

    def test_mismatched_rgb_timestamp_alignment_is_rejected(self):
        config = {'history_frames': 9, 'horizon_steps': 20, 'minimum_future_steps': 2,
                  'dt_s': .05, 'contact_guard_s': .02, 'image_width': 32,
                  'image_height': 24, 'xy_scale_m': [1., 1.]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.hdf5'
            self.make_episode(path)
            with h5py.File(path, 'a') as file:
                file['observations/timestamps'][3] += .01
            with self.assertRaisesRegex(ValueError, 'timestamp mismatch'):
                PuckRGBDataset(directory, ['episode.hdf5'], config)

    def test_duplicate_seed_groups_do_not_leak_between_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            for seed in range(10):
                for replica in range(2):
                    with h5py.File(Path(directory) / f'{seed}_{replica}.hdf5', 'w') as file:
                        file.attrs['seed'] = seed
            splits = split_episodes(directory)
            self.assertEqual(splits, split_episodes(directory))
            seed_sets = [{int(name.split('_')[0]) for name in names} for names in splits.values()]
            for index, seeds in enumerate(seed_sets):
                for other in seed_sets[index + 1:]:
                    self.assertFalse(seeds & other)
            self.assertEqual(sum(map(len, splits.values())), 20)

    def test_irregular_sampling_is_not_silently_interpolated(self):
        with self.assertRaisesRegex(ValueError, 'sampling'):
            precontact_windows([0, .05, .11], 2., 2, 20, 2, .05, .02)

    def test_masked_labels_have_no_loss_or_gradient(self):
        forecast = torch.zeros((1, 3, 2), requires_grad=True)
        prediction = {'future_xy': forecast, 'history_xy': torch.zeros(1, 2, 2)}
        batch = {'future_xy': torch.tensor([[[1., 1.], [100., 100.], [200., 200.]]]),
                 'history_xy': torch.zeros(1, 2, 2), 'mask': torch.tensor([[True, False, False]])}
        loss = trajectory_loss(prediction, batch)
        self.assertEqual(float(loss), 2.)
        loss.backward()
        self.assertTrue(torch.equal(forecast.grad[:, 1:], torch.zeros(1, 2, 2)))

    def test_rgb_only_model_shapes_and_training_gradients(self):
        torch.set_num_threads(2)
        model = RGBPuckPredictor(hidden_dim=32)
        rgb = torch.randint(0, 256, (2, 9, 3, 48, 64), dtype=torch.uint8)
        output = model(rgb, torch.arange(1, 21) * .05)
        self.assertEqual(output['future_xy'].shape, (2, 20, 2))
        self.assertEqual(output['history_xy'].shape, (2, 9, 2))
        output['future_xy'].square().mean().backward()
        self.assertIsNotNone(model.encoder[0].weight.grad)
        self.assertTrue(torch.isfinite(output['future_xy']).all())

    def test_camera_projection_and_spatial_supervision(self):
        torch.set_num_threads(2)
        config = {'history_frames': 9, 'horizon_steps': 20, 'minimum_future_steps': 2,
                  'dt_s': .05, 'contact_guard_s': .02, 'image_width': 32,
                  'image_height': 24, 'xy_scale_m': [1., 1.],
                  'model_type': 'spatial', 'puck_plane_z_m': .75}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.hdf5'
            self.make_episode(path)
            with h5py.File(path, 'r') as file:
                projection, calibration = camera_plane_projection(file, .75, [1., 1.])
            center = projection @ np.array([0., 0., 1.])
            np.testing.assert_allclose(center[:2] / center[2], [.5, .5])
            recovered = calibration @ (projection @ np.array([.7, -.3, 1.]))
            np.testing.assert_allclose(recovered[:2] / recovered[2], [.7, -.3])
            config['image_to_world'] = calibration.tolist()
            dataset = PuckRGBDataset(directory, ['episode.hdf5'], config)
            sample = dataset[0]
            batch = {key: value[None] for key, value in sample.items() if isinstance(value, torch.Tensor)}
            model = SpatialRGBPuckPredictor(calibration, hidden_dim=32)
            prediction = model(batch['rgb'], torch.arange(1, 21) * .05)
            loss = trajectory_loss(prediction, batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(float(model.heatmap_head.weight.grad.abs().sum()), 0)
            config['image_to_world'][0][2] += .1
            with self.assertRaisesRegex(ValueError, 'calibration mismatch'):
                PuckRGBDataset(directory, ['episode.hdf5'], config)

    def test_evaluation_masks_future_and_compares_same_rgb_perception(self):
        class PerfectMotion(torch.nn.Module):
            def forward(self, rgb, times):
                return {'history_xy': torch.tensor([[[0., 0.], [.05, 0.]]]),
                        'future_xy': torch.tensor([[[.1, 0.], [.15, 0.]]])}

        batch = {'rgb': torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8),
                 'history_xy': torch.tensor([[[0., 0.], [.05, 0.]]]),
                 'future_xy': torch.tensor([[[.1, 0.], [999., 999.]]]),
                 'mask': torch.tensor([[True, False]]),
                 'current_xy_m': torch.tensor([[.05, 0.]]),
                 'current_velocity_m_s': torch.tensor([[1., 0.]])}
        metrics = evaluate(PerfectMotion(), [batch], torch.tensor([.05, .1]),
                           torch.ones(2), torch.device('cpu'))
        self.assertAlmostEqual(metrics['ade_m'], 0.)
        self.assertAlmostEqual(metrics['rgb_constant_velocity_ade_m'], 0.)
        self.assertEqual(metrics['valid_future_points'], 1)
        self.assertIsNone(metrics['per_horizon'][1]['mean_error_m'])


if __name__ == '__main__':
    unittest.main()
