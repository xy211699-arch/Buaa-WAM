"""Load synchronized native RGB and pre-contact XY supervision in SI units."""

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from transforms3d.quaternions import quat2mat

from data.decode_image_bit import decode_image_bit


def camera_plane_projection(file, plane_z_m, scale):
    """Calibrate normalized image UV to normalized world XY on the puck plane.

    SAPIEN camera pose uses X forward, Y left, Z up. Use recorded camera pose,
    not the legacy extrinsics field (whose conventions differ across exports).
    Calibration is fixed camera geometry, not a ground-truth runtime input.
    """
    metadata = json.loads(file['episode/metadata_json'][()])
    pose = metadata['camera']['pose']
    rotation = quat2mat(pose['q'])
    cv_axes = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    extrinsic = cv_axes @ np.column_stack((rotation.T, -rotation.T @ np.asarray(pose['p'])))
    intrinsic = file['vision/cam_head/intrinsic_matrix'][0]
    projection = intrinsic @ extrinsic
    homography = np.column_stack((projection[:, :2],
                                  projection[:, 2] * plane_z_m + projection[:, 3]))
    height, width = file['vision/cam_head/shape'][:2]
    world_to_uv = np.diag([1. / width, 1. / height, 1.]) @ homography
    uv_to_xy = np.diag([1. / scale[0], 1. / scale[1], 1.]) @ np.linalg.inv(world_to_uv)
    return world_to_uv, uv_to_xy


def split_episodes(root, seed=42):
    """Split complete seed groups 70/15/15; duplicate seeds never cross splits."""
    groups = {}
    for path in sorted(Path(root).glob('*.hdf5')):
        with h5py.File(path, 'r') as file:
            episode_seed = int(file.attrs['seed'])
        groups.setdefault(episode_seed, []).append(path.name)
    keys = sorted(groups)
    if len(keys) < 7:
        raise ValueError('At least seven independent seeds are required')
    np.random.default_rng(seed).shuffle(keys)
    train_end = int(len(keys) * .7)
    val_end = train_end + max(1, int(len(keys) * .15))
    return {name: [filename for key in subset for filename in groups[key]]
            for name, subset in zip(('train', 'val', 'test'),
                                    (keys[:train_end], keys[train_end:val_end], keys[val_end:]))}


def precontact_windows(timestamps, contact_time, history, horizon, minimum_future,
                       dt, guard):
    """Return (last input index, available future length) strictly before contact."""
    times = np.asarray(timestamps)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError('Expected finite one-dimensional timestamps')
    if not np.allclose(np.diff(times), dt, atol=1e-7, rtol=0):
        raise ValueError('RGB/state sampling must match the configured fixed timestep')
    if not np.isfinite(contact_time):
        raise ValueError('Missing finite physical contact time')
    count = int(np.searchsorted(times, contact_time - guard, side='left'))
    return [(index, min(horizon, count - index - 1))
            for index in range(history - 1, count - minimum_future)]


class PuckRGBDataset(Dataset):
    """RGB is the only model input; positions and velocities are labels/diagnostics.

    Decode each episode once into CPU uint8 memory. HDF5 handles are not retained.
    Native cam_head frames correspond to task_state timestamps, not 50-FPS video.
    Future labels at/after the first physical striker contact are always masked.
    """

    def __init__(self, root, filenames, config):
        self.history = config['history_frames']
        self.horizon = config['horizon_steps']
        self.scale = np.asarray(config['xy_scale_m'], dtype=np.float32)
        self.episodes, self.windows = [], []
        self.report = {'episodes': [], 'windows': 0, 'full_horizon_windows': 0}
        if self.scale.shape != (2,) or np.any(self.scale <= 0):
            raise ValueError('xy_scale_m must contain two positive lengths')
        for filename in filenames:
            with h5py.File(Path(root) / filename, 'r') as file:
                times = file['task_state/timestamps'][:]
                images = file['vision/cam_head/colors']
                xyz = file['task_state/puck_pose'][:, :2].astype(np.float32)
                velocity = file['task_state/puck_linear_velocity'][:, :2].astype(np.float32)
                if len(images) != len(times) or len(xyz) != len(times):
                    raise ValueError(f'{filename}: RGB/state length mismatch')
                if not np.allclose(times, file['observations/timestamps'][:-1], atol=1e-8):
                    raise ValueError(f'{filename}: RGB/state timestamp mismatch')
                if not np.isfinite(xyz).all() or not np.isfinite(velocity).all():
                    raise ValueError(f'{filename}: nonfinite puck labels')
                metadata = json.loads(file['episode/metadata_json'][()])
                contacts = metadata.get('contacts', [])
                if not contacts:
                    raise ValueError(f'{filename}: no physical contact metadata')
                contact = min(float(item['episode_time']) for item in contacts)
                history_uv = None
                if config.get('model_type') == 'spatial':
                    projection, calibration = camera_plane_projection(
                        file, config['puck_plane_z_m'], self.scale)
                    if not np.allclose(calibration, config['image_to_world'], atol=1e-5):
                        raise ValueError(f'{filename}: fixed-camera calibration mismatch')
                    uv = np.column_stack((xyz, np.ones(len(xyz)))) @ projection.T
                    history_uv = (uv[:, :2] / uv[:, 2:]).astype(np.float32)
                windows = precontact_windows(times, contact, self.history, self.horizon,
                    config['minimum_future_steps'], config['dt_s'], config['contact_guard_s'])
                if not windows:
                    self.report['episodes'].append({'file': filename, 'windows': 0})
                    continue
                count = max(index + available + 1 for index, available in windows)
                frames = []
                for index in range(count):
                    rgb = decode_image_bit(images[index])
                    rgb = cv2.resize(rgb, (config['image_width'], config['image_height']),
                                     interpolation=cv2.INTER_AREA)
                    frames.append(rgb.transpose(2, 0, 1))
                episode = {'rgb': np.stack(frames), 'xy': xyz[:count],
                           'velocity': velocity[:count], 'times': times[:count],
                           'filename': filename}
                if history_uv is not None:
                    episode['uv'] = history_uv[:count]
            episode_id = len(self.episodes)
            self.episodes.append(episode)
            self.windows.extend((episode_id, index, available) for index, available in windows)
            full = sum(available == self.horizon for _, available in windows)
            self.report['episodes'].append({'file': filename, 'windows': len(windows),
                                            'full_horizon_windows': full})
            self.report['full_horizon_windows'] += full
        self.report['windows'] = len(self.windows)
        if not self.windows:
            raise ValueError('No valid pre-contact windows in this split')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, item):
        episode_id, index, available = self.windows[item]
        episode = self.episodes[episode_id]
        start = index - self.history + 1
        target = np.zeros((self.horizon, 2), dtype=np.float32)
        target[:available] = episode['xy'][index + 1:index + 1 + available] / self.scale
        mask = np.arange(self.horizon) < available
        sample = {
            'rgb': torch.from_numpy(episode['rgb'][start:index + 1].copy()),
            'history_xy': torch.from_numpy(episode['xy'][start:index + 1] / self.scale),
            'future_xy': torch.from_numpy(target),
            'mask': torch.from_numpy(mask),
            'current_xy_m': torch.from_numpy(episode['xy'][index].copy()),
            'current_velocity_m_s': torch.from_numpy(episode['velocity'][index].copy()),
            'episode_id': episode_id, 'frame_index': index,
        }
        if 'uv' in episode:
            sample['history_uv'] = torch.from_numpy(episode['uv'][start:index + 1].copy())
        return sample
