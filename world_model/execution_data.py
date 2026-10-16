"""Execution-state dataset; only decision-time encoder/command measurements are inputs."""

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from .execution import robot_features
from .hit_data import HitRGBDataset, planned_action


def native_samples(root, files, config):
    dataset = HitRGBDataset(root, files, config)
    for sample in dataset.samples:
        with h5py.File(Path(root) / sample['filename']) as file:
            times = file['task_state/timestamps'][:]
            index = int(np.argmin(abs(times - sample['decision_s'])))
            names = [value.decode() for value in file['observations/left/joint_names'][:]]
            sample['robot'] = torch.from_numpy(robot_features(
                file['observations/left/joint_positions'][index],
                file['observations/left/joint_velocities'][index],
                file['control/left_joint_drive_target'][index], names, config['arm_joint_names']))
        sample['contact'] = torch.tensor(1.)
    return dataset.samples


def branch_samples(root, split, allowed, config):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['split'] != split or not set(manifest['files']).issubset(allowed):
        raise ValueError('Branch split mismatch')
    samples = []
    scale = np.array(config['xy_scale_m'], dtype=np.float32)
    for path in sorted(root.glob('execution_*.npz')):
        with np.load(path, allow_pickle=False) as file:
            filename = str(file['source_filename'])
            if filename not in allowed or filename not in manifest['files']:
                raise ValueError('Branch source crossed splits')
            plan = json.loads(str(file['plan_json']))
            decision = float(file['decision_s'])
            contact = bool(file['contact'])
            action = planned_action(plan, decision, scale)
            if decision > plan['prep_time'] - config['contact_guard_s']:
                raise ValueError('History does not precede fast stroke')
            rgb = np.stack([cv2.resize(image, (config['image_width'], config['image_height']),
                                      interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
                            for image in file['rgb']])
            target = np.zeros((config['horizon_steps'], 2), dtype=np.float32)
            mask = np.zeros(config['horizon_steps'], dtype=bool)
            delay = 0.
            if contact:
                impact = float(file['actual_contact_s'])
                trajectory = file['actual_trajectory']
                if not np.isfinite(trajectory).all() or not np.all(np.diff(trajectory[:, 0]) > 0):
                    raise ValueError('Invalid physical trajectory')
                query = impact + np.arange(1, config['horizon_steps'] + 1) * config['dt_s']
                mask = (query >= trajectory[0, 0]) & (query <= trajectory[-1, 0] + 1e-8)
                target = np.column_stack([np.interp(query, trajectory[:, 0], trajectory[:, axis])
                                          for axis in (1, 2)]).astype(np.float32) / scale
                delay = decision + impact - plan['contact_time']
            names = [str(value) for value in file['joint_names']]
            measured = robot_features(file['q'], file['dq'], file['target'], names, config['arm_joint_names'])
            samples.append({'rgb': torch.from_numpy(rgb), 'robot': torch.from_numpy(measured),
                'action': torch.from_numpy(action), 'future_xy': torch.from_numpy(target),
                'mask': torch.from_numpy(mask), 'contact_delay_s': torch.tensor(delay, dtype=torch.float32),
                'contact': torch.tensor(float(contact)), 'filename': filename})
    if not samples:
        raise ValueError('No execution records')
    return samples
