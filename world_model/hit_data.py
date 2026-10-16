"""Pre-strike RGB and planned left-hit actions; outcomes are supervision only."""

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.decode_image_bit import decode_image_bit
from .data import camera_plane_projection


def planned_action(plan, decision_s, scale):
    """Return planned delay(s), contact XY(normalized), direction XY, speed(m/s).

    Only command fields are read. Actual contact, outgoing velocity, score,
    physical parameters, and goal outcomes must never enter this vector.
    """
    if plan.get('arm') != 'left' or plan.get('coordinate_frame') != 'world':
        raise ValueError('First implementation requires a left-arm world-frame plan')
    angle = np.deg2rad(plan['nominal_theta_deg'] + plan['direction_correction_deg'])
    action = np.array([plan['contact_time'] - decision_s,
                       plan['contact_x'] / scale[0], plan['contact_y'] / scale[1],
                       np.cos(angle), np.sin(angle), plan['hit_speed']], dtype=np.float32)
    if not np.isfinite(action).all() or action[0] <= 0 or action[-1] <= 0:
        raise ValueError('Expected finite future planned hit with positive speed')
    return action


def posthit_targets(times, xy, contact_s, next_contact_s, future_times, guard_s):
    """Interpolate only between measured states strictly after this hit/before next.

    Contact timestamps are labels for selecting target states, not model inputs.
    Never extrapolate or interpolate across either striker contact.
    """
    query = contact_s + future_times
    upper = np.searchsorted(times, query, side='right')
    lower = upper - 1
    valid = (lower >= 0) & (upper < len(times))
    lo, hi = np.clip(lower, 0, len(times) - 1), np.clip(upper, 0, len(times) - 1)
    valid &= (times[lo] > contact_s) & (times[hi] < next_contact_s - guard_s)
    result = np.zeros((len(query), 2), dtype=np.float32)
    fraction = (query[valid] - times[lo[valid]]) / (times[hi[valid]] - times[lo[valid]])
    result[valid] = xy[lo[valid]] * (1 - fraction[:, None]) + xy[hi[valid]] * fraction[:, None]
    return result, valid


class HitRGBDataset(Dataset):
    """Four pre-strike decisions per seed; histories may contain preparation motion."""

    def __init__(self, root, filenames, config):
        self.samples = []
        scale = np.asarray(config['xy_scale_m'], dtype=np.float32)
        if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Expected two finite positive XY scales')
        if config['contact_guard_s'] < 0 or any(
                not np.isfinite(offset) or offset < 0 for offset in config['decision_offsets_s']):
            raise ValueError('Decision offsets and contact guard must be nonnegative')
        future_times = np.arange(1, config['horizon_steps'] + 1) * config['dt_s']
        for filename in filenames:
            with h5py.File(Path(root) / filename, 'r') as file:
                times = file['task_state/timestamps'][:]
                if not np.allclose(np.diff(times), config['dt_s'], atol=1e-7, rtol=0):
                    raise ValueError('Fixed state sampling required')
                images = file['vision/cam_head/colors']
                if len(images) != len(times) or not np.allclose(
                        times, file['observations/timestamps'][:-1], atol=1e-8):
                    raise ValueError('RGB timestamps mismatch')
                xy = file['task_state/puck_pose'][:, :2]
                metadata = json.loads(file['episode/metadata_json'][()])
                summary = json.loads(file['episode/expert_summary_json'][()])
                plan = summary['final_best']
                left = min(item['episode_time'] for item in metadata['contacts'] if item['arm'] == 'left')
                right = min(item['episode_time'] for item in metadata['contacts'] if item['arm'] == 'right')
                if right <= left:
                    raise ValueError('Expected left contact before right contact')
                _, calibration = camera_plane_projection(file, config['puck_plane_z_m'], scale)
                if not np.allclose(calibration, config['image_to_world'], atol=1e-5):
                    raise ValueError('Camera calibration mismatch')
                target, mask = posthit_targets(times, xy, left, right, future_times,
                                               config['contact_guard_s'])
                if not np.isfinite(target).all() or int(mask.sum()) < 2:
                    raise ValueError('Insufficient finite post-hit labels')
                seen = set()
                for offset in config['decision_offsets_s']:
                    cutoff = min(plan['prep_time'], left) - config['contact_guard_s'] - offset
                    end = int(np.searchsorted(times, cutoff, side='right')) - 1
                    start = end - config['history_frames'] + 1
                    if start < 0 or end in seen:
                        continue
                    seen.add(end)
                    rgb = []
                    for index in range(start, end + 1):
                        image = decode_image_bit(images[index])
                        rgb.append(cv2.resize(image, (config['image_width'], config['image_height']),
                                              interpolation=cv2.INTER_AREA).transpose(2, 0, 1))
                    self.samples.append({
                        'rgb': torch.from_numpy(np.stack(rgb)),
                        'action': torch.from_numpy(planned_action(plan, times[end], scale)),
                        'future_xy': torch.from_numpy(target / scale),
                        'mask': torch.from_numpy(mask),
                        'contact_delay_s': torch.tensor(left - plan['contact_time'], dtype=torch.float32),
                        'filename': filename,
                        'decision_s': float(times[end]),
                        'last_rgb_s': float(times[end]),
                        'actual_contact_s': float(left),
                        'prep_s': float(plan['prep_time']),
                    })
        if not self.samples:
            raise ValueError('No valid pre-command histories')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def counterfactual_samples(root, split, allowed_filenames, config):
    """Load recorded same-state branches, refusing cross-split seed provenance."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['split'] != split or not set(manifest['files']).issubset(allowed_filenames):
        raise ValueError('Counterfactual split mismatch')
    scale = np.asarray(config['xy_scale_m'], dtype=np.float32)
    samples = []
    for path in sorted(root.glob('*.npz')):
        if path.name.startswith('execution_'):
            continue
        with np.load(path, allow_pickle=False) as file:
            filename = str(file['source_filename'])
            if filename not in allowed_filenames:
                raise ValueError('Counterfactual episode crossed splits')
            plan = json.loads(str(file['plan_json']))
            decision = float(file['decision_s'])
            contact = float(file['actual_contact_s'])
            trajectory = file['actual_trajectory']
            if not np.isfinite(trajectory).all() or not np.all(np.diff(trajectory[:, 0]) > 0):
                raise ValueError('Invalid counterfactual timeline')
            query = contact + np.arange(1, config['horizon_steps'] + 1) * config['dt_s']
            mask = (query > trajectory[0, 0]) & (query < trajectory[-1, 0] + 1e-8)
            target = np.column_stack([np.interp(query, trajectory[:, 0], trajectory[:, axis])
                                      for axis in (1, 2)]).astype(np.float32) / scale
            rgb = np.stack([cv2.resize(image, (config['image_width'], config['image_height']),
                                      interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
                            for image in file['rgb']])
            samples.append({'rgb': torch.from_numpy(rgb),
                'action': torch.from_numpy(planned_action(plan, decision, scale)),
                'future_xy': torch.from_numpy(target), 'mask': torch.from_numpy(mask),
                'contact_delay_s': torch.tensor(decision + contact - plan['contact_time'], dtype=torch.float32),
                'filename': filename, 'decision_s': decision, 'last_rgb_s': decision,
                'actual_contact_s': decision + contact, 'prep_s': plan['prep_time']})
    if not samples:
        raise ValueError('No valid counterfactual samples')
    return samples
