"""Decision-relative supervision, never exposing future labels to policy inputs."""

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from .execution import robot_features
from .execution_data import native_samples
from .policy import expert_intention


def future_targets(times, xy, decision_s, query_offsets, contacts=(), upper_s=float('inf')):
    """Use exact timestamp states or interpolate without crossing contact events."""
    times, xy = np.asarray(times), np.asarray(xy)
    query_offsets = np.asarray(query_offsets)
    if not np.isfinite(decision_s) or query_offsets.ndim != 1 or not np.isfinite(query_offsets).all() or np.any(query_offsets <= 0):
        raise ValueError('Finite positive decision-relative offsets required')
    if len(times) < 2 or xy.shape != (len(times), 2) or not np.isfinite(times).all() or not np.isfinite(xy).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid measured trajectory')
    target = np.zeros((len(query_offsets), 2), dtype=np.float32)
    mask = np.zeros(len(query_offsets), dtype=bool)
    for k, query in enumerate(decision_s + query_offsets):
        if query >= upper_s or query < times[0] - 1e-8 or query > times[-1] + 1e-8:
            continue
        nearest = int(np.argmin(abs(times - query)))
        if abs(times[nearest] - query) < 1e-7:
            target[k], mask[k] = xy[nearest], True
            continue
        hi = int(np.searchsorted(times, query)); lo = hi - 1
        if lo < 0 or hi >= len(times) or any(times[lo] <= event <= times[hi] for event in contacts):
            continue
        alpha = (query - times[lo]) / (times[hi] - times[lo])
        target[k], mask[k] = xy[lo] * (1 - alpha) + xy[hi] * alpha, True
    return target, mask


def load_native(root, files, config):
    samples = native_samples(root, files, config)
    offsets = np.arange(1, config['horizon_steps'] + 1) * config['dt_s']
    scale = np.asarray(config['xy_scale_m'], dtype=np.float32)
    metadata = {}
    for filename in files:
        with h5py.File(Path(root) / filename) as file:
            meta = json.loads(file['episode/metadata_json'][()])
            contacts = meta['contacts']
            left = min(item['episode_time'] for item in contacts if item['arm'] == 'left')
            right = min(item['episode_time'] for item in contacts if item['arm'] == 'right')
            metadata[filename] = (file['task_state/timestamps'][:], file['task_state/puck_pose'][:, :2],
                json.loads(file['episode/expert_summary_json'][()])['final_best'], left, right)
    rows = []
    for sample in samples:
        times, xy, plan, left, right = metadata[sample['filename']]
        if plan.get('attack_mode', 'DIRECT') != 'DIRECT':
            continue
        target, mask = future_targets(times, xy, sample['decision_s'], offsets,
                                      contacts=[left, right], upper_s=right - config['contact_guard_s'])
        intention, bank = expert_intention(plan, sample['decision_s'])
        rows.append({'rgb': sample['rgb'], 'robot': sample['robot'],
            'intention': torch.tensor(intention), 'bank': torch.tensor(bank),
            'future_xy': torch.tensor(target / scale), 'mask': torch.tensor(mask),
            'expert': torch.tensor(True), 'contact': torch.tensor(1.),
            'contact_delay_s': torch.tensor(left - sample['decision_s'], dtype=torch.float32),
            'filename': sample['filename']})
    return rows


def load_branches(root, allowed, split, config):
    """Existing branches are dynamics-only labels; failed motions retain observed XY."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') == 'joint_full_decision_tail_v2' and not (root / 'summary.json').is_file():
        raise ValueError('Branch export not complete')
    if manifest['split'] != split or not set(manifest['files']).issubset(allowed):
        raise ValueError('Branch split mismatch')
    rows = []
    offsets = np.arange(1, config['horizon_steps'] + 1) * config['dt_s']
    for path in sorted(root.glob('execution_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            filename = str(data['source_filename'])
            if filename not in allowed or filename not in manifest['files']:
                raise ValueError('Cross-split branch')
            plan = json.loads(str(data['plan_json']))
            if plan.get('attack_mode', 'DIRECT') != 'DIRECT':
                continue
            decision = float(data['decision_s'])
            if decision > plan['prep_time'] - config['contact_guard_s']:
                raise ValueError('History follows stroke')
            intention, bank = expert_intention(plan, decision)
            trajectory = data['actual_trajectory']
            contact = bool(data['contact'])
            impact = float(data['actual_contact_s']) if contact else None
            observed = float(data['observed_contact_s']) if 'observed_contact_s' in data else impact
            events = [observed] if observed is not None and np.isfinite(observed) else []
            target, mask = future_targets(trajectory[:, 0], trajectory[:, 1:3], 0., offsets,
                                          contacts=events)
            rgb = np.stack([cv2.resize(frame, (config['image_width'], config['image_height']),
                                      interpolation=cv2.INTER_AREA).transpose(2, 0, 1) for frame in data['rgb']])
            names = [str(value) for value in data['joint_names']]
            robot = robot_features(data['q'], data['dq'], data['target'], names, config['arm_joint_names'])
            rows.append({'rgb': torch.tensor(rgb), 'robot': torch.tensor(robot),
                'intention': torch.tensor(intention), 'bank': torch.tensor(bank),
                'future_xy': torch.tensor(target / np.asarray(config['xy_scale_m']), dtype=torch.float32),
                'mask': torch.tensor(mask), 'expert': torch.tensor(False),
                'contact': torch.tensor(float(contact)),
                'contact_delay_s': torch.tensor(impact if contact else 0., dtype=torch.float32),
                'filename': filename})
    return rows
