"""Train/evaluate the RGB-only pre-contact puck predictor without SAPIEN."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
import yaml

from world_model.data import PuckRGBDataset, camera_plane_projection, split_episodes
from world_model.model import build_model, trajectory_loss


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def validate_config(config):
    for key in ('history_frames', 'horizon_steps', 'minimum_future_steps', 'image_width',
                'image_height', 'hidden_dim', 'batch_size', 'epochs', 'cpu_threads'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['history_frames'] < 2:
        raise ValueError('Motion prediction requires at least two history frames')
    if config['minimum_future_steps'] > config['horizon_steps']:
        raise ValueError('minimum_future_steps exceeds horizon_steps')
    for key in ('dt_s', 'learning_rate'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('contact_guard_s', 'localization_weight', 'weight_decay', 'horizon_weight_growth'):
        if not np.isfinite(config.get(key, 0.)) or config.get(key, 0.) < 0:
            raise ValueError(f'{key} must be finite and nonnegative')


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, future_times, scale, device, max_batches=None):
    """Report valid-point errors in meters, with sample counts for each horizon.

    Constant-velocity baseline uses privileged GT current position/velocity;
    it is a diagnostic reference, not a deployable RGB baseline.
    """
    model.eval()
    errors, masks, baselines, rgb_baselines, localization = [], [], [], [], []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = to_device(batch, device)
        result = model(batch['rgb'], future_times)
        truth = batch['future_xy'] * scale
        error = torch.linalg.vector_norm((result['future_xy'] * scale) - truth, dim=-1)
        baseline = batch['current_xy_m'][:, None] + (
            future_times[None, :, None] * batch['current_velocity_m_s'][:, None])
        errors.append(error.cpu().numpy())
        masks.append(batch['mask'].cpu().numpy())
        baselines.append(torch.linalg.vector_norm(baseline - truth, dim=-1).cpu().numpy())
        rgb_history = result['history_xy'] * scale
        rgb_velocity = (rgb_history[:, -1] - rgb_history[:, -2]) / future_times[0]
        rgb_baseline = rgb_history[:, -1, None] + future_times[None, :, None] * rgb_velocity[:, None]
        rgb_baselines.append(torch.linalg.vector_norm(rgb_baseline - truth, dim=-1).cpu().numpy())
        localization.append(torch.linalg.vector_norm(
            (result['history_xy'][:, -1] - batch['history_xy'][:, -1]) * scale,
            dim=-1).cpu().numpy())
    if not errors:
        raise ValueError('Evaluation received no batches')
    error, mask, baseline = np.concatenate(errors), np.concatenate(masks), np.concatenate(baselines)
    rgb_baseline = np.concatenate(rgb_baselines)
    summary = {
        'windows': len(error), 'valid_future_points': int(mask.sum()),
        'ade_m': float(error[mask].mean()),
        'point_error_p95_m': float(np.percentile(error[mask], 95)),
        'current_localization_error_m': float(np.concatenate(localization).mean()),
        'privileged_constant_velocity_ade_m': float(baseline[mask].mean()),
        'rgb_constant_velocity_ade_m': float(rgb_baseline[mask].mean()),
        'per_horizon': [],
    }
    for index, seconds in enumerate(future_times.cpu().numpy()):
        valid = mask[:, index]
        summary['per_horizon'].append({
            'seconds': round(float(seconds), 6), 'windows': int(valid.sum()),
            'mean_error_m': float(error[valid, index].mean()) if valid.any() else None,
            'p95_error_m': float(np.percentile(error[valid, index], 95)) if valid.any() else None,
            'privileged_constant_velocity_error_m': (
                float(baseline[valid, index].mean()) if valid.any() else None),
            'rgb_constant_velocity_error_m': (
                float(rgb_baseline[valid, index].mean()) if valid.any() else None),
        })
    return summary


@torch.no_grad()
def plot_predictions(model, dataset, future_times, scale, device, output):
    """Export held-out RGB inputs and world-XY forecast examples for review."""
    os.environ.setdefault('MPLCONFIGDIR', str(Path(output).parent / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    chosen, seen = [], set()
    candidates = sorted(range(len(dataset)),
                        key=lambda index: -dataset.windows[index][2])
    for index in candidates:
        episode_id = dataset.windows[index][0]
        if episode_id not in seen:
            chosen.append(index)
            seen.add(episode_id)
        if len(chosen) == 4:
            break
    fig, axes = plt.subplots(len(chosen), 2, figsize=(11, 3.2 * len(chosen)), squeeze=False)
    for row, index in enumerate(chosen):
        sample = dataset[index]
        result = model(sample['rgb'][None].to(device), future_times)
        prediction = (result['future_xy'][0] * scale).cpu().numpy()
        valid = sample['mask'].numpy()
        actual = sample['future_xy'].numpy() * scale.cpu().numpy()
        history = sample['history_xy'].numpy() * scale.cpu().numpy()
        episode = dataset.episodes[sample['episode_id']]
        axes[row, 0].imshow(sample['rgb'][-1].permute(1, 2, 0).numpy())
        axes[row, 0].set_title(f"{episode['filename']} | t={episode['times'][sample['frame_index']]:.2f}s")
        axes[row, 0].axis('off')
        ax = axes[row, 1]
        ax.plot(history[:, 0], history[:, 1], 'k.-', label='Observed history (GT for display)')
        ax.plot(actual[valid, 0], actual[valid, 1], 'g.-', label='Actual pre-contact future')
        ax.plot(prediction[valid, 0], prediction[valid, 1], 'r.-', label='RGB prediction')
        ax.set(xlim=(-float(scale[0]), float(scale[0])),
               ylim=(-float(scale[1]), float(scale[1])), xlabel='World X (m)', ylabel='World Y (m)')
        ax.set_aspect('equal')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'env_cfg/world_model_rgb.yml')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-batches', type=int, help='Limited smoke run; not a full evaluation')
    parser.add_argument('--evaluate', type=Path, help='Evaluate an existing checkpoint; no fitting')
    args = parser.parse_args()
    checkpoint = None
    if args.evaluate:
        # Only load checkpoints produced by this script and trusted by the user.
        checkpoint = torch.load(args.evaluate, map_location='cpu', weights_only=True)
        config, splits = checkpoint['config'], checkpoint['splits']
    else:
        config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        if args.epochs is not None:
            config['epochs'] = args.epochs
        if args.batch_size is not None:
            config['batch_size'] = args.batch_size
        splits = None
    validate_config(config)
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError('--max-batches must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; repair the GPU environment or use --device cpu')
    device = torch.device('cuda' if args.device == 'cuda' or (
        args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    torch.set_num_threads(config['cpu_threads'])
    random.seed(config['training_seed'])
    np.random.seed(config['training_seed'])
    torch.manual_seed(config['training_seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['training_seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    data_root = ROOT / config['data_root']
    splits = splits or split_episodes(data_root, config['split_seed'])
    if config.get('model_type') == 'spatial' and not checkpoint:
        with h5py.File(data_root / splits['train'][0], 'r') as file:
            _, calibration = camera_plane_projection(
                file, config['puck_plane_z_m'], config['xy_scale_m'])
        config['image_to_world'] = calibration.tolist()
    fingerprints = {name: hashlib.sha256((data_root / name).read_bytes()).hexdigest()
                    for names in splits.values() for name in names}
    initialization = None
    if config.get('initial_checkpoint') and not checkpoint:
        initialization = torch.load(ROOT / config['initial_checkpoint'],
                                    map_location='cpu', weights_only=True)
        if initialization['splits'] != splits or initialization['dataset_sha256'] != fingerprints:
            raise ValueError('Initialization dataset or splits mismatch')
        for key in ('model_type', 'hidden_dim', 'dt_s', 'history_frames', 'horizon_steps',
                    'image_width', 'image_height', 'xy_scale_m', 'image_to_world'):
            if initialization['config'].get(key) != config.get(key):
                raise ValueError(f'Initialization configuration mismatch: {key}')
    if checkpoint and fingerprints != checkpoint['dataset_sha256']:
        raise ValueError('Dataset changed since training; refusing incompatible evaluation')
    run_dir = ROOT / config['output_root'] / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True)
    write_json(run_dir / 'config.json', config)
    write_json(run_dir / 'splits.json', splits)
    write_json(run_dir / 'dataset_sha256.json', fingerprints)
    print(f'RUN_DIR={run_dir}', flush=True)
    print(f'Device: {device}; RGB-only input; pre-first-contact supervision', flush=True)
    datasets = {name: PuckRGBDataset(data_root, files, config)
                for name, files in splits.items() if not checkpoint or name == 'test'}
    write_json(run_dir / 'data_report.json', {name: value.report for name, value in datasets.items()})
    for name, dataset in datasets.items():
        print(f'{name}: {len(dataset)} windows, '
              f'{dataset.report["full_horizon_windows"]} full-horizon windows', flush=True)
    loaders = {name: DataLoader(dataset, batch_size=config['batch_size'], shuffle=name == 'train',
                               num_workers=0, pin_memory=device.type == 'cuda')
               for name, dataset in datasets.items()}
    model = build_model(config).to(device)
    if initialization:
        model.load_state_dict(initialization['model'])
    if config.get('freeze_localization', False):
        if not initialization:
            raise ValueError('Freezing localization requires an initial checkpoint')
        for module in (model.encoder, model.heatmap_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    parameter_count = sum(value.numel() for value in model.parameters())
    print(f'Parameters: {parameter_count:,}', flush=True)
    future_times = torch.arange(1, config['horizon_steps'] + 1, device=device) * config['dt_s']
    scale = torch.tensor(config['xy_scale_m'], device=device)
    best_epoch, best_score = None, float('inf')
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        best_epoch = checkpoint['epoch']
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                      weight_decay=config['weight_decay'])
        initial = evaluate(model, loaders['val'], future_times, scale, device, args.max_batches)
        write_json(run_dir / 'initial_validation.json', initial)
        for epoch in range(1, config['epochs'] + 1):
            start = time.monotonic()
            model.train()
            total, examples = 0., 0
            for index, batch in enumerate(loaders['train']):
                if args.max_batches is not None and index >= args.max_batches:
                    break
                batch = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = trajectory_loss(model(batch['rgb'], future_times), batch,
                                       config['localization_weight'], config.get('heatmap_weight', .2),
                                       config.get('horizon_weight_growth', 0.))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += loss.item() * len(batch['rgb'])
                examples += len(batch['rgb'])
            validation = evaluate(model, loaders['val'], future_times, scale, device, args.max_batches)
            record = {'epoch': epoch, 'train_loss': total / examples,
                      'validation': validation, 'seconds': time.monotonic() - start}
            with (run_dir / 'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
            print(f'epoch={epoch} loss={record["train_loss"]:.5f} '
                  f'val_ADE={validation["ade_m"]:.4f}m '
                  f'localization={validation["current_localization_error_m"]:.4f}m '
                  f'time={record["seconds"]:.1f}s', flush=True)
            if validation['ade_m'] < best_score:
                best_score, best_epoch = validation['ade_m'], epoch
                torch.save({'model': model.state_dict(), 'config': config, 'splits': splits,
                            'dataset_sha256': fingerprints, 'epoch': epoch,
                            'validation': validation}, run_dir / 'best.pt')
        checkpoint = torch.load(run_dir / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
    result = {
        'device': str(device), 'parameters': parameter_count, 'best_epoch': best_epoch,
        'limited_batches': args.max_batches,
        'input': 'RGB history only; GT used exclusively for supervision and diagnostics',
        'scope': 'future strictly before first physical striker contact; walls allowed',
        'test': evaluate(model, loaders['test'], future_times, scale, device, args.max_batches),
        'cuda_peak_allocated_bytes': (torch.cuda.max_memory_allocated() if device.type == 'cuda' else None),
    }
    if not args.evaluate:
        result['validation'] = checkpoint['validation']
    write_json(run_dir / 'metrics.json', result)
    plot_predictions(model, datasets['test'], future_times, scale, device, run_dir / 'test_predictions.png')
    print(f'Test ADE: {result["test"]["ade_m"]:.4f}m; metrics: {run_dir / "metrics.json"}', flush=True)


if __name__ == '__main__':
    main()
