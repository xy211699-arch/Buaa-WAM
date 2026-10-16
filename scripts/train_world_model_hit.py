"""Train and compare action-conditioned and RGB-only post-left-hit predictors."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from world_model.hit_data import HitRGBDataset, counterfactual_samples
from world_model.hit_model import HitPredictor, hit_loss


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


@torch.no_grad()
def encode(model, dataset, device, batch_size):
    features, actions, targets, masks, delays = [], [], [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size):
        features.append(model.encode_rgb(batch['rgb'].to(device)))
        actions.append(batch['action'].to(device))
        targets.append(batch['future_xy'].to(device))
        masks.append(batch['mask'].to(device))
        delays.append(batch['contact_delay_s'].to(device))
    return tuple(torch.cat(items) for items in (features, actions, targets, masks, delays))


@torch.no_grad()
def evaluate(model, data, times, scale):
    features, actions, targets, masks, delays = data
    result = model.forecast_features(features, actions, times)
    errors = torch.linalg.vector_norm((result['future_xy'] - targets) * scale, dim=-1)
    report = {
        'windows': len(features), 'valid_points': int(masks.sum()),
        'ade_m': float(errors[masks].mean()),
        'p95_m': float(torch.quantile(errors[masks], .95)),
        'contact_delay_mae_s': float((result['contact_delay_s'] - delays).abs().mean()),
        'per_horizon': [],
    }
    for index, seconds in enumerate(times.cpu().numpy()):
        valid = masks[:, index]
        report['per_horizon'].append({
            'seconds_after_actual_contact': round(float(seconds), 6),
            'windows': int(valid.sum()),
            'mean_error_m': float(errors[valid, index].mean()) if valid.any() else None,
        })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'env_cfg/world_model_hit.yml')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for key in ('history_frames', 'horizon_steps', 'batch_size', 'epochs', 'cpu_threads'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['history_frames'] < 2 or not np.isfinite(config['learning_rate']) or config['learning_rate'] <= 0:
        raise ValueError('Invalid history length or learning rate')
    backbone = torch.load(ROOT / config['backbone_checkpoint'], map_location='cpu', weights_only=True)
    config['backbone_config'] = backbone['config']
    config['image_to_world'] = backbone['config']['image_to_world']
    for key in ('dt_s', 'history_frames', 'image_width', 'image_height', 'xy_scale_m', 'puck_plane_z_m'):
        if config[key] != backbone['config'][key]:
            raise ValueError(f'Backbone input mismatch: {key}')
    splits = backbone['splits']
    root = ROOT / config['data_root']
    fingerprints = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for names in splits.values() for name in names}
    if fingerprints != backbone['dataset_sha256']:
        raise ValueError('Data differs from frozen backbone training set')
    torch.set_num_threads(config['cpu_threads'])
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device(args.device)
    random.seed(config['training_seed'])
    np.random.seed(config['training_seed'])
    torch.manual_seed(config['training_seed'])
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    run = ROOT / config['output_root'] / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    save_json(run / 'config.json', config)
    save_json(run / 'splits.json', splits)
    save_json(run / 'dataset_sha256.json', fingerprints)
    print(f'RUN_DIR={run}', flush=True)
    datasets = {name: HitRGBDataset(root, files, config) for name, files in splits.items()}
    counterfactual_val = None
    cf_hashes = {}
    for name in ('train', 'val'):
        key = f'counterfactual_{name}_root'
        if config.get(key):
            branch_root = ROOT / config[key]
            samples = counterfactual_samples(branch_root, name, splits[name], config)
            cf_hashes.update({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in branch_root.glob('*.npz')})
            if name == 'train':
                datasets[name].samples.extend(samples)
            else:
                counterfactual_val = samples
    save_json(run / 'counterfactual_sha256.json', cf_hashes)
    reports = {name: {
        'episodes': len(set(sample['filename'] for sample in dataset.samples)),
        'windows': len(dataset),
        'valid_horizon_windows': torch.stack([sample['mask'] for sample in dataset.samples]).sum(0).tolist(),
    } for name, dataset in datasets.items()}
    save_json(run / 'data_report.json', reports)
    print(json.dumps(reports), flush=True)
    encoder = HitPredictor(config).to(device).eval()
    encoder.backbone.load_state_dict(backbone['model'])
    data = {name: encode(encoder, dataset, device, config['batch_size'])
            for name, dataset in datasets.items()}
    cf_val_data = None
    if counterfactual_val:
        cf_val_data = list(encode(encoder, counterfactual_val, device, config['batch_size']))
        cf_val_data[3] = cf_val_data[3].clone()
        cf_val_data[3][:, 10:] = False
        cf_val_data = tuple(cf_val_data)
    del encoder
    times = torch.arange(1, config['horizon_steps'] + 1, device=device) * config['dt_s']
    scale = torch.tensor(config['xy_scale_m'], device=device)
    results = {}
    for label, conditioned in [('action', True), ('no_action', False)]:
        torch.manual_seed(config['training_seed'])
        model = HitPredictor(config, use_action=conditioned).to(device)
        model.backbone.load_state_dict(backbone['model'])
        if conditioned and config.get('initial_hit_checkpoint'):
            initial = torch.load(ROOT / config['initial_hit_checkpoint'], map_location='cpu', weights_only=True)
            if initial['splits'] != splits or initial['dataset_sha256'] != fingerprints:
                raise ValueError('Initial hit model split/data mismatch')
            model.load_state_dict(initial['model'])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=config['learning_rate'], weight_decay=config['weight_decay'])
        best_score = float('inf')
        for epoch in range(1, config['epochs'] + 1):
            model.train()
            indices = torch.randperm(len(data['train'][0]), device=device)
            for index in indices.split(config['batch_size']):
                features, actions, targets, masks, delays = [items[index] for items in data['train']]
                optimizer.zero_grad(set_to_none=True)
                loss = hit_loss(model.forecast_features(features, actions, times), targets, masks, delays)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            model.eval()
            validation = evaluate(model, data['val'], times, scale)
            cf_validation = evaluate(model, cf_val_data, times, scale) if cf_val_data else None
            selection_score = (validation['ade_m'] if not cf_validation else
                               .5 * validation['ade_m'] + .5 * cf_validation['ade_m'])
            with (run / f'{label}_history.jsonl').open('a') as stream:
                stream.write(json.dumps({'epoch': epoch, 'validation': validation,
                                         'counterfactual_validation': cf_validation}) + '\n')
            if selection_score < best_score:
                best_score = selection_score
                torch.save({'model': model.state_dict(), 'config': config, 'splits': splits,
                            'dataset_sha256': fingerprints, 'epoch': epoch,
                            'validation': validation, 'use_action': conditioned,
                            'training_support_counts': reports['train']['valid_horizon_windows'],
                            'counterfactual_sha256': cf_hashes,
                            'counterfactual_validation': cf_validation},
                           run / f'{label}_best.pt')
            if epoch % 25 == 0:
                print(f'{label} epoch={epoch} val_ADE={validation["ade_m"]:.4f}m', flush=True)
        best = torch.load(run / f'{label}_best.pt', map_location=device, weights_only=True)
        model.load_state_dict(best['model'])
        model.eval()
        results[label] = {'epoch': best['epoch'], 'validation': best['validation'],
                          'counterfactual_validation': best.get('counterfactual_validation'),
                          'test': evaluate(model, data['test'], times, scale)}
        if conditioned:
            shuffled = list(data['test'])
            shuffled[1] = shuffled[1].roll(4, dims=0)
            results[label]['test_wrong_episode_action'] = evaluate(model, tuple(shuffled), times, scale)
    save_json(run / 'metrics.json', {
        'results': results, 'device': str(device),
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
        'scope': 'expert selected left hits; labels masked before next right contact',
        'timing': 'targets relative to true contact for supervision; runtime uses predicted contact delay',
    })
    print(json.dumps({name: {'epoch': result['epoch'], 'test_ADE_m': result['test']['ade_m']}
                      for name, result in results.items()}), flush=True)


if __name__ == '__main__':
    main()
