"""Train measured-state and state-ablated execution models on separate seed groups."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from world_model.execution import ExecutionPredictor, execution_loss
from world_model.execution_data import native_samples, branch_samples


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


@torch.no_grad()
def cache(model, samples, batch_size, device):
    columns = [[] for _ in range(7)]
    keys = ('rgb', 'robot', 'action', 'future_xy', 'mask', 'contact_delay_s', 'contact')
    tensors = [{key: sample[key] for key in keys} for sample in samples]
    for batch in DataLoader(tensors, batch_size=batch_size):
        values = [model.visual.encode_rgb(batch['rgb'].to(device))] + [batch[key].to(device)
                  for key in ('robot', 'action', 'future_xy', 'mask', 'contact_delay_s', 'contact')]
        for column, value in zip(columns, values):
            column.append(value)
    return tuple(torch.cat(column) for column in columns)


@torch.no_grad()
def evaluate(model, data, times, scale):
    features, robot, actions, target, mask, delay, contact = data
    result = model(features, robot, actions, times)
    valid = mask.clone()
    valid[:, 10:] = False
    errors = torch.linalg.vector_norm((result['future_xy'] - target) * scale, dim=-1)
    probability = result['contact_logit'].sigmoid()
    predicted = probability >= .5
    positive = contact > .5
    negative = ~positive
    return {'samples': len(features), 'positive': int(positive.sum()), 'negative': int(negative.sum()),
        'ade_0_5s_m': float(errors[valid].mean()),
        'point_p95_m': float(torch.quantile(errors[valid], .95)),
        'contact_brier': float((probability - contact).square().mean()),
        'miss_recall': float((~predicted[negative]).float().mean()) if negative.any() else None,
        'contact_recall': float(predicted[positive].float().mean()),
        'contact_delay_mae_s': float((result['contact_delay_s'][positive] - delay[positive]).abs().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for key in ('batch_size', 'epochs', 'cpu_threads'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'Invalid {key}')
    backbone = torch.load(ROOT / config['backbone_checkpoint'], map_location='cpu', weights_only=True)
    config['backbone_config'] = backbone['config']
    config['image_to_world'] = backbone['config']['image_to_world']
    asset = yaml.safe_load((ROOT / 'env_cfg/air_hockey_assets.yml').read_text())
    config['arm_joint_names'] = asset['robot_preview']['arm_joint_names']
    splits = backbone['splits']
    root = ROOT / config['data_root']
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for names in splits.values() for name in names}
    if hashes != backbone['dataset_sha256']:
        raise ValueError('Native dataset changed')
    device = torch.device(args.device)
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(config['training_seed'])
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
    run = ROOT / config['output_root'] / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    write(run / 'config.json', config)
    write(run / 'splits.json', splits)
    print(f'RUN_DIR={run}', flush=True)
    samples = {name: native_samples(root, files, config) for name, files in splits.items()}
    branches = {name: branch_samples(ROOT / config[f'{name}_branch_root'], name, splits[name], config)
                for name in ('train', 'val', 'test')}
    samples['train'] += branches['train']
    encoder = ExecutionPredictor(config).to(device).eval()
    encoder.visual.backbone.load_state_dict(backbone['model'])
    data = {name: cache(encoder, value, config['batch_size'], device) for name, value in samples.items()}
    branch_data = {name: cache(encoder, value, config['batch_size'], device) for name, value in branches.items()}
    del encoder
    # Encoder/control dimensions have very different scales; fit on train only.
    robot_mean = data['train'][1].mean(0)
    robot_std = data['train'][1].std(0).clamp_min(1e-4)
    write(run / 'robot_normalization.json', {'mean': robot_mean.cpu().tolist(),
                                           'std': robot_std.cpu().tolist()})
    branch_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for name in ('train', 'val', 'test')
                     for path in (ROOT / config[f'{name}_branch_root']).glob('execution_*.npz')}
    write(run / 'branch_sha256.json', branch_hashes)
    times = torch.arange(1, config['horizon_steps'] + 1, device=device) * config['dt_s']
    scale = torch.tensor(config['xy_scale_m'], device=device)
    report = {}
    for label, use_robot in [('robot_state', True), ('no_robot_state', False)]:
        torch.manual_seed(config['training_seed'])
        model = ExecutionPredictor(config, use_robot).to(device)
        model.visual.backbone.load_state_dict(backbone['model'])
        model.robot_mean.copy_(robot_mean)
        model.robot_std.copy_(robot_std)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=config['learning_rate'], weight_decay=config['weight_decay'])
        best = float('inf')
        for epoch in range(1, config['epochs'] + 1):
            model.train()
            for index in torch.randperm(len(data['train'][0]), device=device).split(config['batch_size']):
                features, robot, actions, target, mask, delay, contact = [column[index] for column in data['train']]
                optimizer.zero_grad(set_to_none=True)
                loss = execution_loss(model(features, robot, actions, times), target, mask, delay, contact)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            model.eval()
            native = evaluate(model, data['val'], times, scale)
            branch = evaluate(model, branch_data['val'], times, scale)
            score = .5 * native['ade_0_5s_m'] + .5 * branch['ade_0_5s_m'] + .03 * branch['contact_brier']
            with (run / f'{label}_history.jsonl').open('a') as stream:
                stream.write(json.dumps({'epoch': epoch, 'native': native, 'branch': branch}) + '\n')
            if score < best:
                best = score
                torch.save({'model': model.state_dict(), 'config': config, 'splits': splits,
                            'epoch': epoch, 'use_robot': use_robot, 'dataset_sha256': hashes,
                            'branch_sha256': branch_hashes, 'validation': {'native': native, 'branch': branch}},
                           run / f'{label}_best.pt')
            if epoch % 50 == 0:
                print(label, epoch, round(branch['ade_0_5s_m'], 4), flush=True)
        checkpoint = torch.load(run / f'{label}_best.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        report[label] = {'epoch': checkpoint['epoch'], 'validation': checkpoint['validation'],
                         'native_test': evaluate(model, data['test'], times, scale),
                         'branch_test': evaluate(model, branch_data['test'], times, scale)}
    write(run / 'metrics.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
