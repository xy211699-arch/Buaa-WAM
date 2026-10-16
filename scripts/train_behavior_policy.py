"""Fit a high-level BC policy; select checkpoints on validation seeds only."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import torch
from torch.utils.data import DataLoader

from world_model.execution import ExecutionPredictor
from world_model.execution_data import native_samples
from world_model.policy import BehaviorPolicy, expert_intention


@torch.no_grad()
def prepare(encoder, root, files, config, device):
    samples = native_samples(root, files, config)
    plans = {}
    for name in files:
        with h5py.File(root / name) as file:
            plans[name] = json.loads(file['episode/expert_summary_json'][()])['final_best']
    rows = []
    for sample in samples:
        label, bank = expert_intention(plans[sample['filename']], sample['decision_s'])
        rows.append({'rgb': sample['rgb'], 'robot': sample['robot'],
                     'label': torch.tensor(label), 'bank': torch.tensor(bank)})
    columns = [[] for _ in range(4)]
    for batch in DataLoader(rows, batch_size=32):
        values = [encoder.visual.encode_rgb(batch['rgb'].to(device))] + [
            batch[key].to(device) for key in ('robot', 'label', 'bank')]
        for column, value in zip(columns, values):
            column.append(value)
    return tuple(torch.cat(column) for column in columns)


@torch.no_grad()
def evaluate(model, data):
    features, robot, target, bank = data
    values, logit = model(features, robot)
    return {'samples': len(features), 'mae': (values - target).abs().mean(0).cpu().tolist(),
            'bank_accuracy': float(((logit >= 0) == (bank >= .5)).float().mean()),
            'selection_loss': float(((values - target) / model.output_std).square().mean()
                                   + torch.nn.functional.binary_cross_entropy_with_logits(logit, bank))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-checkpoint', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError('Positive epoch count required')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device(args.device)
    checkpoint = torch.load(args.world_checkpoint, map_location='cpu', weights_only=True)
    config, splits = checkpoint['config'], checkpoint['splits']
    root = ROOT / config['data_root']
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for names in splits.values() for name in names}
    if hashes != checkpoint['dataset_sha256']:
        raise ValueError('Dataset changed')
    encoder = ExecutionPredictor(config, checkpoint['use_robot']).to(device).eval()
    encoder.load_state_dict(checkpoint['model'])
    data = {key: prepare(encoder, root, files, config, device) for key, files in splits.items()}
    run = ROOT / 'outputs/behavior_policy' / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    print(f'RUN_DIR={run}', flush=True)
    model = BehaviorPolicy(data['train'][0].shape[1]).to(device)
    inputs = torch.cat(data['train'][:2], -1)
    model.input_mean.copy_(inputs.mean(0))
    model.input_std.copy_(inputs.std(0).clamp_min(.01))
    model.output_mean.copy_(data['train'][2].mean(0))
    model.output_std.copy_(data['train'][2].std(0).clamp_min(.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        for index in torch.randperm(len(inputs), device=device).split(32):
            features, robot, target, bank = [column[index] for column in data['train']]
            values, logit = model(features, robot)
            loss = ((values - target) / model.output_std).square().mean() + torch.nn.functional.binary_cross_entropy_with_logits(logit, bank)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        metrics = evaluate(model, data['val'])
        if metrics['selection_loss'] < best:
            best = metrics['selection_loss']
            torch.save({'model': model.state_dict(), 'feature_dim': data['train'][0].shape[1],
                        'world_checkpoint': str(args.world_checkpoint.resolve()),
                        'world_checkpoint_sha256': hashlib.sha256(args.world_checkpoint.read_bytes()).hexdigest(),
                        'splits': splits, 'dataset_sha256': hashes, 'epoch': epoch,
                        'validation': metrics, 'training': {'seed': 42, 'epochs': args.epochs,
                        'learning_rate': .001, 'weight_decay': .001}}, run / 'best.pt')
        if epoch % 50 == 0:
            print(epoch, round(metrics['selection_loss'], 4), flush=True)
    best_checkpoint = torch.load(run / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint['model'])
    report = {'epoch': best_checkpoint['epoch'], 'validation': best_checkpoint['validation'],
              'test': evaluate(model, data['test']),
              'mae_units': ['s', 'm', 'm/s', 'degrees/30', 'ratio', 's'],
              'scope': 'expert-prepared pre-strike decisions; no RL; no success guarantee'}
    (run / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
