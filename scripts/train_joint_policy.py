"""Matched direct-policy / joint / stop-gradient experiments on fixed seed groups."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from world_model.joint_data import load_native, load_branches
from world_model.joint_model import JointHitModel, masked_xy_loss
from world_model.model import build_model


@torch.no_grad()
def cache(vision, rows, batch_size, device):
    keys = ('robot', 'intention', 'bank', 'future_xy', 'mask', 'expert', 'contact', 'contact_delay_s')
    data = {key: [] for key in ('visual',) + keys}
    for batch in DataLoader([{key: row[key] for key in ('rgb',) + keys} for row in rows], batch_size=batch_size):
        data['visual'].append(vision.encode_observation(batch['rgb'].to(device))['tokens'])
        for key in keys:
            data[key].append(batch[key].to(device))
    return {key: torch.cat(value) for key, value in data.items()}


def objective(model, data, indices, noise, tau, times, config, variant):
    visual, robot = data['visual'][indices], data['robot'][indices]
    z = model.encode_present(visual, robot)
    expert = data['expert'][indices]
    normalized = (data['intention'][indices] - model.action_mean) / model.action_std
    velocity = model.velocity(z, (1 - tau[:, None]) * normalized + tau[:, None] * noise, tau)
    point = (velocity - (noise - normalized)).square().mean(-1)
    action = ((point + F.binary_cross_entropy_with_logits(model.mode_head(z).squeeze(-1), data['bank'][indices], reduction='none')) * expert).sum() / expert.sum().clamp_min(1)
    if variant == 'action_only':
        return action
    detached = variant == 'stop_gradient'
    current = visual[:, -1, 24:26]
    future = model.future(z, current, times, detach_context=detached)
    dynamics = model.future(z, current, times, normalized, detach_context=detached)
    target, mask = data['future_xy'][indices], data['mask'][indices]
    future_loss = masked_xy_loss(future['xy'], target, mask & expert[:, None])
    dynamics_loss = masked_xy_loss(dynamics['xy'], target, mask)
    contact = data['contact'][indices]
    classified = F.binary_cross_entropy_with_logits(dynamics['contact_logit'], contact, reduction='none')
    positive = contact > .5
    classification = classified.mean() if positive.all() or not positive.any() else .5 * classified[positive].mean() + .5 * classified[~positive].mean()
    timing = ((dynamics['contact_delay_s'][positive] - data['contact_delay_s'][indices][positive]).square().mean()
              if positive.any() else dynamics['contact_delay_s'].sum() * 0.)
    return action + config['future_weight'] * future_loss + config['dynamics_weight'] * dynamics_loss + config['contact_weight'] * (classification + timing)


@torch.no_grad()
def evaluate(model, data, times, config, noise, tau):
    model.eval()
    indices = torch.arange(len(data['visual']), device=times.device)
    expert = data['expert']
    z = model.encode_present(data['visual'], data['robot'])
    normalized = (data['intention'] - model.action_mean) / model.action_std
    flow = model.velocity(z, (1 - tau[:, None]) * normalized + tau[:, None] * noise, tau)
    action_loss = ((flow - (noise - normalized)).square().mean(-1)[expert].mean()
                   + F.binary_cross_entropy_with_logits(model.mode_head(z).squeeze(-1)[expert], data['bank'][expert]))
    sampled, _ = model.policy(data['visual'][expert], data['robot'][expert], noise[expert], config['sampling_steps'])
    future = model.future(z, data['visual'][:, -1, 24:26], times)
    dynamics = model.future(z, data['visual'][:, -1, 24:26], times, normalized)
    scale = times.new_tensor(config['xy_scale_m'])
    mask = data['mask']
    errors = torch.linalg.vector_norm((dynamics['xy'] - data['future_xy']) * scale, dim=-1)
    expert_errors = torch.linalg.vector_norm((future['xy'] - data['future_xy']) * scale, dim=-1)
    pos = data['contact'] > .5
    prediction = dynamics['contact_logit'] >= 0
    return {'action_validation_loss': float(action_loss),
            'expert_windows': int(expert.sum()), 'branch_windows': int((~expert).sum()),
            'action_mae': (sampled - data['intention'][expert]).abs().mean(0).cpu().tolist(),
            'future_ade_m': float(expert_errors[mask & expert[:, None]].mean()),
            'dynamics_ade_m': float(errors[mask].mean()),
            'contact_recall': float(prediction[pos].float().mean()),
            'miss_recall': float((~prediction[~pos]).float().mean()) if (~pos).any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'env_cfg/joint_air_hockey.yml')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    experiment = yaml.safe_load(args.config.read_text())
    for key in ('batch_size', 'epochs', 'context_layers', 'cpu_threads', 'sampling_steps'):
        if not isinstance(experiment[key], int) or experiment[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if experiment['context_width'] % 4 or experiment['context_width'] <= 0:
        raise ValueError('context_width must be positive and divisible by four')
    if any(experiment[key] < 0 for key in ('future_weight', 'dynamics_weight', 'contact_weight')) or experiment['learning_rate'] <= 0:
        raise ValueError('Invalid loss weights/learning rate')
    seeds = experiment['training_seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError('Distinct nonnegative training seeds required')
    torch.set_num_threads(experiment['cpu_threads'])
    device = torch.device(args.device)
    source_path = ROOT / experiment['source_checkpoint']
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    config = {**source['config'], **experiment}
    config['gradient_clip_scheme'] = 'separate_shared_auxiliary_v1'
    splits = source['splits']
    root = ROOT / config['data_root']
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for names in splits.values() for name in names}
    if hashes != source['dataset_sha256']:
        raise ValueError('Native dataset changed')
    visual_path = ROOT / config['backbone_checkpoint']
    visual_checkpoint = torch.load(visual_path, map_location='cpu', weights_only=True)
    vision = build_model(visual_checkpoint['config']).to(device).eval()
    if not hasattr(vision, 'encode_observation'):
        raise ValueError('Spatial observation encoder required')
    vision.load_state_dict(visual_checkpoint['model'])
    for parameter in vision.parameters():
        parameter.requires_grad_(False)
    run = ROOT / config['output_root'] / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    print(f'RUN_DIR={run}', flush=True)
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    (run / 'splits.json').write_text(json.dumps(splits, indent=2))
    data = {}
    for split, files in splits.items():
        rows = load_native(root, files, config)
        rows += load_branches(ROOT / config[f'{split}_branch_root'], files, split, config)
        if split == 'train' and config.get('extra_train_branch_root'):
            rows += load_branches(ROOT / config['extra_train_branch_root'], files, split, config)
        print(f'CACHE {split} {len(rows)} windows', flush=True)
        data[split] = cache(vision, rows, config['batch_size'], device)
        del rows
    del vision
    train = data['train']
    stats = {'visual_mean': train['visual'].mean((0, 1)), 'visual_std': train['visual'].std((0, 1)).clamp_min(.01),
             'robot_mean': train['robot'].mean(0), 'robot_std': train['robot'].std(0).clamp_min(.01),
             'action_mean': train['intention'][train['expert']].mean(0),
             'action_std': train['intention'][train['expert']].std(0).clamp_min(.01)}
    times = torch.arange(1, config['horizon_steps'] + 1, device=device) * config['dt_s']
    generator = torch.Generator().manual_seed(991)
    fixed = {split: (torch.randn(len(d['visual']), 6, generator=generator).to(device),
                     torch.rand(len(d['visual']), generator=generator).to(device)) for split, d in data.items()}
    branch_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for split in splits for p in (ROOT / config[f'{split}_branch_root']).glob('execution_*.npz')}
    if config.get('extra_train_branch_root'):
        branch_hashes.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (ROOT / config['extra_train_branch_root']).glob('execution_*.npz')})
    report = []
    for seed in config['training_seeds']:
        for variant in ('action_only', 'joint', 'stop_gradient'):
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            model = JointHitModel(config).to(device)
            auxiliary_prefixes = ('future_head.', 'dynamics_context.', 'dynamics_head.', 'contact_head.')
            shared_parameters = [p for name, p in model.named_parameters() if not name.startswith(auxiliary_prefixes)]
            auxiliary_parameters = [p for name, p in model.named_parameters() if name.startswith(auxiliary_prefixes)]
            for key, value in stats.items():
                getattr(model, key).copy_(value)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            best, start = float('inf'), time.perf_counter()
            path = run / f'{variant}_seed{seed}_best.pt'
            for epoch in range(1, config['epochs'] + 1):
                model.train()
                for indices in torch.randperm(len(train['visual']), device=device).split(config['batch_size']):
                    noise = torch.randn(len(indices), 6, device=device)
                    tau = torch.rand(len(indices), device=device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = objective(model, train, indices, noise, tau, times, config, variant)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    loss.backward()
                    # Auxiliary-only gradients must not rescale action gradients
                    # in the stop-gradient control through global clipping.
                    torch.nn.utils.clip_grad_norm_(shared_parameters, 1.)
                    torch.nn.utils.clip_grad_norm_(auxiliary_parameters, 1.)
                    optimizer.step()
                metrics = evaluate(model, data['val'], times, config, *fixed['val'])
                score = metrics['action_validation_loss']
                with (run / f'{variant}_seed{seed}_history.jsonl').open('a') as stream:
                    stream.write(json.dumps({'epoch': epoch, **metrics}) + '\n')
                if score < best:
                    best = score
                    temporary = path.with_suffix('.pt.tmp')
                    torch.save({'model': model.state_dict(), 'config': config, 'splits': splits,
                                'epoch': epoch, 'variant': variant, 'training_seed': seed,
                                'validation': metrics, 'dataset_sha256': hashes, 'branch_sha256': branch_hashes,
                                'visual_checkpoint': str(visual_path.resolve()),
                                'visual_checkpoint_sha256': hashlib.sha256(visual_path.read_bytes()).hexdigest()}, temporary)
                    temporary.replace(path)
                if epoch % 20 == 0:
                    print(variant, seed, epoch, round(score, 4), flush=True)
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model'])
            metrics = evaluate(model, data['test'], times, config, *fixed['test'])
            item = {'variant': variant, 'training_seed': seed, 'epoch': checkpoint['epoch'],
                    'validation': checkpoint['validation'], 'test': metrics,
                    'trainable_parameters': sum(p.numel() for p in model.parameters()),
                    'seconds': time.perf_counter() - start,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else None}
            report.append(item)
            (run / 'metrics.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(item), flush=True)
    print(f'COMPLETE={run}', flush=True)


if __name__ == '__main__':
    main()
