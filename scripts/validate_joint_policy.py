"""Paired actual left-contact evaluation; current warm start and RGB bridge fixed."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from envs._air_hockey import expert, runtime
from scripts.validate_hit_pipeline import capture_warm_start
from world_model.joint_inference import JointHitPipeline
from world_model.pipeline import HitPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--reference-policy-checkpoint', type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError('Positive episodes required')
    torch.set_num_threads(4)
    checkpoints = sorted(args.training_run.glob('*_best.pt'))
    if not checkpoints:
        raise ValueError('No checkpoints')
    models = {path.stem: JointHitPipeline(path, 'cuda') for path in checkpoints}
    first = next(iter(models.values()))
    experiment = json.loads((args.training_run / 'config.json').read_text())
    if len(checkpoints) != 3 * len(experiment['training_seeds']) or len(json.loads((args.training_run / 'metrics.json').read_text())) != len(checkpoints):
        raise ValueError('Training experiment not complete')
    if args.reference_policy_checkpoint:
        models['P2_BC_reference'] = HitPipeline(args.reference_policy_checkpoint, device='cuda')
    if any(model.splits != first.splits for model in models.values()):
        raise ValueError('Mismatched scene splits')
    files = first.splits[args.split][:args.episodes]
    run = ROOT / 'outputs/joint_evaluation' / datetime.now().strftime('run_%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    (run / 'manifest.json').write_text(json.dumps({'split': args.split, 'files': files,
        'checkpoints': [str(p.resolve()) for p in checkpoints], 'sampling_seed': 42,
        'scope': 'paired left-only contact; expert-prepositioned warm start; original RGB execution bridge'}, indent=2))
    print(f'RUN_DIR={run}', flush=True)
    reports = []
    for filename in files:
        warm = capture_warm_start(first, filename, run)
        capture, frames = warm['capture'], warm['frames']
        context = dict(capture['kwargs'], initial_state=capture['snapshot'], viewer=None, render=False,
                       after_strike_handler=None, post_contact_seconds=1.)
        runtime._EP_REC['paused'] = True
        names = [joint.name for joint in context['left_active_joints']]
        for label, model in models.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            proposal = model.propose(np.stack(frames[-9:]), capture['robot']['q'],
                capture['robot']['dq'], capture['target'], names)
            torch.cuda.synchronize()
            total_seconds = time.perf_counter() - start
            branch = dict(context, puck_reference=proposal['reference'], **proposal['intentions'][0])
            plan = expert.evaluate_candidate(**branch, plan_only=True)
            if plan.get('valid'):
                actual = expert.evaluate_candidate(**branch)
                physical = actual.get('physical_contact_time') is not None
            else:
                actual, physical = plan, False
            item = {'seed': warm['seed'], 'filename': filename, 'model': label,
                    'prepared': bool(plan.get('valid')), 'physical_contact': physical,
                    'reason': actual.get('reason'), 'actual': actual,
                    'intention': proposal['intentions'][0], 'policy_seconds': proposal.get('policy_seconds'),
                    'bridge_seconds': proposal.get('bridge_seconds'), 'total_proposal_seconds': total_seconds}
            reports.append(item)
            (run / 'reports.json').write_text(json.dumps(reports, indent=2))
        runtime.reset_runtime()
        summary = {label: {'episodes': sum(row['model'] == label for row in reports),
            'contacts': sum(row['physical_contact'] for row in reports if row['model'] == label),
            'rejected': sum(not row['prepared'] for row in reports if row['model'] == label)} for label in models}
        (run / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({'seed': warm['seed'], 'summary': summary}), flush=True)
    print(f'COMPLETE={run}', flush=True)


if __name__ == '__main__':
    main()
