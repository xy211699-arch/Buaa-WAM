"""Small training-only same-state command set with full decision-relative tail."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from envs._air_hockey import expert, runtime
from scripts.validate_hit_pipeline import capture_warm_start
from world_model.pipeline import HitPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-checkpoint', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--actions', choices=('expert', 'mixed'), default='mixed')
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError('Positive episodes required')
    torch.set_num_threads(4)
    pipeline = HitPipeline(args.policy_checkpoint, device='cuda')
    run = ROOT / 'outputs/joint_branch_data' / datetime.now().strftime('run_%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    files = pipeline.splits['train'][:args.episodes]
    config_hash = hashlib.sha256((ROOT / 'env_cfg/air_hockey_assets.yml').read_bytes()).hexdigest()
    (run / 'manifest.json').write_text(json.dumps({'schema': 'joint_full_decision_tail_v2', 'split': 'train',
        'contact_definition': 'physical_contact_during_commanded_strike',
        'files': files, 'policy_checkpoint': str(args.policy_checkpoint), 'asset_config_sha256': config_hash,
        'horizon_s': 1., 'physics_timestep_s': .001,
        'actions': args.actions,
        'scope': 'training-only expert-prepositioned snapshots; privileged expert data generation'}, indent=2))
    print(f'RUN_DIR={run}', flush=True)
    reports = []
    original_step = expert.physics_step
    perturbations = [(0., 0., 1.), (-.02, 0., 1.), (.02, 0., 1.),
                     (0., -8., .9), (0., 8., 1.1), (-.02, 8., 1.1),
                     (.02, -8., .9), (0., 0., .8)]
    for filename in files:
        warm = capture_warm_start(pipeline, filename, run)
        capture, selected = warm['capture'], warm['selected']
        context = dict(capture['kwargs'], initial_state=capture['snapshot'], viewer=None, render=False,
                       after_strike_handler=None, post_contact_seconds=1.)
        runtime._EP_REC['paused'] = True
        reference = expert.build_physx_puck_reference(scene=context['scene'], timestep=context['timestep'],
            assembly=context['assembly'], initial_state=capture['snapshot'], puck=context['assembly']['puck'],
            puck_body=context['puck_body'], horizon=1.)
        names = [joint.name for joint in context['left_active_joints']]
        proposal = pipeline.propose(np.stack(warm['frames'][-9:]), capture['robot']['q'],
                                    capture['robot']['dq'], capture['target'], names)
        for index, (delay_delta, angle_delta, multiplier) in enumerate(perturbations):
            intention = {key: selected.get(key, default) for key, default in (
                ('goal_target_y', 0.), ('desired_out_speed', 1.35), ('attack_mode', 'DIRECT'),
                ('direction_correction_deg', 0.), ('speed_scale', 1.), ('command_lead_s', 0.))}
            intention['contact_time'] = selected['contact_time'] - capture['decision_s'] + delay_delta
            if args.actions == 'mixed':
                mixtures = [('expert', 0., 0., 1.), ('policy', 0., 0., 1.),
                            ('expert', -.02, 8., 1.1), ('expert', .02, -8., .9),
                            ('policy', -.04, 8., 1.2), ('policy', .04, -8., .8),
                            ('expert', -.06, 0., 1.3), ('policy', 0., 0., 1.3)]
                source, delay_delta, angle_delta, multiplier = mixtures[index]
                if source == 'policy':
                    intention = dict(proposal['intentions'][0])
                    intention['contact_time'] += delay_delta
                else:
                    intention['contact_time'] = selected['contact_time'] - capture['decision_s'] + delay_delta
            intention['direction_correction_deg'] += angle_delta
            intention['speed_scale'] *= multiplier
            identifier = f'execution_joint_seed_{warm["seed"]}_candidate_{index}'
            with (run / 'commands_before_execution.jsonl').open('a') as stream:
                stream.write(json.dumps({'id': identifier, 'seed': warm['seed'], 'source_filename': filename,
                    'decision_s': capture['decision_s'], 'intention_relative_to_decision': intention}) + '\n')
            branch = dict(context, puck_reference=reference, **intention)
            plan = expert.evaluate_candidate(**branch, plan_only=True)
            item = {'seed': warm['seed'], 'candidate': index, 'prepared': bool(plan.get('valid')),
                    'reason': plan.get('reason'), 'physical_contact': False}
            if plan.get('valid'):
                trajectory, contacts, elapsed = [], [], [0.]

                def record_step(scene):
                    original_step(scene)
                    elapsed[0] += context['timestep']
                    trajectory.append([elapsed[0], *context['assembly']['puck'].get_pose().p[:2]])
                    if not contacts and expert.has_physical_contact(scene, context['assembly']['puck'], context['left_striker']):
                        contacts.append(elapsed[0])

                expert.physics_step = record_step
                try:
                    actual = expert.evaluate_candidate(**branch)
                    # Keep the same command/drive targets; observe coast after a
                    # failed stroke instead of fabricating or truncating future.
                    while elapsed[0] < 1. - .5 * context['timestep']:
                        expert.physics_step(context['scene'])
                finally:
                    expert.physics_step = original_step
                command = dict(plan, contact_time=plan['contact_time'] + capture['decision_s'],
                               prep_time=plan['prep_time'] + capture['decision_s'])
                destination = run / f'{identifier}.npz'
                with tempfile.NamedTemporaryFile(dir=run, suffix='.npz', delete=False) as stream:
                    temporary = Path(stream.name)
                    np.savez_compressed(stream, rgb=np.stack(warm['frames'][-9:]),
                        actual_trajectory=np.asarray(trajectory), plan_json=json.dumps(command),
                        decision_s=capture['decision_s'],
                        actual_contact_s=actual.get('physical_contact_time') if actual.get('physical_contact_time') is not None else np.nan,
                        observed_contact_s=contacts[0] if contacts else np.nan,
                        actual_result_json=json.dumps(actual),
                        source_filename=filename, contact=actual.get('physical_contact_time') is not None, q=capture['robot']['q'],
                        dq=capture['robot']['dq'], target=capture['target'], joint_names=names,
                        execution_valid=bool(actual.get('valid')), execution_reason=str(actual.get('reason')))
                os.replace(temporary, destination)
                item.update(reason=actual.get('reason'), physical_contact=actual.get('physical_contact_time') is not None,
                            observed_contact=bool(contacts),
                            execution_valid=bool(actual.get('valid')), observed_seconds=elapsed[0],
                            file=destination.name)
            reports.append(item)
            (run / 'reports.json').write_text(json.dumps(reports, indent=2))
        runtime.reset_runtime()
        print(json.dumps({'seed': warm['seed'], 'commands': len(reports),
                          'recorded': sum(item['prepared'] for item in reports),
                          'contacts': sum(item['physical_contact'] for item in reports)}), flush=True)
    (run / 'summary.json').write_text(json.dumps({'states': len(files), 'commands': len(reports),
        'recorded': sum(item['prepared'] for item in reports),
        'contacts': sum(item['physical_contact'] for item in reports)}, indent=2))
    print(f'COMPLETE={run}', flush=True)


if __name__ == '__main__':
    main()
