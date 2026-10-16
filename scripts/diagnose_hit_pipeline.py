"""Paired snapshot ablations to distinguish RGB reference, policy and servo timing.

Oracle references/expert labels appear only in diagnostic branches, never in
policy inference. Each branch starts from the same preserved physical snapshot.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from envs._air_hockey import expert, runtime
from scripts.validate_hit_pipeline import capture_warm_start
from world_model.diagnostics import decompose_xy_error
from world_model.pipeline import HitPipeline


def diagnose_one(pipeline, filename, run, components_only=False):
    warm = capture_warm_start(pipeline, filename, run)
    capture, frames = warm['capture'], warm['frames']
    context = dict(capture['kwargs'])
    context.update(initial_state=capture['snapshot'], viewer=None, render=False,
                   after_strike_handler=None, post_contact_seconds=1.)
    runtime._EP_REC['paused'] = True
    names = [joint.name for joint in context['left_active_joints']]
    proposal = pipeline.propose(np.stack(frames[-9:]), capture['robot']['q'],
                                capture['robot']['dq'], capture['target'], names)
    oracle = expert.build_physx_puck_reference(scene=context['scene'], timestep=context['timestep'],
        assembly=context['assembly'], initial_state=capture['snapshot'], puck=context['assembly']['puck'],
        puck_body=context['puck_body'], horizon=1.)
    prepared, rejected = [], []
    for index, intention in enumerate(proposal['intentions']):
        plan = expert.evaluate_candidate(**dict(context, puck_reference=proposal['reference'], **intention), plan_only=True)
        if plan.get('valid') and plan['prep_time'] >= pipeline.config['contact_guard_s']:
            prepared.append((index, plan))
        else:
            rejected.append({'candidate_index': index, 'reason': plan.get('reason')})
    ranked = pipeline.rank(proposal, prepared)
    chosen = ranked[0]['candidate_index'] if ranked else 0
    ranked_intention = proposal['intentions'][chosen]
    expert_intention = {key: warm['selected'].get(key, default) for key, default in (
        ('contact_time', 0.), ('goal_target_y', 0.), ('desired_out_speed', 1.35),
        ('attack_mode', 'DIRECT'), ('direction_correction_deg', 0.),
        ('speed_scale', 1.), ('command_lead_s', 0.))}
    expert_intention['contact_time'] -= capture['decision_s']
    variants = [
        ('rgb_ranked', proposal['reference'], dict(ranked_intention)),
        ('rgb_bc_base', proposal['reference'], dict(proposal['intentions'][0])),
        ('oracle_ranked_diagnostic', oracle, dict(ranked_intention)),
        ('rgb_ranked_lead20ms', proposal['reference'], dict(ranked_intention, command_lead_s=ranked_intention['command_lead_s'] + .02)),
        ('rgb_ranked_lead40ms', proposal['reference'], dict(ranked_intention, command_lead_s=ranked_intention['command_lead_s'] + .04)),
        ('rgb_expert_intention_diagnostic', proposal['reference'], expert_intention),
        ('oracle_expert_intention_diagnostic', oracle, expert_intention),
    ]
    if components_only:
        base = proposal['intentions'][0]
        variants = [
            ('rgb_bc_base', proposal['reference'], dict(base)),
            ('rgb_bc_expert_time_diagnostic', proposal['reference'],
             dict(base, contact_time=expert_intention['contact_time'])),
            ('rgb_bc_expert_speed_diagnostic', proposal['reference'],
             dict(base, desired_out_speed=expert_intention['desired_out_speed'],
                  speed_scale=expert_intention['speed_scale'])),
            ('rgb_bc_expert_direction_diagnostic', proposal['reference'],
             dict(base, goal_target_y=expert_intention['goal_target_y'],
                  direction_correction_deg=expert_intention['direction_correction_deg'],
                  attack_mode=expert_intention['attack_mode'])),
            ('rgb_expert_intention_diagnostic', proposal['reference'], expert_intention),
        ]
    results = []
    original_step = expert.physics_step
    for label, reference, intention in variants:
        branch = dict(context, puck_reference=reference, **intention)
        plan = expert.evaluate_candidate(**branch, plan_only=True)
        if not plan.get('valid'):
            results.append({'variant': label, 'physical_contact': False, 'reason': plan['reason'], 'prepared': False})
            continue
        predicted = expert.reference_state_at_time(reference, intention['contact_time'])
        truth = expert.reference_state_at_time(oracle, intention['contact_time'])
        error = decompose_xy_error(predicted['xy'], truth['xy'], truth['vxy'])
        trajectory, elapsed = [], [0.]

        def record_step(scene):
            original_step(scene)
            elapsed[0] += branch['timestep']
            puck = branch['assembly']['puck'].get_pose().p
            striker = branch['left_striker'].get_pose().p
            trajectory.append([elapsed[0], *puck[:2], *striker[:3]])

        expert.physics_step = record_step
        try:
            actual = expert.evaluate_candidate(**branch)
        finally:
            expert.physics_step = original_step
        measured = np.asarray(trajectory)
        separation = np.linalg.norm(measured[:, 1:3] - measured[:, 3:5], axis=1)
        closest = int(np.argmin(separation))
        nominal = min(int(np.searchsorted(measured[:, 0], intention['contact_time'])), len(measured) - 1)
        radius_sum = branch['puck_radius'] + branch['striker_radius']
        physical = actual.get('physical_contact_time') is not None
        results.append({'variant': label, 'prepared': True, 'physical_contact': physical,
            'valid': bool(actual.get('valid')), 'reason': actual.get('reason'),
            'nominal_contact_s': intention['contact_time'],
            'physical_contact_s': actual.get('physical_contact_time'),
            'contact_lag_s': actual['physical_contact_time'] - intention['contact_time'] if physical else None,
            'reference_error': error, 'closest_distance_m': float(separation[closest]),
            'closest_clearance_m': float(separation[closest] - radius_sum),
            'closest_time_s': float(measured[closest, 0]),
            'nominal_separation_m': float(separation[nominal]),
            'nominal_striker_xy_m': measured[nominal, 3:5].tolist(),
            'nominal_actual_puck_xy_m': measured[nominal, 1:3].tolist(),
            'planned_striker_contact_xy_m': (predicted['xy'] - radius_sum * np.array([
                np.cos(np.deg2rad(plan['nominal_theta_deg'] + plan['direction_correction_deg'])),
                np.sin(np.deg2rad(plan['nominal_theta_deg'] + plan['direction_correction_deg']))])).tolist(),
            'plan': plan})
        np.savez_compressed(warm['episode_dir'] / f'{label}.npz', trajectory=measured,
                            columns=np.array(['time_s', 'puck_x_m', 'puck_y_m', 'striker_x_m', 'striker_y_m', 'striker_z_m']))
        (warm['episode_dir'] / 'diagnosis.json').write_text(json.dumps(results, indent=2))
    runtime.reset_runtime()
    return {'seed': warm['seed'], 'filename': filename, 'selected_candidate': chosen if ranked else None,
            'ranked_candidate_available': bool(ranked), 'initial_rejected': rejected,
            'decision_s': capture['decision_s'], 'policy_intentions': proposal['intentions'],
            'branches': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--episode-offset', type=int, default=0)
    parser.add_argument('--components-only', action='store_true',
                        help='Compare BC time, speed and direction using expert labels only for diagnosis')
    args = parser.parse_args()
    if args.episodes <= 0 or args.episode_offset < 0:
        raise ValueError('Positive episodes required')
    torch.set_num_threads(4)
    pipeline = HitPipeline(args.policy_checkpoint, device='cuda')
    run = ROOT / 'outputs/hit_pipeline_diagnosis' / datetime.now().strftime('run_%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    files = pipeline.splits[args.split][args.episode_offset:args.episode_offset + args.episodes]
    if not files:
        raise ValueError('No episodes in requested split range')
    (run / 'manifest.json').write_text(json.dumps({'split': args.split, 'files': files,
        'components_only': args.components_only,
        'policy_checkpoint': str(args.policy_checkpoint), 'world_checkpoint': str(pipeline.world_checkpoint)}, indent=2))
    print(f'RUN_DIR={run}', flush=True)
    reports = []
    for filename in files:
        reports.append(diagnose_one(pipeline, filename, run, args.components_only))
        (run / 'reports.json').write_text(json.dumps(reports, indent=2))
        print(json.dumps({'seed': reports[-1]['seed'], 'contacts': {item['variant']: item['physical_contact'] for item in reports[-1]['branches']}}), flush=True)
    variants = sorted({item['variant'] for report in reports for item in report['branches']})
    summary = {variant: {'contacts': sum(item['physical_contact'] for report in reports for item in report['branches'] if item['variant'] == variant),
                         'episodes': sum(item['variant'] == variant for report in reports for item in report['branches'])} for variant in variants}
    (run / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
