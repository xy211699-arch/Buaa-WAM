"""Branch the canonical expert from identical pre-strike snapshots and compare H1.

This is a privileged simulation benchmark, not a deployable candidate planner.
No core execution or success logic is duplicated or relaxed.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch

from envs._air_hockey import expert, runtime
from world_model.hit_inference import HitTrajectoryPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'outputs/world_model_hit/run_20261011_112552_482301/action_best.pt')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError('episodes must be positive')
    predictor = HitTrajectoryPredictor(args.checkpoint, device='cuda')
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    run = ROOT / 'outputs/hit_counterfactual' / datetime.now().strftime('run_%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    original_evaluate = expert.evaluate_candidate
    original_step = expert.physics_step
    results = []
    # Fix acceptance thresholds before executing branches; no policy fitting unless passed.
    gate = {'maximum_0_5s_ade_m': .06, 'maximum_point_p95_m': .15,
            'minimum_contact_fraction': .75, 'minimum_episodes': 3}
    (run / 'gate.json').write_text(json.dumps(gate, indent=2))
    (run / 'manifest.json').write_text(json.dumps({'split': args.split,
        'checkpoint': str(args.checkpoint), 'files': checkpoint['splits'][args.split][:args.episodes]}, indent=2))
    for filename in checkpoint['splits'][args.split][:args.episodes]:
        with h5py.File(ROOT / predictor.config['data_root'] / filename) as file:
            seed = int(file.attrs['seed'])
            summary = json.loads(file['episode/expert_summary_json'][()])
        selected = summary['final_best']
        right = summary['actual_execution']['right_return']['selected_search_candidate']
        target_decision = selected['prep_time'] - .02 - .15
        frames, capture = [], {}

        def capture_kwargs(**kwargs):
            if kwargs.get('frame_sign', 1.) > 0:
                capture['kwargs'] = kwargs.copy()
            return original_evaluate(**kwargs)

        def sink(state, record, rgb):
            t = record['episode_time']
            if t <= target_decision + 1e-8:
                frames.append(rgb.copy())
                capture['decision_s'] = float(t)
                capture['snapshot'] = runtime.snapshot_hit_state(state['assembly'], preserve_drive_targets=True)
                capture['robot'] = record['left_robot']
                capture['target'] = record['action']['left_joint_drive_target']

        expert.evaluate_candidate = capture_kwargs
        record_dir = run / f'seed_{seed}_replay'
        episode_args = SimpleNamespace(seed=seed, config=ROOT / 'env_cfg/air_hockey_assets.yml',
            coarse_only=True, verify=False, verify_top_k=12, verify_repeats=3,
            render_search=False, collect=True, headless=True, hold=0, capture_fps=20,
            summary_path=record_dir / 'summary.json')
        try:
            expert.run_episode(episode_args, replay_plan={'left': selected, 'right': right},
                               frame_sink=sink, record_dir=record_dir)
        finally:
            expert.evaluate_candidate = original_evaluate
        if len(frames) < 9 or 'kwargs' not in capture:
            raise RuntimeError('Missing synchronized history or execution context')
        if runtime._EP_REC is not None:
            runtime._EP_REC['paused'] = True
        kwargs = capture['kwargs']
        decision = capture['decision_s']
        snapshot = capture['snapshot']
        history = np.stack(frames[-9:])
        reference = expert.build_physx_puck_reference(
            scene=kwargs['scene'], timestep=kwargs['timestep'], assembly=kwargs['assembly'],
            initial_state=snapshot, puck=kwargs['assembly']['puck'], puck_body=kwargs['puck_body'],
            horizon=1.0)
        for angle_delta, speed_multiplier in [(0., 1.), (-8., 1.), (8., 1.),
                                              (0., .9), (0., 1.1)]:
            branch = kwargs.copy()
            branch.update(initial_state=snapshot, puck_reference=reference,
                          contact_time=selected['contact_time'] - decision,
                          direction_correction_deg=selected['direction_correction_deg'] + angle_delta,
                          speed_scale=selected['speed_scale'] * speed_multiplier,
                          viewer=None, render=False, after_strike_handler=None,
                          post_contact_seconds=1.0)
            trajectory, elapsed = [], [0.]

            def step(scene):
                original_step(scene)
                elapsed[0] += kwargs['timestep']
                trajectory.append([elapsed[0], *kwargs['assembly']['puck'].get_pose().p[:2]])

            expert.physics_step = step
            try:
                actual = original_evaluate(**branch)
            finally:
                expert.physics_step = original_step
            item = {'seed': seed, 'angle_delta_deg': angle_delta, 'speed_multiplier': speed_multiplier,
                    'valid': bool(actual.get('valid')), 'reason': actual.get('reason'),
                    'physical_contact': actual.get('physical_contact_time') is not None,
                    'event': actual.get('event'), 'score': actual.get('score')}
            # Retain failed-contact commands and measured decision state as well.
            if all(key in actual for key in ('contact_time', 'prep_time', 'nominal_theta_deg', 'hit_speed')):
                command = actual.copy()
                command['contact_time'] += decision
                command['prep_time'] += decision
                np.savez_compressed(run / f'execution_seed_{seed}_angle_{angle_delta}_speed_{speed_multiplier}.npz',
                    rgb=history, actual_trajectory=np.asarray(trajectory), plan_json=json.dumps(command),
                    decision_s=decision, actual_contact_s=(actual.get('physical_contact_time')
                        if actual.get('physical_contact_time') is not None else np.nan),
                    source_filename=filename, contact=bool(item['physical_contact']),
                    q=capture['robot']['q'], dq=capture['robot']['dq'], target=capture['target'],
                    joint_names=[joint.name for joint in kwargs['left_active_joints']])
            if item['valid'] and item['physical_contact']:
                plan = actual.copy()
                plan['contact_time'] += decision
                plan['prep_time'] += decision
                prediction = predictor.predict(history, plan, decision)
                trajectory = np.asarray(trajectory)
                contact = actual['physical_contact_time']
                query = contact + np.arange(1, 21) * .05
                truth = np.column_stack([np.interp(query, trajectory[:, 0], trajectory[:, axis])
                                         for axis in (1, 2)])
                error = np.linalg.norm(prediction['xy_m'] - truth, axis=-1)
                item.update(ade_0_5s_m=float(error[:10].mean()), errors_0_5s_m=error[:10].tolist(),
                            ade_1s_m=float(error.mean()),
                            contact_time_error_s=abs(prediction['predicted_contact_time_s'] - decision - contact))
                np.savez_compressed(run / f'seed_{seed}_angle_{angle_delta}_speed_{speed_multiplier}.npz',
                                    rgb=history, actual_trajectory=trajectory,
                                    predicted_xy=prediction['xy_m'], plan_json=json.dumps(plan),
                                    decision_s=decision, actual_contact_s=contact,
                                    source_filename=filename)
            results.append(item)
            (run / 'branches.json').write_text(json.dumps(results, indent=2))
            print(json.dumps(item), flush=True)
        runtime.reset_runtime()
    valid = [item for item in results if 'ade_0_5s_m' in item]
    errors = [error for item in valid for error in item['errors_0_5s_m']]
    ade = float(np.mean(errors)) if errors else None
    p95 = float(np.percentile(errors, 95)) if errors else None
    fraction = len(valid) / len(results)
    passed = (len(set(item['seed'] for item in valid)) >= gate['minimum_episodes']
              and fraction >= gate['minimum_contact_fraction'] and ade is not None
              and ade <= gate['maximum_0_5s_ade_m'] and p95 <= gate['maximum_point_p95_m'])
    report = {'passed': passed, 'valid_contact_branches': len(valid), 'branches': len(results),
              'ade_0_5s_m': ade, 'point_p95_m': p95, 'contact_fraction': fraction,
              'gate': gate, 'scope': 'same pre-strike physical snapshot; privileged candidate construction; no right strike'}
    (run / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
