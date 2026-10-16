"""Warm-started left learned pipeline + canonical right expert, with real contacts.

Expert replay creates the scene and prepositioning snapshot; it is NOT a fully
learned reset-to-task policy. Left decisions use RGB predictions/robot encoders;
right return remains the privileged existing expert. No retry on physical outcomes.
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
from world_model.pipeline import HitPipeline, execution_candidates


def capture_warm_start(pipeline, filename, run):
    """Build the explicitly expert-prepositioned diagnostic initial condition."""
    with h5py.File(ROOT / pipeline.config['data_root'] / filename) as file:
        seed = int(file.attrs['seed'])
        summary = json.loads(file['episode/expert_summary_json'][()])
    selected = summary['final_best']
    right = summary['actual_execution']['right_return']['selected_search_candidate']
    target_decision = selected['prep_time'] - .02 - .15
    frames, capture = [], {}
    original_evaluate = expert.evaluate_candidate

    def capture_context(**kwargs):
        if kwargs.get('frame_sign', 1.) > 0:
            capture['kwargs'] = kwargs.copy()
        return original_evaluate(**kwargs)

    def sink(state, record, rgb):
        if record['episode_time'] <= target_decision + 1e-8:
            frames.append(rgb.copy())
            capture.update(snapshot=runtime.snapshot_hit_state(state['assembly'], preserve_drive_targets=True),
                           robot=record['left_robot'], target=record['action']['left_joint_drive_target'],
                           decision_s=float(record['episode_time']))

    episode_dir = run / f'seed_{seed}'
    episode_dir.mkdir()
    args = SimpleNamespace(seed=seed, config=ROOT / 'env_cfg/air_hockey_assets.yml',
        coarse_only=True, verify=False, verify_top_k=12, verify_repeats=3,
        render_search=False, collect=True, headless=True, hold=0, capture_fps=20,
        summary_path=episode_dir / 'warmup_summary.json')
    expert.evaluate_candidate = capture_context
    try:
        expert.run_episode(args, replay_plan={'left': selected, 'right': right},
                           frame_sink=sink, record_dir=episode_dir / 'expert_warmup')
    finally:
        expert.evaluate_candidate = original_evaluate
    if len(frames) < 9 or 'kwargs' not in capture:
        raise RuntimeError('Missing RGB history/robot execution context')
    return {'capture': capture, 'frames': frames, 'selected': selected,
            'episode_dir': episode_dir, 'args': args, 'seed': seed}


def run_one(pipeline, filename, run, selection='bc'):
    warm = capture_warm_start(pipeline, filename, run)
    capture, frames, selected = warm['capture'], warm['frames'], warm['selected']
    episode_dir, args, seed = warm['episode_dir'], warm['args'], warm['seed']
    original_evaluate = expert.evaluate_candidate
    context = capture['kwargs'].copy()
    context.update(initial_state=capture['snapshot'], viewer=None, render=False,
                   after_strike_handler=None, post_contact_seconds=1.)
    runtime._EP_REC['paused'] = True
    names = [joint.name for joint in context['left_active_joints']]
    proposal = pipeline.propose(np.stack(frames[-9:]), capture['robot']['q'],
                                capture['robot']['dq'], capture['target'], names)
    # Replace all privileged future references and original action intentions.
    context['puck_reference'] = proposal['reference']
    physics_steps = [0]
    original_step = expert.physics_step

    def count_step(scene):
        physics_steps[0] += 1
        original_step(scene)

    expert.physics_step = count_step
    prepared, rejected = [], []
    try:
        for index, intention in enumerate(proposal['intentions']):
            branch = dict(context)
            branch.update(intention)
            plan = original_evaluate(**branch, plan_only=True)
            if plan.get('valid') and plan['prep_time'] >= pipeline.config['contact_guard_s']:
                prepared.append((index, plan))
            else:
                rejected.append({'candidate_index': index, 'reason': plan.get('reason')})
    finally:
        expert.physics_step = original_step
    if physics_steps[0] != 0:
        raise RuntimeError('Dry planning advanced physics')
    ranked = pipeline.rank(proposal, prepared)
    executable = execution_candidates(ranked, selection)
    report = {'seed': seed, 'filename': filename, 'decision_s': capture['decision_s'],
              'raw_policy': proposal['raw_policy'], 'bank_logit': proposal['bank_logit'],
              'intentions': proposal['intentions'], 'ranked_candidates': ranked,
              'selection': selection,
              'rejected': rejected, 'planning_physics_steps': physics_steps[0],
              'left_prediction_input': 'RGB history and current robot encoders/drive target only',
              'scope': 'expert-prepositioned warm start; learned left policy; privileged right expert',
              'pipeline_completed': False, 'physical_left_contact': False,
              'physical_right_contact': False, 'task_success': False}
    if hasattr(pipeline, 'variant'):
        report.update(model_variant=pipeline.variant, sampling_seed=pipeline.sampling_seed,
                      policy_seconds=proposal['policy_seconds'],
                      execution_bridge_seconds=proposal['bridge_seconds'])
    np.savez_compressed(episode_dir / 'decision_inputs.npz', rgb=np.stack(frames[-9:]),
                        q=capture['robot']['q'], dq=capture['robot']['dq'], target=capture['target'],
                        joint_names=names, predicted_inbound_t=proposal['reference']['t'],
                        predicted_inbound_xy=proposal['reference']['xy'])
    if executable:
        selected_index = executable[0]['candidate_index']
        branch = dict(context)
        branch.update(proposal['intentions'][selected_index])
        branch['post_contact_seconds'] = None
        scene, assembly = branch['scene'], branch['assembly']
        config = expert.load_asset_config(args.config)
        config.setdefault('episode_camera', {})['fps'] = 20
        robot_cfg = config['robot_preview']
        left_ee = next(link.entity for link in assembly['left_robot'].get_links()
                       if link.get_name() == robot_cfg['ee_link_name'])
        runtime.restore_hit_state(assembly, capture['snapshot'])
        runtime.configure_episode_recorder(scene, config, assembly, left_ee,
                                           output_dir=episode_dir / 'policy_execution')
        runtime.episode_recorder_after_step(scene)
        branch['after_strike_handler'] = lambda result: expert.execute_right_return(
            scene=scene, timestep=branch['timestep'], assembly=assembly, config=config,
            puck_body=branch['puck_body'], coarse_only=True, verify=False,
            verify_top_k=12, verify_repeats=3, viewer=None, selected_candidate=None)
        actual = original_evaluate(**branch)
        runtime.episode_finish(actual.get('event', actual.get('reason', 'UNKNOWN')))
        metadata = json.loads((episode_dir / 'policy_execution/meta.json').read_text())
        contacts = {item['arm']: item['episode_time'] for item in metadata['contacts']}
        right_actual = actual.get('right_return', {}).get('actual_execution', {})
        left_contact = actual.get('physical_contact_time') is not None and 'left' in contacts
        right_contact = right_actual.get('physical_contact_time') is not None and 'right' in contacts
        tail_complete = right_contact and abs(metadata['episode_time'] - contacts['right'] - 1.) <= 2. * branch['timestep']
        report.update(selected_candidate=selected_index, actual=actual,
                      pipeline_completed=True,
                      physical_left_contact=left_contact, physical_right_contact=right_contact,
                      right_contact_tail_complete=tail_complete,
                      task_success=actual.get('event') == 'RIGHT_RETURN' and left_contact and tail_complete)
    (episode_dir / 'report.json').write_text(json.dumps(report, indent=2))
    runtime.reset_runtime()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument('--policy-checkpoint', type=Path)
    checkpoints.add_argument('--joint-checkpoint', type=Path)
    parser.add_argument('--sampling-seed', type=int, default=42)
    parser.add_argument('--world-checkpoint', type=Path)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--episode-offset', type=int, default=0)
    parser.add_argument('--selection', choices=('world-model', 'bc'), default='bc',
                        help='bc executes only the unperturbed policy action; predictions remain logged')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.episodes <= 0 or args.episode_offset < 0 or args.sampling_seed < 0:
        raise ValueError('Positive episode count required')
    torch.set_num_threads(4)
    if args.joint_checkpoint:
        if args.world_checkpoint:
            parser.error('Joint checkpoint binds its own visual checkpoint')
        from world_model.joint_inference import JointHitPipeline
        pipeline = JointHitPipeline(args.joint_checkpoint, args.device, args.sampling_seed)
    else:
        pipeline = HitPipeline(args.policy_checkpoint, args.world_checkpoint, args.device)
    run = ROOT / 'outputs/hit_pipeline' / datetime.now().strftime('run_%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    print(f'RUN_DIR={run}', flush=True)
    reports = []
    files = pipeline.splits['test'][args.episode_offset:args.episode_offset + args.episodes]
    if not files:
        raise ValueError('No episodes in requested split range')
    for filename in files:
        reports.append(run_one(pipeline, filename, run, args.selection))
        summary = {'episodes': len(reports),
                   'pipeline_completed': sum(item['pipeline_completed'] for item in reports),
                   'physical_left_contacts': sum(item['physical_left_contact'] for item in reports),
                   'dual_arm_task_successes': sum(item['task_success'] for item in reports),
                   'scope': 'warm-started left learned pipeline + right expert; no performance gate',
                   'selection': args.selection,
                   'policy_checkpoint': str(args.joint_checkpoint or args.policy_checkpoint),
                   'world_checkpoint': str(pipeline.world_checkpoint),
                   'seeds': [item['seed'] for item in reports]}
        (run / 'report.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
