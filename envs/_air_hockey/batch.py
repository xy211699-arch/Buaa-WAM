"""Resumable node15--24 collection: 20-Hz state, 50-FPS overhead video.

Each worker owns ten global episode IDs and a disjoint strided seed stream.
Uses the official env's plan/cache/replay/export protocol, one isolated child
process per attempt. Only audited HDF5+MP4 pairs count toward the quota.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback


def seed_for_attempt(node, attempt):
    if not 15 <= node <= 24 or attempt < 0:
        raise ValueError('Expected node15--24 and a nonnegative attempt')
    return node - 15 + 10 * attempt


def episode_ids(node, count):
    return range((node - 15) * count, (node - 14) * count)


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def stop_child(child):
    """Stop and reap only this worker's own isolated attempt process group."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def audit_episode(root, episode, node=None):
    import cv2
    import h5py
    import numpy as np
    name = f'episode_{episode:07d}'
    with h5py.File(root / 'data' / (name + '.hdf5')) as file:
        assert file.attrs['state_hz'] == 20 and file.attrs['video_fps'] == 50
        seed = int(file.attrs['seed'])
        if node is not None:
            assert seed % 10 == node - 15
        ts = file['observations/timestamps'][:]
        vt = file['video/timestamps'][:]
        assert len(ts) >= 2 and len(vt) >= 2 and np.all(np.diff(ts) > 0)
        assert np.allclose(np.diff(ts)[:-1], .05, atol=1e-7)
        assert 0 < ts[-1] - ts[-2] <= .050001
        assert np.allclose(np.diff(vt), .02, atol=1e-7)
        assert abs(ts[0]) < 1e-8 and abs(vt[0]) < 1e-8
        for side in ('left', 'right'):
            for field in ('joint_positions', 'joint_velocities'):
                values = file[f'observations/{side}/{field}'][:]
                assert values.shape == (len(ts), 8) and np.isfinite(values).all()
            pose = file[f'observations/{side}/eef_pose'][:]
            assert pose.shape == (len(ts), 7) and np.isfinite(pose).all()
            assert np.allclose(np.linalg.norm(pose[:, 3:], axis=1), 1, atol=1e-5)
            assert len(file[f'observations/{side}/joint_names']) == 8
        metadata = json.loads(file['episode/metadata_json'][()])
        contacts = {c['arm']: c['episode_time'] for c in metadata['contacts']}
        assert 'left' in contacts and 'right' in contacts
        assert abs(ts[-1] - contacts['right'] - 1) < .0011
        assert json.loads(file['episode/expert_summary_json'][()])['episode_success']
        assert list(file['vision']) == ['cam_head']
    video = cv2.VideoCapture(str(root / 'video' / (name + '.mp4')))
    try:
        assert video.isOpened() and video.get(cv2.CAP_PROP_FPS) == 50
        assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == len(vt)
        assert (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (640, 360)
        ok, _ = video.read()
        assert ok
    finally:
        video.release()
    return dict(episode=episode, seed=seed, state_count=len(ts), video_frames=len(vt),
                duration=float(ts[-1]))


def collect_one(root, node, episode, seed):
    from envs.air_hockey import air_hockey
    task = air_hockey()
    common = dict(now_ep_num=episode, seed=seed, save_path=str(root),
                  save_freq=20, video_fps=50, expert_coarse_only=True)
    try:
        task.setup_demo(**common, need_plan=True)
        task.play_once()
        if not task.plan_success or not task.check_success() or not task.planned_joints_legal()[0]:
            return 1
        task.save_traj_data(episode)
    finally:
        task.close_env()
    try:
        task.setup_demo(**common, need_plan=False, save_data=True)
        task.set_path_lst(task.load_tran_data(episode))
        info = task.play_once()
        task.close_env()
        instruction = 'Hit the puck with the left arm, then return it with the right arm.'
        task.merge_pkl_to_hdf5_video(instructions=[instruction])
        task.remove_data_cache()
        atomic_json(root / 'instruction' / f'episode_{episode:07d}.json',
                    {'seen': [instruction], 'unseen': [instruction]})
        atomic_json(root / 'scene_info' / f'episode_{episode:07d}.json', info)
    finally:
        task.close_env()
    try:
        audit = audit_episode(root, episode, node)
    except Exception:
        traceback.print_exc()
        return 20  # A schema/integrity failure must stop the worker, not retry forever.
    print('EPISODE AUDIT PASS', json.dumps(audit), flush=True)
    return 0


def run_worker(args):
    root = args.root.resolve()
    worker = root / '_workers' / f'node{args.node}'
    worker.mkdir(parents=True, exist_ok=True)
    lock = (worker / 'worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This node already has an active worker for this dataset')
    path = worker / 'status.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    attempt = int(previous.get('next_attempt', 0))
    status = dict(node=args.node, hostname=socket.gethostname(), pid=os.getpid(),
                  target=args.count, completed=0, state='starting', next_attempt=attempt,
                  state_hz=20, video_fps=50, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    completed = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'worker received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    def publish(**changes):
        status.update(changes, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                      completed=len(completed), episodes=completed)
        atomic_json(path, status)
    publish()
    for episode in episode_ids(args.node, args.count):
        existing = root / 'data' / f'episode_{episode:07d}.hdf5'
        if existing.exists():
            completed.append(audit_episode(root, episode, args.node))
            publish()
            continue
        while attempt < args.max_attempts:
            seed = seed_for_attempt(args.node, attempt)
            attempt += 1
            log = worker / f'episode{episode:07d}_seed{seed:06d}.log'
            command = [sys.executable, '-u', '-m', 'envs._air_hockey.batch',
                       '--node', str(args.node), '--root', str(root),
                       '--single', '--episode', str(episode), '--seed', str(seed)]
            with log.open('w') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=(lock.fileno(),))
                try:
                    publish(state='running', episode=episode, seed=seed, child_pid=child.pid,
                            next_attempt=attempt, log=str(log))
                    code = child.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    stop_child(child)
                    code = 124
                except BaseException:
                    stop_child(child)
                    publish(state='interrupted', child_pid=None)
                    raise
            if code == 0:
                completed.append(audit_episode(root, episode, args.node))
                publish(state='running', child_pid=None, last_exit=code)
                break
            publish(state='retrying' if code != 20 else 'failed_integrity',
                    child_pid=None, last_exit=code)
            if code == 20:
                raise SystemExit('Integrity validation failed; inspect the recorded log')
        else:
            publish(state='attempt_limit_reached')
            raise SystemExit('Attempt cap reached before quota; inspect failed-seed logs')
    publish(state='complete', child_pid=None)
    print(f'node{args.node}: {len(completed)}/{args.count} complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', type=int, required=True, choices=range(15, 25))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--max-attempts', type=int, default=300)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--episode', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.count != 10:
        parser.error('This run is partitioned into exactly 10 episodes per node')
    if args.single:
        if args.episode is None or args.seed is None:
            parser.error('--single requires --episode and --seed')
        raise SystemExit(collect_one(args.root.resolve(), args.node, args.episode, args.seed))
    run_worker(args)


if __name__ == '__main__':
    main()
