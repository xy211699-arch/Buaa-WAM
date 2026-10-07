"""Official collection entry for the finalized dual-arm air-hockey expert.

Run: bash collect_data.sh air_hockey air_hockey_clean 0
The two official phases search/cache the expert, then replay it without search.
This task supplies its own procedural scene, not Base_Task's generic table.
"""
import gc
import hashlib
import json
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from ._base_task import Base_Task
from ._air_hockey import expert, runtime
from .utils.pkl2hdf5 import process_folder_to_hdf5_video


class air_hockey(Base_Task):
    PLAN_VERSION = 1

    def setup_demo(self, now_ep_num=0, seed=0, **kwargs):
        self.ep_num, self.seed = int(now_ep_num), int(seed)
        self.save_dir = str(Path(kwargs.get('save_path', './data')).resolve())
        self.need_plan = bool(kwargs.get('need_plan', True))
        self.save_data = bool(kwargs.get('save_data', False)) and not self.need_plan
        self.save_freq = int(kwargs.get('save_freq', 30))
        self.video_fps = int(kwargs.get('video_fps', self.save_freq))
        if self.save_freq <= 0 or self.video_fps <= 0:
            raise ValueError('Sampling rates must be positive integers')
        self.capture_fps = math.lcm(self.save_freq, self.video_fps)
        if self.capture_fps > 1000:
            raise ValueError('Sampling rates exceed the 1000-Hz physics clock')
        if kwargs.get('render_freq', 0):
            raise ValueError('Official air_hockey collection requires render_freq: 0')
        self._config = Path(kwargs.get('air_hockey_config',
                                      expert.ROOT / 'env_cfg/air_hockey_assets.yml'))
        if not self._config.is_absolute():
            self._config = expert.ROOT / self._config
        self._config_hash = hashlib.sha256(self._config.read_bytes()).hexdigest()
        self._asset_config = expert.load_asset_config(self._config)
        self._coarse_only = bool(kwargs.get('expert_coarse_only', True))
        self._verify = bool(kwargs.get('expert_verify', False))
        self.FRAME_IDX = 0
        self.plan_success = False
        self._success = False
        self._summary, self._cached_plan, self._obs = {}, None, None
        self._records, self._metadata = [], {}
        self._video_frames, self._video_times = [], []
        self._next_state_time = self._next_video_time = 0.0
        self._last_state_time = -1.0
        self._joint_names = {}
        self._joint_bounds, self._indices = {}, {}
        self.left_joint_path, self.right_joint_path = [], []

    @staticmethod
    def _pose(pose):
        # SAPIEN and this repository use quaternion order w,x,y,z.
        return np.asarray(pose['p'] + pose['q'], dtype=np.float32)

    def _capture(self, state, record, rgb):
        if not self._indices:
            arm_names = set(self._asset_config['robot_preview']['arm_joint_names'])
            for side in ('left', 'right'):
                robot = state['assembly'][side + '_robot']
                joints = robot.get_active_joints()
                arm = [i for i, joint in enumerate(joints) if joint.name in arm_names]
                fingers = [i for i in range(len(joints)) if i not in arm]
                if len(arm) != 6 or len(fingers) != 2:
                    raise ValueError('Expected the configured UR5-WSG 6+2 joint layout')
                self._indices[side] = (arm, fingers)
                self._joint_names[side] = [joint.name for joint in joints]
                self._joint_bounds[side] = np.asarray(robot.get_qlimits())[arm]
        state_due, video_due = self._sample_due(record['episode_time'],
                                                record.get('terminal', False))
        if video_due and self.save_data:
            self._video_frames.append(rgb.copy())
            self._video_times.append(record['episode_time'])
        if not state_due:
            return
        self._records.append(record)
        if not self.save_data:
            return
        joint_action = {}
        for side, (arm, fingers) in self._indices.items():
            q = np.asarray(record[side + '_robot']['q'], dtype=np.float32)
            joint_action[side + '_arm'] = q[arm]
            # Preserve both physical finger joints (meters), not a fabricated
            # normalized gripper action. The strikers stay gripped throughout.
            joint_action[side + '_gripper'] = q[fingers]
        camera = state['camera']
        self._obs = {
            'observation': {'head_camera': {
                'rgb': rgb,
                'intrinsic_cv': np.asarray(camera.get_intrinsic_matrix()),
                'cam2world_gl': np.asarray(camera.get_model_matrix()),
            }},
            'joint_action': joint_action,
            'endpose': {'left_endpose': self._pose(record['tcp']),
                        'right_endpose': self._pose(record['right_tcp'])},
        }
        self._take_picture()

    def _sample_due(self, t, terminal=False):
        """Select exact independent sampling grids, plus one terminal state."""
        state_due = t + 1e-9 >= self._next_state_time
        video_due = t + 1e-9 >= self._next_video_time
        if state_due:
            self._next_state_time = (math.floor((t + 1e-9) * self.save_freq) + 1) / self.save_freq
        if video_due:
            self._next_video_time = (math.floor((t + 1e-9) * self.video_fps) + 1) / self.video_fps
        state_due = state_due or (terminal and t > self._last_state_time + 1e-9)
        if state_due:
            self._last_state_time = t
        return state_due, video_due

    def get_obs(self):
        if self._obs is None:
            raise RuntimeError('No synchronized frame is available yet')
        return self._obs

    def play_once(self):
        if not self.need_plan and self._cached_plan is None:
            raise RuntimeError('Load the saved expert plan before the recording phase')
        phase = 'plan' if self.need_plan else 'replay'
        record_dir = (Path(self.save_dir) / '_expert' /
                      f'episode{self.ep_num}_seed{self.seed}_{phase}')
        args = SimpleNamespace(seed=self.seed, config=self._config,
            coarse_only=self._coarse_only, verify=self._verify,
            verify_top_k=12, verify_repeats=3, render_search=False,
            collect=True, headless=True, hold=0,
            capture_fps=self.capture_fps,
            summary_path=record_dir / 'summary.json')
        self._summary = expert.run_episode(args, replay_plan=self._cached_plan,
                                          frame_sink=self._capture, record_dir=record_dir)
        self._metadata = json.loads((record_dir / 'meta.json').read_text())
        self._metadata.update(state_hz=self.save_freq, video_fps=self.video_fps,
                              state_count=len(self._records), video_frame_count=len(self._video_times))
        for side in ('left', 'right'):
            indices = self._indices[side][0]
            positions = np.asarray([r['action'][side + '_joint_drive_target']
                                    for r in self._records], dtype=float)[:, indices]
            setattr(self, side + '_joint_path', [{'position': positions}])
        contacts = {c['arm']: c for c in self._metadata.get('contacts', [])}
        right = contacts.get('right')
        self._success = bool(self._summary.get('episode_success') and
            'left' in contacts and right is not None and
            abs(self._metadata['episode_time'] - right['episode_time'] - 1.0) < 0.0011)
        self.plan_success = self._success
        if not self.need_plan and not self._success:
            raise RuntimeError('Cached expert replay failed; no episode will be exported')
        return {'info': {}, 'seed': self.seed,
                'actual_event': self._summary.get('actual_event'),
                'episode_success': self._success,
                'contacts': self._metadata.get('contacts', []),
                'post_right_contact_seconds': 1.0}

    def check_success(self):
        return self._success

    def planned_joints_legal(self):
        # UR5 hardware limits are explicit ±2π. These are not the generic
        # curobo placeholder/wrap limits used by Base_Task for other robots.
        excess = 0.0
        for side in ('left', 'right'):
            path = getattr(self, side + '_joint_path', [])
            bounds = self._joint_bounds.get(side)
            if not path or bounds is None:
                return False, float('inf')
            for segment in path:
                q = np.asarray(segment['position'], dtype=float)
                if q.ndim != 2 or q.shape[1] != len(bounds) or not len(q) or not np.isfinite(q).all():
                    return False, float('inf')
                excess = max(excess, float(np.maximum(bounds[:, 0] - q, q - bounds[:, 1]).max()))
        return excess <= 1e-6, excess

    def save_traj_data(self, idx):
        plan = {'left': self._summary['final_best'],
                'right': self._summary['actual_execution']['right_return']['selected_search_candidate']}
        data = {'version': self.PLAN_VERSION, 'seed': self.seed,
                'config_sha256': self._config_hash, 'expert_plan': plan,
                'left_joint_path': self.left_joint_path,
                'right_joint_path': self.right_joint_path}
        path = Path(self.save_dir) / '_traj_data' / f'episode{idx}.pkl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream:
            pickle.dump(data, stream)

    def load_tran_data(self, idx):
        data = super().load_tran_data(idx)
        if (data.get('version') != self.PLAN_VERSION or data.get('seed') != self.seed
                or data.get('config_sha256') != self._config_hash):
            raise ValueError('Cached air_hockey plan does not match this seed/config/version')
        self._cached_plan = data['expert_plan']
        return data

    def set_path_lst(self, args):
        self.left_joint_path = args['left_joint_path']
        self.right_joint_path = args['right_joint_path']

    def close_env(self, clear_cache=False):
        runtime.reset_runtime()
        self._obs = None
        gc.collect()
        if clear_cache:
            from sapien.render import clear_cache as clear_render_cache
            clear_render_cache()

    def merge_pkl_to_hdf5_video(self, instructions=None):
        if not self.save_data:
            return
        if not self.check_success():
            raise RuntimeError('Refusing to export a failed air_hockey episode')
        root = Path(self.save_dir)
        name = f'episode_{self.ep_num:07d}'
        (root / 'data').mkdir(parents=True, exist_ok=True)
        (root / 'video').mkdir(parents=True, exist_ok=True)
        h5_path = root / 'data' / (name + '.hdf5.partial')
        video_path = root / 'video' / (name + '.partial.mp4')
        # Use the same native writer as Base_Task. Publish only after all task
        # metadata is written, so resume cannot mistake a partial HDF5 for done.
        process_folder_to_hdf5_video(self.folder_path['cache'], str(h5_path), str(video_path),
                                    instructions=instructions, frequency=self.save_freq,
                                    video_frames=self._video_frames or None, video_fps=self.video_fps)
        with h5py.File(h5_path, 'a') as file:
            file.attrs['task_name'] = 'air_hockey'
            file.attrs['seed'] = self.seed
            file.attrs['action_semantics'] = 'next measured joint state, per repository native writer'
            file.attrs['gripper_semantics'] = 'two physical UR5-WSG finger positions in meters'
            file.attrs['quaternion_order'] = 'wxyz'
            file.attrs['state_hz'] = self.save_freq
            file.attrs['video_fps'] = self.video_fps
            episode = file.create_group('episode')
            episode.create_dataset('timestamps', data=[r['episode_time'] for r in self._records])
            episode.create_dataset('metadata_json', data=json.dumps(self._metadata))
            episode.create_dataset('expert_summary_json', data=json.dumps(self._summary))
            video = file.create_group('video')
            video.create_dataset('timestamps', data=self._video_times)
            video.attrs['path'] = '../video/' + name + '.mp4'
            video.attrs['fps'] = self.video_fps
            # Full observations preserve the terminal state; native state/action
            # below still retain the repository's N-1 transition convention.
            observations = file.create_group('observations')
            observations.create_dataset('timestamps', data=[r['episode_time'] for r in self._records])
            observations.attrs['nominal_hz'] = self.save_freq
            observations.attrs['terminal_sample'] = 'final interval may be shorter than 1/state_hz'
            for side in ('left', 'right'):
                arm = observations.create_group(side)
                arm.create_dataset('joint_names', data=self._joint_names[side], dtype=h5py.string_dtype())
                arm.create_dataset('joint_positions', data=[r[side + '_robot']['q'] for r in self._records])
                arm.create_dataset('joint_velocities', data=[r[side + '_robot']['dq'] for r in self._records])
                pose_key = 'tcp' if side == 'left' else 'right_tcp'
                arm.create_dataset('eef_pose', data=[self._pose(r[pose_key]) for r in self._records])
                arm.attrs['eef_pose_layout'] = 'x,y,z,qw,qx,qy,qz; world frame; meters'
                arm.attrs['joint_units'] = 'revolute: rad, rad/s; prismatic fingers: m, m/s'
            controls = file.create_group('control')
            controls.attrs['semantics'] = 'drive targets sampled at observation time, aligned with state'
            for side in ('left', 'right'):
                controls.create_dataset(side + '_joint_drive_target', data=np.asarray([
                    r['action'][side + '_joint_drive_target'] for r in self._records[:-1]]))
            task_state = file.create_group('task_state')
            task_state.create_dataset('timestamps', data=[r['episode_time'] for r in self._records[:-1]])
            task_state.create_dataset('next_timestamps', data=[r['episode_time'] for r in self._records[1:]])
            task_state.create_dataset('puck_pose', data=[self._pose(r['puck_gt']['pose']) for r in self._records[:-1]])
            task_state.create_dataset('puck_linear_velocity', data=[r['puck_gt']['linear_velocity'] for r in self._records[:-1]])
            task_state.create_dataset('phase', data=[r['phase'] for r in self._records[:-1]],
                                      dtype=h5py.string_dtype('utf-8'))
        video_path.replace(root / 'video' / (name + '.mp4'))
        h5_path.replace(root / 'data' / (name + '.hdf5'))
