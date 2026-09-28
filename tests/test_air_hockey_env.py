"""Collection contract tests; GPU integration is exercised with collect_data.sh."""
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np


class AirHockeyCollectionTests(unittest.TestCase):
    def task(self):
        self.assertIsNotNone(importlib.util.find_spec('envs.air_hockey'),
                             'official envs.air_hockey entry is missing')
        from envs.air_hockey import air_hockey
        return air_hockey()

    def test_official_entry_and_cached_plan_roundtrip(self):
        task = self.task()
        with tempfile.TemporaryDirectory() as directory:
            task.setup_demo(seed=6, now_ep_num=0, save_path=directory)
            task._summary = {'final_best': {'contact_time': 2.5},
                'actual_execution': {'right_return': {
                    'selected_search_candidate': {'contact_time': 0.8}}}}
            task.left_joint_path = [{'position': np.zeros((2, 6))}]
            task.right_joint_path = [{'position': np.ones((2, 6))}]
            task.save_traj_data(0)
            loaded = task.load_tran_data(0)
            self.assertEqual(loaded['expert_plan']['left']['contact_time'], 2.5)
            self.assertEqual(loaded['expert_plan']['right']['contact_time'], 0.8)
            np.testing.assert_array_equal(loaded['right_joint_path'][0]['position'], 1)

    def test_empty_and_nonfinite_paths_are_not_legal(self):
        task = self.task()
        task._joint_bounds = {side: np.tile([-6.28, 6.28], (6, 1))
                              for side in ('left', 'right')}
        task.left_joint_path = []
        task.right_joint_path = []
        self.assertFalse(task.planned_joints_legal()[0])
        task.left_joint_path = [{'position': np.zeros((2, 6))}]
        task.right_joint_path = [{'position': np.full((2, 6), np.nan)}]
        self.assertFalse(task.planned_joints_legal()[0])
        task.right_joint_path = [{'position': np.full((2, 6), 3.5)}]
        self.assertTrue(task.planned_joints_legal()[0])
        task.right_joint_path = [{'position': np.full((2, 6), 6.5)}]
        self.assertFalse(task.planned_joints_legal()[0])

    def test_legacy_modules_share_canonical_implementation(self):
        self.task()
        from envs._air_hockey import expert, runtime
        import scripts.plan_direct_counterattack_stage5_v1_5 as legacy_expert
        import scripts.collect_episode_rgb_auto_v1 as legacy_runtime
        self.assertIs(legacy_runtime, runtime)
        self.assertIs(legacy_expert, expert)

    def test_native_export_pairs_states_and_preserves_terminal_time(self):
        # A one-row timing shift, loss of drive targets, or dropping the final
        # contact+1s timestamp must fail even when the HDF5 file is readable.
        task = self.task()
        with tempfile.TemporaryDirectory() as directory:
            task.setup_demo(seed=6, now_ep_num=0, save_path=directory,
                            need_plan=False, save_data=True)
            task._success = True
            task._metadata = {'contacts': [{'arm': 'right', 'episode_time': 0.05}],
                              'episode_time': 1.05}
            task._summary = {'episode_success': True}
            task._joint_names = {side: ['joint' + str(i) for i in range(8)]
                                 for side in ('left', 'right')}
            for i, timestamp in enumerate([0.0, 0.033, 1.05]):
                task._obs = {'observation': {'head_camera': {
                    'rgb': np.full((32, 32, 3), i * 50, dtype=np.uint8)}},
                    'joint_action': {'left_arm': np.full(6, i),
                                     'right_arm': np.full(6, i + 10),
                                     'left_gripper': np.zeros(2),
                                     'right_gripper': np.zeros(2)}}
                task._take_picture()
                task._records.append({'episode_time': timestamp, 'phase': 'FLIGHT',
                    'left_robot': {'q': [i] * 8, 'dq': [0.1] * 8},
                    'right_robot': {'q': [i + 10] * 8, 'dq': [0.2] * 8},
                    'tcp': {'p': [0, 0, 0], 'q': [1, 0, 0, 0]},
                    'right_tcp': {'p': [1, 0, 0], 'q': [1, 0, 0, 0]},
                    'action': {'left_joint_drive_target': [20 + i] * 8,
                               'right_joint_drive_target': [30 + i] * 8},
                    'puck_gt': {'pose': {'p': [0, 0, 0], 'q': [1, 0, 0, 0]},
                                'linear_velocity': [1, 0, 0]}})
            task.merge_pkl_to_hdf5_video(instructions=['Return the puck.'])
            path = Path(directory) / 'data/episode_0000000.hdf5'
            with h5py.File(path, 'r') as file:
                np.testing.assert_array_equal(file['state/left_arm_joint_states'][:, 0], [0, 1])
                np.testing.assert_array_equal(file['action/left_arm_joint_states'][:, 0], [1, 2])
                np.testing.assert_array_equal(file['control/left_joint_drive_target'][:, 0], [20, 21])
                np.testing.assert_allclose(file['task_state/next_timestamps'][:], [0.033, 1.05])
                self.assertEqual(file['episode/timestamps'][-1], 1.05)
                self.assertEqual(list(file['vision']), ['cam_head'])
                self.assertEqual(len(file['vision/cam_head/colors']), 2)
                self.assertEqual(file['observations/left/joint_positions'].shape, (3, 8))
                np.testing.assert_array_equal(file['observations/left/joint_positions'][-1], [2] * 8)
                np.testing.assert_allclose(file['observations/right/eef_pose'][0], [1, 0, 0, 1, 0, 0, 0])
            self.assertTrue((Path(directory) / 'video/episode_0000000.mp4').is_file())

    def test_finish_does_not_duplicate_an_already_captured_terminal_frame(self):
        self.task()
        from envs._air_hockey import runtime
        frames = [1.0]
        state = {'started': True, 'finished': False, 'paused': False,
                 'episode_time': 1.0, 'last_capture_time': 1.0,
                 'frames_fp': io.StringIO(), 'frame_index': 1,
                 'planner_steps': 0, 'out_dir': '/unused'}
        # Rendering/file I/O is irrelevant here; model its observable frame
        # append while exercising the actual finish decision and state changes.
        with patch.object(runtime, '_EP_REC', state), \
             patch.object(runtime, '_ep_write_meta'), \
             patch.object(runtime, '_ep_capture_frame',
                          side_effect=lambda **kw: frames.append(state['episode_time'])):
            runtime.episode_finish('RIGHT_RETURN')
        self.assertEqual(frames, [1.0])
        self.assertTrue(state['finished'])


if __name__ == '__main__':
    unittest.main()
