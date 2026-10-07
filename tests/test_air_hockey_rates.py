import tempfile
import unittest
import numpy as np
from envs.air_hockey import air_hockey


class MultiRateTests(unittest.TestCase):
    def test_20hz_state_50hz_video_are_independent(self):
        task = air_hockey()
        with tempfile.TemporaryDirectory() as directory:
            try:
                task.setup_demo(save_path=directory, save_freq=20, video_fps=50)
            except ValueError as error:
                self.fail(f'20/50 Hz collection is unsupported: {error}')
            self.assertEqual(task.capture_fps, 100)
            state_times, video_times = [], []
            for tick in range(101):
                t = tick / 100
                state_due, video_due = task._sample_due(t)
                if state_due:
                    state_times.append(t)
                if video_due:
                    video_times.append(t)
            np.testing.assert_allclose(state_times, np.arange(21) / 20)
            np.testing.assert_allclose(video_times, np.arange(51) / 50)

    def test_terminal_state_preserved_without_extra_video_frame(self):
        task = air_hockey()
        with tempfile.TemporaryDirectory() as directory:
            try:
                task.setup_demo(save_path=directory, save_freq=20, video_fps=50)
            except ValueError as error:
                self.fail(str(error))
            for tick in range(101):
                task._sample_due(tick / 100)
            self.assertEqual(task._sample_due(1.007, terminal=True), (True, False))


if __name__ == '__main__':
    unittest.main()
