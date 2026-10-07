import importlib.util
import unittest
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import patch


class BatchTests(unittest.TestCase):
    def test_ten_node_seed_streams_never_overlap(self):
        self.assertIsNotNone(importlib.util.find_spec('envs._air_hockey.batch'),
                             'distributed batch launcher is missing')
        from envs._air_hockey.batch import seed_for_attempt, episode_ids
        streams = [{seed_for_attempt(node, k) for k in range(300)} for node in range(15, 25)]
        self.assertEqual(len(set.union(*streams)), 3000)
        self.assertEqual([seed_for_attempt(15, 0), seed_for_attempt(24, 0),
                          seed_for_attempt(15, 1)], [0, 9, 10])
        ids = [i for n in range(15, 25) for i in episode_ids(n, 10)]
        self.assertEqual(ids, list(range(100)))

    def test_child_shutdown_reaps_the_process(self):
        from envs._air_hockey import batch
        self.assertTrue(hasattr(batch, 'stop_child'), 'worker has no child cleanup')
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],
                                 start_new_session=True)
        try:
            batch.stop_child(child)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_video_encoding_without_system_ffmpeg(self):
        import numpy as np
        import cv2
        from envs.utils.images_to_video import images_to_video
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'PATH': directory}):
            target = str(Path(directory) / 'test.mp4')
            images_to_video(np.zeros((3, 32, 32, 3), dtype=np.uint8), target, fps=50)
            video = cv2.VideoCapture(target)
            self.assertTrue(video.isOpened())
            self.assertEqual(video.get(cv2.CAP_PROP_FPS), 50)
            video.release()


if __name__ == '__main__':
    unittest.main()
