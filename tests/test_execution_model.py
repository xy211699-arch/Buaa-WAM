"""Robot-state semantics and failed-contact supervision regression checks."""

import unittest
import numpy as np
import torch

from world_model.execution import ExecutionPredictor, execution_loss, robot_features


class ExecutionTests(unittest.TestCase):
    def test_joint_names_determine_state_order(self):
        names = ['finger'] + [f'j{i}' for i in range(6)]
        arm = [f'j{i}' for i in range(6)]
        q = np.arange(7, dtype=np.float32)
        result = robot_features(q, q * 2, q + 1, names, arm)
        np.testing.assert_allclose(result[:6], q[1:] / np.pi)
        np.testing.assert_allclose(result[6:12], q[1:])
        np.testing.assert_allclose(result[12:], np.ones(6) / np.pi)
        with self.assertRaises(ValueError):
            robot_features(q, q, q, names, arm[:-1])

    def test_miss_samples_do_not_train_trajectory_or_contact_time(self):
        forecast = torch.randn(2, 20, 2, requires_grad=True)
        delay = torch.randn(2, requires_grad=True)
        logit = torch.zeros(2, requires_grad=True)
        result = {'future_xy': forecast, 'contact_delay_s': delay, 'contact_logit': logit}
        loss = execution_loss(result, torch.zeros_like(forecast), torch.zeros(2, 20, dtype=torch.bool),
                              torch.zeros(2), torch.zeros(2))
        loss.backward()
        self.assertTrue(torch.all(forecast.grad == 0))
        self.assertTrue(torch.all(delay.grad == 0))
        self.assertTrue(torch.all(logit.grad > 0))

    def test_state_changes_forecast_and_ablation_ignores_state(self):
        config = {'backbone_config': {'model_type': 'spatial', 'image_to_world': np.eye(3).tolist(),
                                     'hidden_dim': 64, 'dt_s': .05, 'horizon_steps': 20},
                  'dt_s': .05, 'history_frames': 9}
        model = ExecutionPredictor(config)
        features, robot, action = torch.randn(2, 58), torch.randn(2, 18), torch.randn(2, 6)
        times = torch.arange(1, 21) * .05
        original = model(features, robot, action, times)
        changed = model(features, robot + 1, action, times)
        self.assertFalse(torch.allclose(original['future_xy'], changed['future_xy']))
        model.use_robot = False
        self.assertTrue(torch.equal(model(features, robot, action, times)['future_xy'],
                                    model(features, robot + 1, action, times)['future_xy']))


if __name__ == '__main__':
    unittest.main()
