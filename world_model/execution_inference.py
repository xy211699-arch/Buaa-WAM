"""Deployment interface using RGB plus encoder measurements and planned commands."""

import cv2
import numpy as np
import torch

from .execution import ExecutionPredictor, robot_features
from .hit_data import planned_action


class ExecutionTrajectoryPredictor:
    def __init__(self, checkpoint_path, device='cpu'):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.config = checkpoint['config']
        self.device = torch.device(device)
        self.model = ExecutionPredictor(self.config, checkpoint['use_robot']).to(self.device).eval()
        self.model.load_state_dict(checkpoint['model'])
        self.times = torch.arange(1, self.config['horizon_steps'] + 1,
                                  device=self.device) * self.config['dt_s']
        self.scale = torch.tensor(self.config['xy_scale_m'], device=self.device)

    @torch.no_grad()
    def predict(self, rgb_history, plan, decision_s, q, dq, target, joint_names):
        """All measurements refer to the last frame; q rad, dq rad/s, targets rad.

        q/dq/target are current encoder/controller readings, not future states.
        Candidate times share decision_s clock. No physical-contact truth needed.
        """
        frames = np.asarray(rgb_history)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != self.config['history_frames'] or frames.shape[-1] != 3:
            raise ValueError('Expected chronological uint8 RGB history T,H,W,3')
        if not np.isfinite(decision_s) or decision_s > plan['prep_time'] - self.config['contact_guard_s']:
            raise ValueError('Decision must precede fast stroke')
        command = planned_action(plan, decision_s, self.config['xy_scale_m'])
        measured = robot_features(q, dq, target, joint_names, self.config['arm_joint_names'])
        rgb = np.stack([cv2.resize(image, (self.config['image_width'], self.config['image_height']),
                                  interpolation=cv2.INTER_AREA).transpose(2, 0, 1) for image in frames])
        features = self.model.visual.encode_rgb(torch.from_numpy(rgb)[None].to(self.device))
        result = self.model(features, torch.from_numpy(measured)[None].to(self.device),
                            torch.from_numpy(command)[None].to(self.device), self.times)
        predicted_contact = plan['contact_time'] + float(result['contact_delay_s'][0])
        return {'contact_score': float(result['contact_logit'][0].sigmoid()),
                'predicted_contact_time_s': predicted_contact,
                'absolute_times_s': predicted_contact + self.times.cpu().numpy(),
                'xy_m': (result['future_xy'][0] * self.scale).cpu().numpy(),
                'scope': 'trajectory conditional on contact; contact score is not calibrated confidence'}
