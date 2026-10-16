"""Runtime RGB plus planned-action interface; no contact truth required."""

import cv2
import numpy as np
import torch

from .hit_data import planned_action
from .hit_model import HitPredictor


class HitTrajectoryPredictor:
    """Predict the first left hit using the calibrated overhead camera.

    Candidate plans must use the same action semantics as the expert. Returned
    trajectories exclude any later robot action; next-hit time is unknown here.
    """

    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if not checkpoint['use_action']:
            raise ValueError('Expected an action-conditioned checkpoint')
        self.config = checkpoint['config']
        self.support_counts = np.array(checkpoint['training_support_counts'])
        self.model = HitPredictor(self.config).to(self.device).eval()
        self.model.load_state_dict(checkpoint['model'])
        self.times = torch.arange(1, self.config['horizon_steps'] + 1,
                                  device=self.device) * self.config['dt_s']
        self.scale = torch.tensor(self.config['xy_scale_m'], device=self.device)

    @torch.no_grad()
    def predict(self, rgb_history, plan, decision_s):
        """Chronological RGB uint8 T,H,W,3, exact dt; plan times share decision clock.

        Last frame must precede the fast stroke at prep_time (preparation ends).
        Preparation motion can already be visible. Only planned contact_time,
        contact_x/y, nominal_theta_deg, direction_correction_deg, hit_speed,
        arm and coordinate_frame are used. No actual outcome fields are read.
        """
        frames = np.asarray(rgb_history)
        if frames.ndim != 4 or frames.shape[0] != self.config['history_frames'] or frames.shape[-1] != 3:
            raise ValueError('Expected exactly history_frames RGB frames T,H,W,3')
        if frames.dtype != np.uint8:
            raise ValueError('Expected RGB uint8')
        if not np.isfinite(decision_s) or decision_s > plan['prep_time'] - self.config['contact_guard_s']:
            raise ValueError('Decision must precede the fast stroke with the configured guard')
        action = planned_action(plan, decision_s, self.config['xy_scale_m'])
        pixels = np.stack([cv2.resize(frame, (self.config['image_width'], self.config['image_height']),
                                     interpolation=cv2.INTER_AREA).transpose(2, 0, 1) for frame in frames])
        result = self.model(torch.from_numpy(pixels)[None].to(self.device),
                            torch.from_numpy(action)[None].to(self.device), self.times)
        predicted_contact = plan['contact_time'] + float(result['contact_delay_s'][0])
        return {
            'predicted_contact_time_s': predicted_contact,
            'times_after_predicted_contact_s': self.times.cpu().numpy(),
            'absolute_times_s': predicted_contact + self.times.cpu().numpy(),
            'xy_m': (result['future_xy'][0] * self.scale).cpu().numpy(),
            'training_support_windows': self.support_counts.copy(),
        }
