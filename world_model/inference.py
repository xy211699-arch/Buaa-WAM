"""Load a trained predictor and forecast directly from RGB camera history."""

import cv2
import numpy as np
import torch

from .model import build_model


class RGBTrajectoryPredictor:
    """Predict free-puck XY in meters using a fixed, calibrated overhead camera.

    Supply chronological RGB uint8 frames at checkpoint dt_s. A camera with a
    different pose/FOV/crop requires recalibration and validation. This predicts
    unstruck motion; no candidate action or robot control is performed.
    """

    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.config = checkpoint['config']
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.times = torch.arange(1, self.config['horizon_steps'] + 1,
                                  device=self.device) * self.config['dt_s']
        self.scale = torch.tensor(self.config['xy_scale_m'], device=self.device)

    @torch.no_grad()
    def predict(self, rgb_history):
        """Return relative times_s (K) and predicted world xy_m (K,2), no GT input.

        rgb_history has shape T,H,W,3, ordered oldest to newest. Predictions are
        relative to the last frame's timestamp. Maintain exact dt_s spacing;
        dropped camera frames must be handled by the caller, not silently reused.
        """
        frames = np.asarray(rgb_history)
        if frames.ndim != 4 or frames.shape[0] != self.config['history_frames'] or frames.shape[-1] != 3:
            raise ValueError('Expected exactly history_frames RGB frames of shape T,H,W,3')
        if frames.dtype != np.uint8:
            raise ValueError('RGB frames must be uint8 in [0,255]')
        resized = [cv2.resize(frame, (self.config['image_width'], self.config['image_height']),
                              interpolation=cv2.INTER_AREA).transpose(2, 0, 1) for frame in frames]
        rgb = torch.from_numpy(np.stack(resized))[None].to(self.device)
        forecast = self.model(rgb, self.times)['future_xy'][0] * self.scale
        return {'times_s': self.times.cpu().numpy(), 'xy_m': forecast.cpu().numpy()}
