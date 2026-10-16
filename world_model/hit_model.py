"""Action-conditioned post-hit head over a frozen RGB free-flight world model."""

import torch
from torch import nn

from .model import build_model


class HitPredictor(nn.Module):
    """Predict post-contact XY and actual minus planned impact delay (seconds).

    Inputs are RGB and six planned command values. Contact truth is never an input.
    Forecast times are seconds after the model's predicted physical contact.
    """

    def __init__(self, config, use_action=True):
        super().__init__()
        self.backbone = build_model(config['backbone_config'])
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.use_action = use_action
        self.history_frames = config['history_frames']
        self.register_buffer('backbone_times', torch.arange(
            1, config['backbone_config']['horizon_steps'] + 1) * config['dt_s'])
        feature_dim = 2 * (config['history_frames'] + len(self.backbone_times))
        self.context = nn.Sequential(nn.Linear(feature_dim + 6, 64), nn.SiLU(),
                                     nn.Linear(64, 64), nn.SiLU())
        self.delay_head = nn.Linear(64, 1)
        self.trajectory_head = nn.Sequential(nn.Linear(65, 64), nn.SiLU(), nn.Linear(64, 4))

    @torch.no_grad()
    def encode_rgb(self, rgb):
        result = self.backbone(rgb, self.backbone_times)
        return torch.cat((result['history_xy'].flatten(1), result['future_xy'].flatten(1)), dim=-1)

    def forecast_features(self, features, action, future_times):
        """Cached features contain only the frozen model's RGB-derived predictions."""
        if not self.use_action:
            action = torch.zeros_like(action)
        context = self.context(torch.cat((features, action), dim=-1))
        times = future_times[None, :, None].expand(len(features), -1, -1)
        expanded = context[:, None].expand(-1, len(future_times), -1)
        trajectory = self.trajectory_head(torch.cat((expanded, times), dim=-1))
        current_xy = features[:, 2 * self.history_frames - 2:2 * self.history_frames]
        return {'future_xy': current_xy[:, None] + trajectory[..., :2] + times * trajectory[..., 2:],
                'contact_delay_s': self.delay_head(context).squeeze(-1)}

    def forward(self, rgb, action, future_times):
        return self.forecast_features(self.encode_rgb(rgb), action, future_times)


def hit_loss(result, target, mask, contact_delay_s):
    """Masked position MSE and impact-timing supervision; no post-next-hit labels."""
    error = (result['future_xy'] - target).square().sum(-1)
    position = (error * mask).sum() / mask.sum().clamp_min(1)
    timing = (result['contact_delay_s'] - contact_delay_s).square().mean()
    return position + timing
