"""Measured robot-state conditioning for post-hit prediction and contact probability."""

import numpy as np
import torch
from torch import nn

from .hit_model import HitPredictor


def robot_features(q, dq, target, names, arm_names):
    """18 features: q/pi, dq/2 rad/s, and target-q/pi, in configured joint order."""
    if len(set(names)) != len(names) or len(arm_names) != 6:
        raise ValueError('Expected unique names and six arm joints')
    arrays = [np.asarray(value, dtype=np.float32) for value in (q, dq, target)]
    if any(value.shape != (len(names),) or not np.isfinite(value).all() for value in arrays):
        raise ValueError('Robot measurements must match joint names and be finite')
    indices = [names.index(name) for name in arm_names]
    position, velocity, command = [value[indices] for value in arrays]
    return np.concatenate((position / np.pi, velocity / 2., (command - position) / np.pi)).astype(np.float32)


class ExecutionPredictor(nn.Module):
    """RGB-derived ball features + measured robot state + candidate command.

    No puck truth or future robot measurements are inputs. Contact probability
    describes executed branches in this dataset, not arbitrary-action safety.
    """

    def __init__(self, config, use_robot=True):
        super().__init__()
        self.visual = HitPredictor(config)
        for parameter in self.visual.parameters():
            parameter.requires_grad_(False)
        self.use_robot = use_robot
        self.register_buffer('robot_mean', torch.zeros(18))
        self.register_buffer('robot_std', torch.ones(18))
        feature_dim = 2 * (config['history_frames'] + config['backbone_config']['horizon_steps'])
        self.context = nn.Sequential(nn.Linear(feature_dim + 18 + 6, 96), nn.SiLU(),
                                     nn.Linear(96, 96), nn.SiLU())
        self.contact_head = nn.Linear(96, 1)
        self.delay_head = nn.Linear(96, 1)
        self.trajectory_head = nn.Sequential(nn.Linear(97, 96), nn.SiLU(), nn.Linear(96, 4))
        self.history_frames = config['history_frames']

    def forward(self, features, robot, action, times):
        robot = (robot - self.robot_mean) / self.robot_std
        if not self.use_robot:
            robot = torch.zeros_like(robot)
        context = self.context(torch.cat((features, robot, action), dim=-1))
        t = times[None, :, None].expand(len(features), -1, -1)
        value = self.trajectory_head(torch.cat((context[:, None].expand(-1, len(times), -1), t), -1))
        current = features[:, 2 * self.history_frames - 2:2 * self.history_frames]
        return {'future_xy': current[:, None] + value[..., :2] + t * value[..., 2:],
                'contact_logit': self.contact_head(context).squeeze(-1),
                'contact_delay_s': self.delay_head(context).squeeze(-1)}


def execution_loss(result, target, mask, delay, contact):
    """Failed branches supervise contact only; never fabricate post-contact labels."""
    position = (result['future_xy'] - target).square().sum(-1)
    trajectory = (position * mask).sum() / mask.sum().clamp_min(1)
    positive = contact > .5
    timing = ((result['contact_delay_s'][positive] - delay[positive]).square().mean()
              if positive.any() else result['contact_delay_s'].sum() * 0.)
    classification_points = nn.functional.binary_cross_entropy_with_logits(
        result['contact_logit'], contact, reduction='none')
    # Rare misses must not disappear behind the many successful expert samples.
    classification = (classification_points.mean() if not positive.any() or positive.all()
                      else .5 * classification_points[positive].mean()
                      + .5 * classification_points[~positive].mean())
    return trajectory + timing + .1 * classification
