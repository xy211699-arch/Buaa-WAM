"""Small behavior-cloning policy over RGB world-model and measured robot features."""

import numpy as np
import torch
from torch import nn


INTENT_KEYS = ('contact_delay_s', 'goal_target_y', 'desired_out_speed',
               'direction_correction_scaled', 'speed_scale', 'command_lead_s')


def expert_intention(plan, decision_s):
    """Training labels only; never read outcomes or randomized physical parameters."""
    values = [plan['contact_time'] - decision_s, plan['goal_target_y'],
              plan['desired_out_speed'], plan.get('direction_correction_deg', 0.) / 30.,
              plan.get('speed_scale', 1.), plan.get('command_lead_s', 0.)]
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all() or result[0] <= 0:
        raise ValueError('Expected finite future expert intention')
    return result, float(plan.get('attack_mode', 'DIRECT') == 'BANK_NEAR_RAIL')


class BehaviorPolicy(nn.Module):
    def __init__(self, feature_dim=58):
        super().__init__()
        self.register_buffer('input_mean', torch.zeros(feature_dim + 18))
        self.register_buffer('input_std', torch.ones(feature_dim + 18))
        self.register_buffer('output_mean', torch.zeros(6))
        self.register_buffer('output_std', torch.ones(6))
        self.net = nn.Sequential(nn.Linear(feature_dim + 18, 96), nn.SiLU(),
                                 nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 7))

    def forward(self, features, robot):
        context = (torch.cat((features, robot), -1) - self.input_mean) / self.input_std
        value = self.net(context)
        return value[:, :6] * self.output_std + self.output_mean, value[:, 6]


def decode_intention(values, bank_logit, horizon_s=1.):
    """Bound the experimental high-level command; no feasibility/success guarantee."""
    values = np.asarray(values, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all() or not np.isfinite(bank_logit):
        raise ValueError('Nonfinite or malformed policy output')
    return {'contact_time': float(np.clip(values[0], .15, horizon_s)),
            'goal_target_y': float(np.clip(values[1], -.16, .16)),
            'desired_out_speed': float(np.clip(values[2], .7, 2.)),
            'direction_correction_deg': float(np.clip(values[3] * 30., -35., 35.)),
            'speed_scale': float(np.clip(values[4], .65, 1.35)),
            'command_lead_s': float(np.clip(values[5], 0., .1)),
            'attack_mode': 'BANK_NEAR_RAIL' if bank_logit >= 0 else 'DIRECT'}
