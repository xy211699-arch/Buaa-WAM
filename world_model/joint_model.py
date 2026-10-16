"""Direct flow policy and future heads sharing a trainable present-only encoder."""

import math

import torch
from torch import nn


class JointHitModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config['context_width']
        self.config = config
        self.register_buffer('visual_mean', torch.zeros(28))
        self.register_buffer('visual_std', torch.ones(28))
        self.register_buffer('robot_mean', torch.zeros(18))
        self.register_buffer('robot_std', torch.ones(18))
        self.register_buffer('action_mean', torch.zeros(6))
        self.register_buffer('action_std', torch.ones(6))
        self.visual_projection = nn.Linear(28, width)
        self.robot_projection = nn.Linear(18, width)
        self.position = nn.Parameter(torch.randn(1, config['history_frames'] + 1, width) * .01)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0.,
                                            activation='gelu', batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, config['context_layers'], enable_nested_tensor=False)
        self.context_norm = nn.LayerNorm(width)
        self.action_flow = nn.Sequential(nn.Linear(width + 6 + 8, width * 2), nn.SiLU(),
                                         nn.Linear(width * 2, width * 2), nn.SiLU(), nn.Linear(width * 2, 6))
        self.mode_head = nn.Linear(width, 1)
        self.future_head = nn.Sequential(nn.Linear(width + 1, width), nn.SiLU(), nn.Linear(width, 2))
        self.dynamics_context = nn.Sequential(nn.Linear(width + 6, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.dynamics_head = nn.Sequential(nn.Linear(width + 1, width), nn.SiLU(), nn.Linear(width, 2))
        self.contact_head = nn.Linear(width, 2)

    def encode_present(self, visual, robot):
        """Only historical RGB-derived features and decision-time robot measurements."""
        tokens = self.visual_projection((visual - self.visual_mean) / self.visual_std)
        robot_token = self.robot_projection((robot - self.robot_mean) / self.robot_std)[:, None]
        return self.context_norm(self.context(torch.cat((tokens, robot_token), 1) + self.position)[:, -1])

    def velocity(self, context, noisy_action, noise_time):
        """Flow time 0=data, 1=noise; unrelated to physical future seconds."""
        frequencies = noisy_action.new_tensor([1., 2., 4., 8.]) * math.pi
        phase = noise_time[:, None] * frequencies
        embedding = torch.cat((phase.sin(), phase.cos()), -1)
        return self.action_flow(torch.cat((context, noisy_action, embedding), -1))

    def policy(self, visual, robot, noise, steps=4):
        """Future heads are never accessed; fixed noise enables paired comparisons."""
        if steps <= 0:
            raise ValueError('Positive integration steps required')
        z = self.encode_present(visual, robot)
        value = noise.clone()
        for index in range(steps):
            tau = value.new_full((len(value),), 1. - index / steps)
            value = value - self.velocity(z, value, tau) / steps
        return value * self.action_std + self.action_mean, self.mode_head(z).squeeze(-1)

    def future(self, context, current_xy, times, executed_action=None, detach_context=False):
        """Predict decision-relative XY; executed_action is a recorded command label.

        Detaching context is only the representation-learning ablation.
        """
        z = context.detach() if detach_context else context
        if executed_action is not None:
            z = self.dynamics_context(torch.cat((z, executed_action), -1))
        t = times[None, :, None].expand(len(z), -1, -1)
        inputs = torch.cat((z[:, None].expand(-1, len(times), -1), t), -1)
        head = self.future_head if executed_action is None else self.dynamics_head
        xy = current_xy[:, None] + t * head(inputs)
        if executed_action is None:
            return {'xy': xy}
        contact = self.contact_head(z)
        return {'xy': xy, 'contact_logit': contact[:, 0], 'contact_delay_s': contact[:, 1]}


def masked_xy_loss(prediction, target, mask):
    errors = (prediction - target).square().sum(-1)
    return (errors * mask).sum() / mask.sum().clamp_min(1)
