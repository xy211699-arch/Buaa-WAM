"""BC intentions and action-world-model ranking, using decision-time inputs only."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from .execution import ExecutionPredictor, robot_features
from .hit_data import planned_action
from .policy import BehaviorPolicy, decode_intention


def execution_candidates(ranked, selection):
    """Select only model-ranked or unperturbed BC commands, without outcome input.

    BC selection does not silently replace an infeasible base action with a
    perturbed candidate. An empty list means no command should be executed.
    """
    if selection == 'world-model':
        return ranked
    if selection == 'bc':
        return [item for item in ranked if item['candidate_index'] == 0]
    raise ValueError('Unknown candidate selection')


def rgb_reference(features, config):
    """Free-flight model output converted to executor reference; spin assumed zero.

    Times are relative to the last RGB frame. No simulator rollout or puck truth
    is accessed. Velocity is a finite difference of predicted positions.
    """
    values = np.asarray(features, dtype=float)
    history = config['history_frames']
    current = values[2 * history - 2:2 * history]
    future = values[2 * history:].reshape(-1, 2)
    xy = np.vstack((current, future)) * np.asarray(config['xy_scale_m'])
    times = np.arange(len(xy), dtype=float) * config['dt_s']
    if not np.isfinite(xy).all():
        raise ValueError('Invalid RGB world-model prediction')
    return {'t': times, 'xy': xy, 'vxy': np.gradient(xy, times, axis=0),
            'omega_z': np.zeros(len(xy))}


class HitPipeline:
    """Left-arm experiment: policy proposal -> dry planning -> world-model ranking."""

    def __init__(self, policy_checkpoint, world_checkpoint=None, device='cpu'):
        policy = torch.load(policy_checkpoint, map_location='cpu', weights_only=True)
        world_checkpoint = Path(world_checkpoint or policy['world_checkpoint'])
        if hashlib.sha256(world_checkpoint.read_bytes()).hexdigest() != policy['world_checkpoint_sha256']:
            raise ValueError('Policy/world-model checkpoint mismatch')
        world = torch.load(world_checkpoint, map_location='cpu', weights_only=True)
        self.world_checkpoint = world_checkpoint.resolve()
        if world['splits'] != policy['splits']:
            raise ValueError('Seed split mismatch')
        self.config, self.splits = world['config'], world['splits']
        self.device = torch.device(device)
        self.world = ExecutionPredictor(self.config, world['use_robot']).to(self.device).eval()
        self.world.load_state_dict(world['model'])
        self.policy = BehaviorPolicy(policy['feature_dim']).to(self.device).eval()
        self.policy.load_state_dict(policy['model'])
        self.times = torch.arange(1, self.config['horizon_steps'] + 1, device=self.device) * self.config['dt_s']
        self.scale = torch.tensor(self.config['xy_scale_m'], device=self.device)

    @torch.no_grad()
    def propose(self, rgb, q, dq, target, names):
        """No seed, cached expert plan, future image or puck truth enters this API."""
        frames = np.asarray(rgb)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != self.config['history_frames'] or frames.shape[-1] != 3:
            raise ValueError('Expected chronological uint8 RGB history T,H,W,3')
        pixels = np.stack([cv2.resize(image, (self.config['image_width'], self.config['image_height']),
                                    interpolation=cv2.INTER_AREA).transpose(2, 0, 1) for image in frames])
        features = self.world.visual.encode_rgb(torch.from_numpy(pixels)[None].to(self.device))
        robot = torch.from_numpy(robot_features(q, dq, target, names, self.config['arm_joint_names']))[None].to(self.device)
        values, bank = self.policy(features, robot)
        base = decode_intention(values[0].cpu().numpy(), float(bank[0]), float(self.times[-1]))
        candidates = []
        for angle, speed in ((0., 1.), (-8., 1.), (8., 1.), (0., .9), (0., 1.1)):
            intention = dict(base)
            intention['direction_correction_deg'] += angle
            intention['speed_scale'] *= speed
            candidates.append(intention)
        return {'features': features, 'robot': robot,
                'reference': rgb_reference(features[0].cpu().numpy(), self.config),
                'intentions': candidates, 'raw_policy': values[0].cpu().tolist(),
                'bank_logit': float(bank[0])}

    @torch.no_grad()
    def rank(self, proposal, prepared_plans):
        """Plans and their times are relative to decision (decision_s=0).

        The currently unreliable contact score is logged but does not veto
        candidates. Ranking favors +X progress and central Y at 0.5 seconds.
        """
        ranked = []
        for index, plan in prepared_plans:
            action = planned_action(plan, 0., self.config['xy_scale_m'])
            result = self.world(proposal['features'], proposal['robot'],
                                torch.from_numpy(action)[None].to(self.device), self.times)
            xy = (result['future_xy'][0] * self.scale).cpu().numpy()
            score = float(xy[9, 0] - .2 * abs(xy[9, 1]))
            ranked.append({'candidate_index': index, 'score': score, 'plan': plan,
                           'predicted_xy_m': xy.tolist(),
                           'contact_score': float(result['contact_logit'][0].sigmoid()),
                           'predicted_contact_s': plan['contact_time'] + float(result['contact_delay_s'][0])})
        return sorted(ranked, key=lambda item: item['score'], reverse=True)
