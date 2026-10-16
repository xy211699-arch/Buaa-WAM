"""Direct present-only policy with an explicitly separate legacy execution bridge."""

import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from .execution import robot_features
from .joint_model import JointHitModel
from .model import build_model
from .pipeline import rgb_reference
from .policy import decode_intention, expert_intention


class JointHitPipeline:
    def __init__(self, checkpoint_path, device='cpu', sampling_seed=42):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.config, self.splits = checkpoint['config'], checkpoint['splits']
        self.variant = checkpoint['variant']
        self.device = torch.device(device)
        self.world_checkpoint = Path(checkpoint['visual_checkpoint'])
        if hashlib.sha256(self.world_checkpoint.read_bytes()).hexdigest() != checkpoint['visual_checkpoint_sha256']:
            raise ValueError('Visual checkpoint mismatch')
        source = torch.load(self.world_checkpoint, map_location='cpu', weights_only=True)
        self.vision = build_model(source['config']).to(self.device).eval()
        self.vision.load_state_dict(source['model'])
        self.model = JointHitModel(self.config).to(self.device).eval()
        self.model.load_state_dict(checkpoint['model'])
        self.sampling_seed = sampling_seed
        self.times = torch.arange(1, self.config['horizon_steps'] + 1, device=self.device) * self.config['dt_s']

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    @torch.no_grad()
    def direct_policy(self, rgb, q, dq, target, names):
        """No true or predicted future enters action generation; q rad, dq rad/s."""
        frames = np.asarray(rgb)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != self.config['history_frames'] or frames.shape[-1] != 3:
            raise ValueError('Expected chronological uint8 RGB T,H,W,3')
        pixels = np.stack([cv2.resize(image, (self.config['image_width'], self.config['image_height']),
                                    interpolation=cv2.INTER_AREA).transpose(2, 0, 1) for image in frames])
        tensor = torch.from_numpy(pixels)[None].to(self.device)
        visual = self.vision.encode_observation(tensor)['tokens']
        robot = torch.from_numpy(robot_features(q, dq, target, names, self.config['arm_joint_names']))[None].to(self.device)
        # Common random numbers for paired models; no outcome-based sample choice.
        noise = torch.randn(1, 6, generator=torch.Generator().manual_seed(self.sampling_seed)).to(self.device)
        values, bank = self.model.policy(visual, robot, noise, self.config['sampling_steps'])
        intention = decode_intention(values[0].cpu().numpy(), -20.)
        return {'tensor': tensor, 'visual': visual, 'robot': robot, 'intention': intention,
                'raw_policy': values[0].cpu().tolist(), 'bank_logit': float(bank[0])}

    @torch.no_grad()
    def propose(self, rgb, q, dq, target, names):
        self.synchronize(); start = time.perf_counter()
        proposal = self.direct_policy(rgb, q, dq, target, names)
        self.synchronize(); policy_seconds = time.perf_counter() - start
        # Existing Cartesian executor still needs a predicted incoming reference.
        # This is separately timed and is not part of the direct policy computation.
        start = time.perf_counter()
        forecast = self.vision(proposal['tensor'], self.times)
        features = torch.cat((forecast['history_xy'].flatten(1), forecast['future_xy'].flatten(1)), -1)
        reference = rgb_reference(features[0].cpu().numpy(), self.config)
        self.synchronize(); bridge_seconds = time.perf_counter() - start
        return {**proposal, 'reference': reference, 'intentions': [proposal['intention']],
                'policy_seconds': policy_seconds, 'bridge_seconds': bridge_seconds}

    @torch.no_grad()
    def rank(self, proposal, prepared):
        result = []
        for index, plan in prepared:
            intention, _ = expert_intention({**plan, 'contact_time': plan['contact_time']}, 0.)
            command = torch.tensor(intention, device=self.device)[None]
            normalized = (command - self.model.action_mean) / self.model.action_std
            z = self.model.encode_present(proposal['visual'], proposal['robot'])
            predicted = self.model.future(z, proposal['visual'][:, -1, 24:26], self.times, normalized)
            result.append({'candidate_index': index, 'score': 0., 'plan': plan,
                'predicted_xy_m': (predicted['xy'][0] * self.times.new_tensor(self.config['xy_scale_m'])).cpu().tolist(),
                'contact_score': float(predicted['contact_logit'][0].sigmoid()),
                'predicted_contact_s': float(predicted['contact_delay_s'][0])})
        return result
