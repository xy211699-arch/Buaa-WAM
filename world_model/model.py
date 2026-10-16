"""Small CNN/GRU predictor. No ground-truth state is accepted by forward()."""

import torch
from torch import nn


class RGBPuckPredictor(nn.Module):
    """Predict normalized world XY from RGB history and requested future times.

    Spatial pooling keeps a grid rather than a global average, preserving where
    the small puck appears. An auxiliary XY head learns visual localization;
    its predictions, never labels, also condition the trajectory head.
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        layers = []
        channels = 3
        for output in (12, 24, 32, 48):
            layers.extend([nn.Conv2d(channels, output, 3, stride=2, padding=1), nn.SiLU()])
            channels = output
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((6, 10)),
                                     nn.Flatten(), nn.Linear(48 * 6 * 10, hidden_dim), nn.SiLU())
        self.position_head = nn.Linear(hidden_dim, 2)
        self.temporal = nn.GRU(hidden_dim + 2, hidden_dim, batch_first=True)
        self.trajectory_head = nn.Sequential(nn.Linear(hidden_dim + 3, 128), nn.SiLU(),
                                             nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, rgb, future_times):
        """rgb: B,T,3,H,W uint8 or float [0,1]; times: K positive seconds."""
        batch, length, channels, height, width = rgb.shape
        pixels = rgb.reshape(batch * length, channels, height, width)
        pixels = pixels.float() / 255 if pixels.dtype == torch.uint8 else pixels
        embeddings = self.encoder(pixels).reshape(batch, length, -1)
        history_xy = self.position_head(embeddings)
        _, hidden = self.temporal(torch.cat((embeddings, history_xy), dim=-1))
        count = len(future_times)
        current = history_xy[:, -1]
        context = torch.cat((hidden[-1], current), dim=-1)[:, None].expand(-1, count, -1)
        times = future_times[None, :, None].expand(batch, -1, -1)
        displacement = self.trajectory_head(torch.cat((context, times), dim=-1))
        # Multiplication by time makes the trajectory continuous at t=0.
        future_xy = current[:, None] + times * displacement
        return {'future_xy': future_xy, 'history_xy': history_xy}


class SpatialRGBPuckPredictor(nn.Module):
    """Learn a puck heatmap from RGB, then forecast its movement with a GRU.

    Image-plane coordinates are converted with fixed camera calibration. No
    color threshold, GT coordinates, velocities, or future frames are inputs.
    """

    def __init__(self, image_to_world, hidden_dim=64, dt_s=.05):
        super().__init__()
        self.dt_s = dt_s
        self.encoder = nn.Sequential(nn.Conv2d(3, 12, 3, stride=2, padding=1), nn.SiLU(),
                                     nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.SiLU())
        self.heatmap_head = nn.Conv2d(24, 1, 1)
        self.register_buffer('image_to_world', torch.tensor(image_to_world, dtype=torch.float32))
        self.temporal = nn.GRU(28, hidden_dim, batch_first=True)
        self.trajectory_head = nn.Sequential(nn.Linear(hidden_dim + 5, 128), nn.SiLU(),
                                             nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
        # Begin with constant velocity inferred from RGB, then learn corrections.
        nn.init.zeros_(self.trajectory_head[-1].weight)
        nn.init.zeros_(self.trajectory_head[-1].bias)

    def encode_observation(self, rgb):
        """Encode only past/current RGB; never evaluate the future trajectory head."""
        batch, length, channels, height, width = rgb.shape
        pixels = rgb.reshape(batch * length, channels, height, width)
        pixels = pixels.float() / 255 if pixels.dtype == torch.uint8 else pixels
        features = self.encoder(pixels)
        logits = self.heatmap_head(features).squeeze(1)
        map_height, map_width = logits.shape[-2:]
        ys = (torch.arange(map_height, device=rgb.device, dtype=torch.float32) * 4 + .5) / height
        xs = (torch.arange(map_width, device=rgb.device, dtype=torch.float32) * 4 + .5) / width
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack((xx, yy), dim=-1)
        probabilities = logits.flatten(1).softmax(-1).reshape(-1, map_height, map_width)
        uv = (probabilities[..., None] * grid).sum((1, 2))
        homogeneous = torch.cat((uv, torch.ones_like(uv[:, :1])), dim=-1)
        world = homogeneous @ self.image_to_world.T
        history_xy = (world[:, :2] / world[:, 2:]).reshape(batch, length, 2)
        velocity = torch.zeros_like(history_xy)
        velocity[:, 1:] = (history_xy[:, 1:] - history_xy[:, :-1]) / self.dt_s
        visual = features.mean((2, 3)).reshape(batch, length, 24)
        return {'tokens': torch.cat((visual, history_xy, velocity), dim=-1),
                'history_xy': history_xy, 'velocity': velocity,
                'history_uv': uv.reshape(batch, length, 2),
                'heatmap_logits': logits.reshape(batch, length, map_height, map_width),
                'heatmap_grid': grid}

    def forward(self, rgb, future_times):
        observation = self.encode_observation(rgb)
        history_xy, velocity = observation['history_xy'], observation['velocity']
        batch = len(rgb)
        _, hidden = self.temporal(observation['tokens'])
        current, current_velocity = history_xy[:, -1], velocity[:, -1]
        context = torch.cat((hidden[-1], current, current_velocity), dim=-1)
        context = context[:, None].expand(-1, len(future_times), -1)
        times = future_times[None, :, None].expand(batch, -1, -1)
        correction = self.trajectory_head(torch.cat((context, times), dim=-1))
        return {'future_xy': current[:, None] + times * (current_velocity[:, None] + correction),
                **{key: observation[key] for key in ('history_xy', 'history_uv',
                                                   'heatmap_logits', 'heatmap_grid')}}


def build_model(config):
    if config.get('model_type', 'cnn_gru') == 'cnn_gru':
        return RGBPuckPredictor(config['hidden_dim'])
    if config['model_type'] == 'spatial':
        return SpatialRGBPuckPredictor(config['image_to_world'], config['hidden_dim'], config['dt_s'])
    raise ValueError('Unknown model_type')


def trajectory_loss(prediction, batch, localization_weight=.5, heatmap_weight=.2,
                    horizon_weight_growth=0.):
    """Masked normalized XY MSE; invalid post-contact targets have zero weight."""
    mask = batch['mask'].to(prediction['future_xy'].dtype)
    squared = (prediction['future_xy'] - batch['future_xy']).square().sum(-1)
    if horizon_weight_growth < 0:
        raise ValueError('horizon_weight_growth must be nonnegative')
    horizon_weights = torch.linspace(1., 1. + horizon_weight_growth,
                                     squared.shape[-1], device=squared.device)
    weights = mask * horizon_weights
    future = (squared * weights).sum() / weights.sum().clamp_min(1)
    localization = (prediction['history_xy'] - batch['history_xy']).square().sum(-1).mean()
    loss = future + localization_weight * localization
    if 'heatmap_logits' in prediction:
        logits, grid = prediction['heatmap_logits'], prediction['heatmap_grid']
        # Pixel-level localization supervision comes only from training labels.
        height, width = batch['rgb'].shape[-2:]
        spacing = logits.new_tensor([4. / width, 4. / height])
        delta = (grid[None, None] - batch['history_uv'][:, :, None, None]) / spacing
        target = (-delta.square().sum(-1) / (2 * .75 ** 2)).flatten(2).softmax(-1)
        heatmap_loss = -(target * logits.flatten(2).log_softmax(-1)).sum(-1).mean()
        loss = loss + heatmap_weight * heatmap_loss
    return loss
