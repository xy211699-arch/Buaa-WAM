"""Validate native held-out RGB replay, timing, and bounce diagnostics."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch

from data.decode_image_bit import decode_image_bit
from world_model.data import PuckRGBDataset
from world_model.inference import RGBTrajectoryPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, help='Trusted training checkpoint')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    predictor = RGBTrajectoryPredictor(args.checkpoint, device=args.device)
    config = predictor.config
    torch.set_num_threads(config['cpu_threads'])
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    root = ROOT / config['data_root']
    # Preserve the training-time data contract even for diagnostic replay.
    import hashlib
    fingerprints = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for names in checkpoint['splits'].values() for name in names}
    if fingerprints != checkpoint['dataset_sha256']:
        raise ValueError('Dataset changed since training')
    dataset = PuckRGBDataset(root, checkpoint['splits']['test'], config)
    scale = torch.tensor(config['xy_scale_m'], device=args.device)
    errors = {'bounce': [], 'no_bounce': []}
    final = {'bounce': [], 'no_bounce': []}
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            episode = dataset.episodes[sample['episode_id']]
            frame, count = sample['frame_index'], int(sample['mask'].sum())
            velocity = episode['velocity'][frame:frame + count + 1]
            # Truth is used only to classify diagnostic groups, never as input.
            bounced = bool(np.any(velocity[1:] * velocity[:-1] < -.01))
            key = 'bounce' if bounced else 'no_bounce'
            result = predictor.model(sample['rgb'][None].to(args.device), predictor.times)
            prediction = (result['future_xy'][0] * scale).cpu().numpy()
            truth = sample['future_xy'].numpy() * np.array(config['xy_scale_m'])
            error = np.linalg.norm(prediction - truth, axis=-1)[:count]
            errors[key].extend(error.tolist())
            if count == config['horizon_steps']:
                final[key].append(float(error[-1]))
    sample = dataset[0]
    episode = dataset.episodes[sample['episode_id']]
    frame = sample['frame_index']
    with h5py.File(root / episode['filename'], 'r') as file:
        rgb = np.stack([decode_image_bit(file['vision/cam_head/colors'][index])
                        for index in range(frame - config['history_frames'] + 1, frame + 1)])
    latencies = []
    for index in range(60):
        if args.device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = predictor.predict(rgb)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        if index >= 10:
            latencies.append((time.perf_counter() - start) * 1000)
    with torch.no_grad():
        result = predictor.model(sample['rgb'][None].to(args.device), predictor.times)
        expected = (result['future_xy'][0] * scale).cpu().numpy()
    report = {
        'scope': 'held-out recorded simulation RGB replay, not live simulation',
        'segments': {key: {
            'valid_points': len(errors[key]),
            'ade_m': float(np.mean(errors[key])) if errors[key] else None,
            'full_horizon_windows': len(final[key]),
            'one_second_error_m': float(np.mean(final[key])) if final[key] else None,
        } for key in errors},
        'bounce_definition': 'GT velocity component sign change, product < -0.01 (m/s)^2',
        'pipeline_median_ms': float(np.median(latencies)),
        'pipeline_p95_ms': float(np.percentile(latencies, 95)),
        'timing_scope': 'RGB resize, input transfer, forward and CPU output; excludes decoding/rendering',
        'finite_output': bool(np.isfinite(output['xy_m']).all()),
        'api_matches_dataset_prediction': bool(np.allclose(output['xy_m'], expected,
                                                           atol=.001, rtol=0)),
        'api_max_difference_m': float(np.abs(output['xy_m'] - expected).max()),
        'consistency_tolerance_m': .001,
    }
    output_path = args.checkpoint.parent / 'rgb_replay_report.json'
    output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['finite_output'] or not report['api_matches_dataset_prediction']:
        raise RuntimeError('RGB inference validation failed; inspect replay report')


if __name__ == '__main__':
    main()
