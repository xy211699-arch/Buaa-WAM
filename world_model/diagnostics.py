"""Offline error measurements; ground truth is used only by diagnostic callers."""

import numpy as np


def decompose_xy_error(predicted_xy, actual_xy, actual_velocity):
    """Signed along-motion/cross-motion error in metres and along timing in seconds.

    A positive along error means the prediction lies ahead along actual motion.
    Stationary points have no defined motion direction and return None components.
    """
    predicted, actual, velocity = [np.asarray(value, dtype=float)
                                   for value in (predicted_xy, actual_xy, actual_velocity)]
    if any(value.shape != (2,) or not np.isfinite(value).all()
           for value in (predicted, actual, velocity)):
        raise ValueError('Expected finite XY pairs')
    error = predicted - actual
    speed = float(np.linalg.norm(velocity))
    if speed < 1e-6:
        return {'total_m': float(np.linalg.norm(error)), 'along_m': None,
                'cross_m': None, 'equivalent_time_s': None}
    direction = velocity / speed
    along = float(error @ direction)
    cross = float(error @ np.array([-direction[1], direction[0]]))
    return {'total_m': float(np.linalg.norm(error)), 'along_m': along,
            'cross_m': cross, 'equivalent_time_s': along / speed}
