"""
dm_control-compatible tolerance reward utility.

Local reimplementation of dm_control's rewards.tolerance() from
dm_control/suite/utils/rewards.py (Tassa et al. 2018, Tunyasuvunakool et al. 2020).
No runtime dependency on dm_control is added.
"""

import numpy as np


def _sigmoid(x, sigmoid):
    """Apply named sigmoid kernel to x (already normalised to distance from bounds)."""
    if sigmoid == 'gaussian':
        return np.exp(-0.5 * x ** 2)
    elif sigmoid == 'linear':
        return np.clip(1.0 - x, 0.0, 1.0)
    elif sigmoid == 'quadratic':
        return np.where(x < 1.0, 1.0 - x ** 2, 0.0)
    elif sigmoid == 'hyperbolic':
        return 1.0 / (1.0 + x)
    elif sigmoid == 'long_tail':
        return 1.0 / (1.0 + x ** 2)
    elif sigmoid == 'cosine':
        return np.where(x < 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    elif sigmoid == 'tanh_squared':
        return 1.0 - np.tanh(x) ** 2
    else:
        raise ValueError(f"Unknown sigmoid: {sigmoid!r}")


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian', value_at_margin=0.1):
    """
    Return a reward in [0, 1] based on how close x is to bounds.

    Inside bounds → 1.0. At margin distance from bounds → value_at_margin.
    Beyond margin → falls off according to the sigmoid kernel.

    Args:
        x: scalar or numpy array.
        bounds: (lower, upper) inclusive interval that yields reward 1.0.
        margin: non-negative distance from bounds at which reward == value_at_margin.
        sigmoid: one of 'gaussian', 'linear', 'quadratic', 'hyperbolic',
                 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: reward value at exactly margin distance from bounds.

    Returns:
        reward array (same shape as x).
    """
    x = np.asarray(x, dtype=float)
    lower, upper = bounds
    if lower > upper:
        raise ValueError(f"bounds[0] must be <= bounds[1], got {bounds}")
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")

    in_bounds = (x >= lower) & (x <= upper)

    if margin == 0.0:
        return np.where(in_bounds, 1.0, 0.0)

    # Scale so that distance == margin maps to value_at_margin via the sigmoid.
    # Solve sigmoid(scale) = value_at_margin for scale, then use d/margin * scale.
    # For gaussian: exp(-0.5 * s^2) = v  →  s = sqrt(-2 ln v)
    # We find scale numerically for generality.
    # dm_control uses a pre-computed scale factor per sigmoid.
    def _find_scale(v, sig):
        # sigmoid(scale) = v  →  find scale >= 0
        if sig == 'gaussian':
            if v <= 0.0:
                return np.inf
            return np.sqrt(-2.0 * np.log(v))
        elif sig == 'linear':
            return 1.0 - v
        elif sig == 'quadratic':
            return np.sqrt(1.0 - v)
        elif sig == 'hyperbolic':
            return (1.0 / v) - 1.0
        elif sig == 'long_tail':
            return np.sqrt((1.0 / v) - 1.0)
        elif sig == 'cosine':
            return np.arccos(2.0 * v - 1.0) / np.pi
        elif sig == 'tanh_squared':
            return np.arctanh(np.sqrt(1.0 - v))
        else:
            raise ValueError(f"Unknown sigmoid: {sig!r}")

    scale = _find_scale(value_at_margin, sigmoid)

    # Normalised distance from the nearest bound edge (0 inside bounds).
    d = np.where(x < lower, lower - x, np.where(x > upper, x - upper, 0.0))
    normalised = d / margin * scale

    reward = np.where(in_bounds, 1.0, _sigmoid(normalised, sigmoid))
    return float(reward) if reward.ndim == 0 else reward
