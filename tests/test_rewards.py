import numpy as np
import pytest
from agents.rewards import tolerance


def test_inside_bounds_exact():
    assert tolerance(0.5, bounds=(0.0, 1.0), margin=1.0) == pytest.approx(1.0)
    assert tolerance(0.0, bounds=(0.0, 0.0), margin=1.0) == pytest.approx(1.0)


def test_at_margin_boundary():
    for sig in ('gaussian', 'linear', 'quadratic', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'):
        v = 0.1
        r = tolerance(1.0, bounds=(0.0, 0.0), margin=1.0, sigmoid=sig, value_at_margin=v)
        assert abs(r - v) < 1e-9, f"sigmoid={sig}: got {r}, expected {v}"


def test_outside_decreasing():
    for sig in ('gaussian', 'linear', 'quadratic'):
        r1 = tolerance(1.5, bounds=(0.0, 0.0), margin=1.0, sigmoid=sig)
        r2 = tolerance(3.0, bounds=(0.0, 0.0), margin=1.0, sigmoid=sig)
        assert 0.0 <= r1 < 0.1, f"sigmoid={sig}: r1={r1} should be < value_at_margin"
        assert r2 <= r1, f"sigmoid={sig}: not monotone decreasing ({r1}, {r2})"


def test_vectorised_matches_scalar():
    xs = np.array([-2.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    vec = tolerance(xs, bounds=(-0.5, 0.5), margin=1.5)
    for i, xi in enumerate(xs):
        expected = tolerance(float(xi), bounds=(-0.5, 0.5), margin=1.5)
        assert abs(vec[i] - expected) < 1e-12, f"x={xi}: vec={vec[i]}, scalar={expected}"
    assert vec.shape == (len(xs),)


def test_invalid_margin_raises():
    with pytest.raises(ValueError, match="margin"):
        tolerance(0.0, bounds=(0.0, 1.0), margin=-0.1)


def test_invalid_bounds_raises():
    with pytest.raises(ValueError, match="bounds"):
        tolerance(0.0, bounds=(1.0, 0.0), margin=1.0)


def test_zero_margin_binary():
    assert tolerance(0.5, bounds=(0.0, 1.0), margin=0.0) == pytest.approx(1.0)
    assert tolerance(1.5, bounds=(0.0, 1.0), margin=0.0) == pytest.approx(0.0)
