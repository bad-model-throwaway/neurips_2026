import numpy as np
import pytest

from agents.spline import TimeSpline


def _make_1d(interp, times, values):
    s = TimeSpline(dim=1, interp=interp)
    for t, v in zip(times, values):
        s.add_knot(t, np.array([v]))
    return s


def test_zero_order_returns_left_knot():
    s = _make_1d('zero', [0.0, 1.0, 2.0], [5.0, 7.0, 9.0])
    assert s.sample(0.0)[0] == 5.0
    assert s.sample(0.5)[0] == 5.0
    assert s.sample(0.999)[0] == 5.0
    assert s.sample(1.0)[0] == 7.0
    assert s.sample(1.5)[0] == 7.0


def test_linear_midpoint_is_average():
    s = _make_1d('linear', [0.0, 1.0, 2.0], [10.0, 20.0, 40.0])
    assert s.sample(0.5)[0] == pytest.approx(15.0)
    assert s.sample(1.5)[0] == pytest.approx(30.0)


def test_cubic_interpolates_at_knots():
    s = _make_1d('cubic', [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, -1.0])
    for t, v in zip([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, -1.0]):
        assert s.sample(t)[0] == pytest.approx(v, abs=1e-12)


def test_cubic_hermite_matches_handcomputed():
    # 3 knots at t = 0, 1, 2 with values 0, 1, 0. Boundary slopes are one-sided.
    # phi_0 = (1-0)/1 = 1; phi_1 (interior) = (0-0)/(2-0) = 0; phi_2 = (0-1)/1 = -1.
    # Sample at t=0.5 (q=0.5, dt=1, interval [0,1]):
    # a = 2(0.125) - 3(0.25) + 1 = 0.5
    # b = (0.125 - 0.5 + 0.5)*1 = 0.125
    # c = -0.25 + 0.75 = 0.5
    # d = (0.125 - 0.25)*1 = -0.125
    # f = 0.5*0 + 0.125*1 + 0.5*1 + (-0.125)*0 = 0.625
    s = _make_1d('cubic', [0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    assert s.sample(0.5)[0] == pytest.approx(0.625, abs=1e-12)


def test_boundary_clamp():
    s = _make_1d('cubic', [0.0, 1.0, 2.0], [3.0, 4.0, 5.0])
    assert s.sample(-1.0)[0] == 3.0
    assert s.sample(99.0)[0] == 5.0
    s_zero = _make_1d('zero', [0.0, 1.0, 2.0], [3.0, 4.0, 5.0])
    assert s_zero.sample(-1.0)[0] == 3.0
    assert s_zero.sample(99.0)[0] == 5.0


def test_multi_dim_independent():
    s = TimeSpline(dim=3, interp='linear')
    s.add_knot(0.0, np.array([0.0, 10.0, -1.0]))
    s.add_knot(1.0, np.array([1.0, 20.0, -2.0]))
    s.add_knot(2.0, np.array([2.0, 40.0, -4.0]))
    v = s.sample(0.5)
    assert v.shape == (3,)
    assert v[0] == pytest.approx(0.5)
    assert v[1] == pytest.approx(15.0)
    assert v[2] == pytest.approx(-1.5)


def test_knots_times_copy_and_len():
    s = _make_1d('linear', [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    assert len(s) == 3
    assert s.times.shape == (3,)
    assert s.knots.shape == (3, 1)
    t = s.copy()
    t.clear()
    assert len(t) == 0
    assert len(s) == 3  # copy is independent


def test_monotone_times_enforced():
    s = TimeSpline(dim=1, interp='linear')
    s.add_knot(1.0, np.array([0.0]))
    with pytest.raises(ValueError):
        s.add_knot(0.5, np.array([0.0]))
