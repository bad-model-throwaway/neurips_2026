"""Time-indexed spline (zero / linear / cubic Hermite). Paper Appendix A."""
import numpy as np


class TimeSpline:
    """P knots of dim-D values queryable at arbitrary time."""

    def __init__(self, dim, interp='cubic'):
        if interp not in ('zero', 'linear', 'cubic'):
            raise ValueError(f"interp must be zero|linear|cubic, got {interp!r}")
        self.dim = int(dim)
        self.interp = interp
        self._times = []
        self._values = []

    def add_knot(self, t, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.dim,):
            raise ValueError(f"values must have shape ({self.dim},), got {values.shape}")
        if self._times and t < self._times[-1]:
            raise ValueError(f"knot times must be monotone non-decreasing (got {t} after {self._times[-1]})")
        self._times.append(float(t))
        self._values.append(values.copy())

    def clear(self):
        self._times = []
        self._values = []

    def copy(self):
        other = TimeSpline(self.dim, self.interp)
        other._times = list(self._times)
        other._values = [v.copy() for v in self._values]
        return other

    def __len__(self):
        return len(self._times)

    @property
    def times(self):
        return np.asarray(self._times, dtype=float)

    @property
    def knots(self):
        if not self._values:
            return np.zeros((0, self.dim))
        return np.stack(self._values, axis=0)

    def _finite_diff_slope(self, j):
        t = self._times
        y = self._values
        P = len(t)
        if j == 0:
            dt = t[1] - t[0]
            return (y[1] - y[0]) / dt if dt > 0 else np.zeros(self.dim)
        if j == P - 1:
            dt = t[P - 1] - t[P - 2]
            return (y[P - 1] - y[P - 2]) / dt if dt > 0 else np.zeros(self.dim)
        dt = t[j + 1] - t[j - 1]
        return (y[j + 1] - y[j - 1]) / dt if dt > 0 else np.zeros(self.dim)

    def sample(self, t):
        P = len(self._times)
        if P == 0:
            raise RuntimeError("cannot sample empty spline")
        times = self._times
        values = self._values
        if t <= times[0]:
            return values[0].copy()
        if t >= times[P - 1]:
            return values[P - 1].copy()

        # binary search for j with times[j] <= t < times[j+1]
        lo, hi = 0, P - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if times[mid] <= t:
                lo = mid
            else:
                hi = mid
        j = lo

        if self.interp == 'zero':
            return values[j].copy()  # paper Eq. 15

        dt = times[j + 1] - times[j]
        q = (t - times[j]) / dt if dt > 0 else 0.0

        if self.interp == 'linear':
            return (1.0 - q) * values[j] + q * values[j + 1]  # paper Eq. 16

        # cubic Hermite (paper Eqs. 17-23)
        phi_j = self._finite_diff_slope(j)
        phi_j1 = self._finite_diff_slope(j + 1)
        a = 2.0 * q ** 3 - 3.0 * q ** 2 + 1.0
        b = (q ** 3 - 2.0 * q ** 2 + q) * dt
        c = -2.0 * q ** 3 + 3.0 * q ** 2
        d = (q ** 3 - q ** 2) * dt
        return a * values[j] + b * phi_j + c * values[j + 1] + d * phi_j1
