import numpy as np

from agents.base import Dynamics

# Try-import cython rollout functions
_HAS_CARTPOLE_CY = False
_HAS_POINTMASS_CY = False

try:
    from agents._cartpole_cy import cartpole_rollout
    _HAS_CARTPOLE_CY = True
except ImportError:
    pass

try:
    from agents._pointmass_cy import pointmass_rollout
    _HAS_POINTMASS_CY = True
except ImportError:
    pass


class PointMass2D(Dynamics):
    """2D point mass subject to conservative force field with tracking target.

    State: [x, y, vx, vy, s]
        x, y: position [m]
        vx, vy: velocity [m/s]
        s: tracking parameter along curve [dimensionless, 0 to 1]
    Action: [fx, fy] control forces [N]

    Units: SI (kg, m, s, N)
    """

    def __init__(self, force_field, curve_func, mass=0.5, dt=0.02,
                 tracking_speed=0.1, curve_scale=1.5, cost_weights=None,
                 noise_std=0.0, stateless=False, initial_state=None, use_cython=True):
        """
        force_field: GPForceField instance
        curve_func: parametric curve function (s, scale) -> (x, y)
        mass: point mass [kg]
        dt: integration timestep [s]
        tracking_speed: rate of s advancement [s⁻¹], 0.1 completes loop in 10 s
        curve_scale: scale parameter for curve [m]
        cost_weights: dict with 'curve', 'tracking', 'control' weights
        noise_std: std of acceleration noise [m/s^2], unmodeled force disturbance
        """
        super().__init__(stateless)
        self.force_field = force_field
        self.curve_func = curve_func
        self.mass = mass
        self.dt = dt
        self.noise_std = noise_std
        self.tracking_speed = tracking_speed
        self.curve_scale = curve_scale

        # Default cost weights
        if cost_weights is None:
            cost_weights = {'curve': 1.0, 'tracking': 1.0, 'control': 0.0}
        self.cost_weights = cost_weights

        # Precompute curve distance lookup table
        self._build_curve_table(curve_func, curve_scale)

        if use_cython and _HAS_POINTMASS_CY:
            self._forward_stateless = self._forward_stateless_cython

        if initial_state is not None:
            self.reset(initial_state)

    @property
    def world_extent(self):
        """Hard boundary of the simulation world [m]."""
        return self._curve_extent

    def _build_curve_table(self, curve_func, curve_scale, n_grid=400, extent=2.0):
        """Precompute minimum distance to curve on a fine grid."""
        self._curve_extent = extent
        self._curve_n = n_grid
        self._curve_step = 2 * extent / (n_grid - 1)

        # Sample curve points
        s_vals = np.linspace(0, 1, 200)
        cx, cy = curve_func(s_vals, scale=curve_scale)
        curve_points = np.stack([cx, cy], axis=1)

        # Compute distance from each grid point to nearest curve point
        grid = np.linspace(-extent, extent, n_grid)
        xx, yy = np.meshgrid(grid, grid)
        positions = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # Chunked to avoid huge intermediate array
        dists = np.empty(positions.shape[0])
        chunk = 1000
        for i in range(0, len(positions), chunk):
            diff = positions[i:i+chunk, np.newaxis, :] - curve_points[np.newaxis, :, :]
            dists[i:i+chunk] = np.min(np.sqrt(np.sum(diff**2, axis=2)), axis=1)

        self._curve_table = dists.reshape(n_grid, n_grid)

    def _curve_distance_lookup(self, positions):
        """Look up precomputed curve distance by nearest grid point. Returns [n]."""
        positions = np.atleast_2d(positions)
        idx = np.round(
            (positions + self._curve_extent) / self._curve_step
        ).astype(int)
        np.clip(idx, 0, self._curve_n - 1, out=idx)
        return self._curve_table[idx[:, 1], idx[:, 0]]

    def cost_function(self, state):
        """Compute combined cost for curve distance and tracking distance."""
        if state.ndim == 1:
            pos = state[:2]
            s = state[4]
        else:
            pos = state[:, :2]
            s = state[:, 4]

        # Curve distance cost (lookup table)
        curve_cost = self._curve_distance_lookup(pos)

        # Tracking target cost
        if state.ndim == 1:
            target = self.curve_func(s, scale=self.curve_scale)
            target = np.array([target[0], target[1]])
        else:
            tx, ty = self.curve_func(s, scale=self.curve_scale)
            target = np.stack([tx, ty], axis=1)
        tracking_cost = self.cost_tracking_distance(pos, target)

        # Combine costs
        cost = (self.cost_weights['curve'] * curve_cost +
                self.cost_weights['tracking'] * tracking_cost)

        # Return scalar for single state, array for batch
        if state.ndim == 1:
            return cost[0]
        return cost

    def _forward_stateless_cython(self, state, actions, params=None):
        """Cython-accelerated forward rollout for batched inputs."""
        if state.ndim == 2:
            ff = self.force_field
            return pointmass_rollout(
                state, actions,
                ff._table_fx.ravel(), ff._table_fy.ravel(),
                ff._table_n, ff._table_extent,
                self._curve_table.ravel(),
                self._curve_n, self._curve_extent,
                self.mass, self.dt,
                self.tracking_speed, self.curve_scale,
                self.cost_weights['curve'], self.cost_weights['tracking'],
                self.cost_weights['control'],
            )
        return super()._forward_stateless(state, actions, params)

    def _step_stateless(self, state, action, params=None):
        """Integrate dynamics with force field and control input."""
        # Unpack state
        if state.ndim == 1:
            x, y, vx, vy, s = state
            ux, uy = action[0], action[1]
        else:
            x, y, vx, vy, s = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4]
            ux, uy = action[:, 0], action[:, 1]

        # Compute force field contribution via lookup table
        if state.ndim == 1:
            forces = self.force_field.force_vectorized(np.array([[x, y]]))
            fx_field, fy_field = forces[0, 0], forces[0, 1]
        else:
            positions = np.stack([x, y], axis=1)
            field_forces = self.force_field.force_vectorized(positions)
            fx_field, fy_field = field_forces[:, 0], field_forces[:, 1]

        # Total acceleration: (field force + control) / mass
        ax = (fx_field + ux) / self.mass
        ay = (fy_field + uy) / self.mass

        # Process noise: unmodeled force disturbance on acceleration
        if self.noise_std > 0:
            if state.ndim == 1:
                ax = ax + np.random.randn() * self.noise_std
                ay = ay + np.random.randn() * self.noise_std
            else:
                ax = ax + np.random.randn(*ax.shape) * self.noise_std
                ay = ay + np.random.randn(*ay.shape) * self.noise_std

        # Euler integration
        next_vx = vx + self.dt * ax
        next_vy = vy + self.dt * ay
        next_x = x + self.dt * next_vx
        next_y = y + self.dt * next_vy

        # Hard boundary: clamp position, zero velocity at walls
        ext = self._curve_extent
        if state.ndim == 1:
            if next_x < -ext:
                next_x, next_vx = -ext, 0.0
            elif next_x > ext:
                next_x, next_vx = ext, 0.0
            if next_y < -ext:
                next_y, next_vy = -ext, 0.0
            elif next_y > ext:
                next_y, next_vy = ext, 0.0
        else:
            hit_lo_x = next_x < -ext
            hit_hi_x = next_x > ext
            hit_lo_y = next_y < -ext
            hit_hi_y = next_y > ext
            next_x = np.clip(next_x, -ext, ext)
            next_y = np.clip(next_y, -ext, ext)
            next_vx = np.where(hit_lo_x | hit_hi_x, 0.0, next_vx)
            next_vy = np.where(hit_lo_y | hit_hi_y, 0.0, next_vy)

        # Advance tracking parameter (wrap to [0, 1])
        next_s = (s + self.dt * self.tracking_speed) % 1.0

        # Pack next state
        if state.ndim == 1:
            next_state = np.array([next_x, next_y, next_vx, next_vy, next_s])
        else:
            next_state = np.stack([next_x, next_y, next_vx, next_vy, next_s], axis=1)

        # Compute cost (includes control effort)
        cost = self.cost_function(next_state)
        control_cost = self.cost_weights['control'] * (ux**2 + uy**2)
        cost = cost + control_cost

        return next_state, cost

    @staticmethod
    def cost_curve_distance(positions, curve_func, curve_scale=1.5, n_samples=100):
        """Compute minimum distance from positions to curve.

        positions: [n, 2] or [2] array of (x, y) positions [m]
        curve_func: parametric curve function (s, scale) -> (x, y)
        curve_scale: scale parameter for curve [m]
        n_samples: number of points to sample along curve for distance computation

        Returns: distances [m]
        """
        positions = np.atleast_2d(positions)

        # Sample curve points
        s_vals = np.linspace(0, 1, n_samples)
        curve_x, curve_y = curve_func(s_vals, scale=curve_scale)
        curve_points = np.stack([curve_x, curve_y], axis=1)  # [n_samples, 2]

        # Compute distances from each position to all curve points
        # positions: [n, 2], curve_points: [n_samples, 2]
        diff = positions[:, np.newaxis, :] - curve_points[np.newaxis, :, :]  # [n, n_samples, 2]
        distances = np.sqrt(np.sum(diff**2, axis=2))  # [n, n_samples]

        # Return minimum distance for each position
        return np.min(distances, axis=1)

    @staticmethod
    def cost_tracking_distance(positions, target):
        """Compute distance from positions to tracking target.

        positions: [n, 2] or [2] array of (x, y) positions [m]
        target: (x, y) tuple or [2] array of target position [m]

        Returns: distances [m]
        """
        positions = np.atleast_2d(positions)
        target = np.asarray(target)
        diff = positions - target
        return np.sqrt(np.sum(diff**2, axis=1))



class CartPoleDynamics(Dynamics):
    """CartPole dynamics for environment and world model.

    Supports batched states [n_samples, state_dim] and actions [n_samples].
    """

    def __init__(self, gravity=9.8, masscart=1.0, masspole=0.1, length=0.5, dt=0.02,
                 noise_std=0.0, stateless=False, initial_state=None, use_cython=True, cost_weights=None):
        super().__init__(stateless)
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.length = length
        self.cost_weights = cost_weights if cost_weights is not None else (1.0, 0.5, 0.01)
        self.dt = dt
        self.noise_std = noise_std

        if use_cython and _HAS_CARTPOLE_CY:
            self._forward_stateless = self._forward_stateless_cython

        if initial_state is not None:
            self.reset(initial_state)

    def cost_function(self, state):
        """Penalize angle deviation and cart displacement (both quadratic)."""
        w_theta, w_x, _ = self.cost_weights
        if state.ndim == 1:
            return w_theta * state[2]**2 + w_x * state[0]**2
        return w_theta * state[:, 2]**2 + w_x * state[:, 0]**2

    def _forward_stateless_cython(self, state, actions, params=None):
        """Cython-accelerated forward rollout for batched inputs."""
        if state.ndim == 2:
            return cartpole_rollout(
                state, actions,
                self.gravity, self.masscart, self.masspole, self.length, self.dt,
                *self.cost_weights,
            )
        return super()._forward_stateless(state, actions, params)

    def _step_stateless(self, state, action, params=None):
        """Integrate dynamics using Euler method. Supports batched inputs."""
        total_mass = self.masscart + self.masspole
        polemass_length = self.masspole * self.length

        # Handle both single and batched states
        if state.ndim == 1:
            x, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]
        else:
            x, x_dot, theta, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]

        u = np.atleast_1d(action).squeeze()

        # Compute accelerations
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (u + polemass_length * theta_dot**2 * sintheta) / total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        # Process noise: unmodeled torque disturbance on angular acceleration
        if self.noise_std > 0:
            if state.ndim == 1:
                thetaacc = thetaacc + np.random.randn() * self.noise_std
            else:
                thetaacc = thetaacc + np.random.randn(*thetaacc.shape) * self.noise_std

        # Euler integration
        if state.ndim == 1:
            next_state = state + self.dt * np.array([x_dot, xacc, theta_dot, thetaacc])
        else:
            derivs = np.stack([x_dot, xacc, theta_dot, thetaacc], axis=1)
            next_state = state + self.dt * derivs

        # State cost (pre-step) plus control effort
        _, _, w_u = self.cost_weights
        cost = self.cost_function(state) + w_u * u**2

        return next_state, cost


def make_perturbation_cartpole(n_steps, force=1.0, duration_sec=0.5, start_frac=0.5):
    """Build a perturbation array: constant force pulse applied to the cart.

    force: pulse magnitude [N]
    duration_sec: pulse duration [s]
    start_frac: fraction of episode at which the pulse begins (0.0–1.0)
    """
    from configs import DT
    perturbation   = np.zeros(n_steps)
    duration_steps = int(duration_sec / DT)
    start_step     = int(start_frac * n_steps)
    end_step       = min(start_step + duration_steps, n_steps)
    perturbation[start_step:end_step] = force
    return perturbation
