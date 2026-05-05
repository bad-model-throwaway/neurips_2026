import numpy as np

def figure_eight(s, scale=1.0):
    """Parametric figure-8 curve. s in [0, 1] completes one loop.

    scale: spatial scale [m], curve spans [-scale, scale] in both dimensions
    Returns: (x, y) positions [m]
    """
    t = 2 * np.pi * s
    x = scale * np.sin(t)
    y = scale * np.sin(t) * np.cos(t)
    return x, y

class GPForceField:
    """Conservative force field derived from GP-sampled potential.

    Samples potential values at inducing points, then computes force
    as negative gradient of RBF-interpolated potential.

    Units: positions in m, potential in J, force in N.
    """

    def __init__(self, n_inducing=25, lengthscale=0.5, amplitude=0.05,
                 extent=2.0, seed=None):
        """
        n_inducing: number of inducing points per dimension (total = n_inducing^2)
        lengthscale: RBF kernel lengthscale [m]
        amplitude: scale of potential variations [J], default gives ~6 N max force
        extent: inducing points span [-extent, extent] in each dimension [m]
        seed: random seed for reproducible potential landscapes
        """
        self.lengthscale = lengthscale
        self.amplitude = amplitude

        # Create grid of inducing points
        grid = np.linspace(-extent, extent, n_inducing)
        xx, yy = np.meshgrid(grid, grid)
        self.inducing_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # Sample potential values from GP prior
        rng = np.random.default_rng(seed)
        n_points = self.inducing_points.shape[0]

        # Compute kernel matrix between inducing points
        K = self._kernel_matrix(self.inducing_points, self.inducing_points)
        K += 1e-6 * np.eye(n_points)  # numerical stability

        # Sample from GP prior: weights that give potential values at inducing points
        L = np.linalg.cholesky(K)
        self.weights = amplitude * L @ rng.standard_normal(n_points)

        # Precompute force lookup table on fine grid
        self._build_force_table(extent, n_grid=400)

    def _kernel(self, p, q):
        """RBF kernel between points p and q."""
        diff = p - q
        return np.exp(-np.sum(diff**2) / (2 * self.lengthscale**2))

    def _kernel_matrix(self, X, Y):
        """Compute kernel matrix between point sets X and Y."""
        n, m = X.shape[0], Y.shape[0]
        K = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                K[i, j] = self._kernel(X[i], Y[j])
        return K

    def potential(self, positions):
        """Compute potential V for batch of positions [n, 2]. Returns [n]."""
        l2 = self.lengthscale**2

        # Compute kernel between all query points and inducing points
        diff = positions[:, np.newaxis, :] - self.inducing_points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)  # [n, m]
        K = np.exp(-dist_sq / (2 * l2))    # [n, m]

        # Potential: V = Σ_i w_i * k(p, x_i)
        return K @ self.weights

    def __call__(self, x, y):
        """Compute force F = -∇V at position (x, y). Returns (fx, fy)."""
        p = np.array([x, y])
        fx, fy = 0.0, 0.0
        l2 = self.lengthscale**2

        for i, xi in enumerate(self.inducing_points):
            k = self._kernel(p, xi)
            # ∇k = k * (xi - p) / l², so -∇V = Σ w_i * k * (p - xi) / l²
            fx += self.weights[i] * k * (p[0] - xi[0]) / l2
            fy += self.weights[i] * k * (p[1] - xi[1]) / l2

        return fx, fy

    def max_force(self, n_grid=50, extent=None):
        """Compute maximum force magnitude over a grid of positions.

        n_grid: number of grid points per dimension
        extent: spatial extent to sample (defaults to inducing point extent)

        Returns: (max_force_magnitude [N], position of max [m])
        """
        if extent is None:
            extent = np.max(np.abs(self.inducing_points))

        # Create grid of positions
        grid = np.linspace(-extent, extent, n_grid)
        xx, yy = np.meshgrid(grid, grid)
        positions = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # Compute forces at all positions
        forces = self.force_vectorized(positions)
        magnitudes = np.sqrt(np.sum(forces**2, axis=1))

        # Find maximum
        max_idx = np.argmax(magnitudes)
        return magnitudes[max_idx], positions[max_idx]

    def _build_force_table(self, extent, n_grid=400):
        """Precompute forces on a fine grid for nearest-neighbor lookup."""
        self._table_extent = extent
        self._table_n = n_grid
        self._table_step = 2 * extent / (n_grid - 1)

        # Evaluate forces at all grid points, chunked to limit peak memory
        grid = np.linspace(-extent, extent, n_grid)
        xx, yy = np.meshgrid(grid, grid)
        positions = np.stack([xx.ravel(), yy.ravel()], axis=1)

        fx = np.empty(positions.shape[0])
        fy = np.empty(positions.shape[0])
        chunk = 1000
        for i in range(0, len(positions), chunk):
            f = self._force_exact(positions[i:i+chunk])
            fx[i:i+chunk] = f[:, 0]
            fy[i:i+chunk] = f[:, 1]
        self._table_fx = fx.reshape(n_grid, n_grid)
        self._table_fy = fy.reshape(n_grid, n_grid)

    def _force_exact(self, positions):
        """Compute forces via full kernel evaluation. Returns [n, 2]."""
        l2 = self.lengthscale**2
        diff = positions[:, np.newaxis, :] - self.inducing_points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        K = np.exp(-dist_sq / (2 * l2))
        weighted_K = K * self.weights[np.newaxis, :]
        return np.sum(weighted_K[:, :, np.newaxis] * diff, axis=1) / l2

    def force_vectorized(self, positions):
        """Look up precomputed forces by nearest grid point. Returns [n, 2]."""
        # Convert positions to grid indices, clamp to valid range
        idx = np.round(
            (positions + self._table_extent) / self._table_step
        ).astype(int)
        np.clip(idx, 0, self._table_n - 1, out=idx)

        fx = self._table_fx[idx[:, 1], idx[:, 0]]
        fy = self._table_fy[idx[:, 1], idx[:, 0]]
        return np.stack([fx, fy], axis=1)

