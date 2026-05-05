# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""Cython-accelerated point mass MPC rollout.

Replaces the Python _forward_stateless loop for batched 2D tracking MPC.
Uses precomputed lookup tables for force field and curve distance.
Physics matches PointMass2D._step_stateless exactly.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, sqrt, fmod, round as cround

ctypedef np.float64_t DTYPE_t

cdef double TWO_PI = 6.283185307179586
np.import_array()


cdef inline void figure_eight(double s, double scale, double *out_x, double *out_y) noexcept nogil:
    """Inline figure-eight curve evaluation."""
    cdef double t = TWO_PI * s
    cdef double sin_t = sin(t)
    out_x[0] = scale * sin_t
    out_y[0] = scale * sin_t * cos(t)


cdef inline double lookup_2d(
        double px, double py,
        double[::1] table, int n_grid,
        double extent, double inv_step) noexcept nogil:
    """Look up a value from a flattened 2D table by nearest grid point."""
    cdef int ix = <int>cround((px + extent) * inv_step)
    cdef int iy = <int>cround((py + extent) * inv_step)

    # Clamp to valid range
    if ix < 0:
        ix = 0
    elif ix >= n_grid:
        ix = n_grid - 1
    if iy < 0:
        iy = 0
    elif iy >= n_grid:
        iy = n_grid - 1

    return table[iy * n_grid + ix]


def pointmass_rollout(
        np.ndarray[DTYPE_t, ndim=2] states,
        np.ndarray[DTYPE_t, ndim=3] actions,
        np.ndarray[DTYPE_t, ndim=1] force_table_fx,
        np.ndarray[DTYPE_t, ndim=1] force_table_fy,
        int force_n_grid, double force_extent,
        np.ndarray[DTYPE_t, ndim=1] curve_dist_table,
        int curve_n_grid, double curve_extent,
        double mass, double dt,
        double tracking_speed,
        double curve_scale,
        double cost_curve_weight, double cost_tracking_weight,
        double cost_control_weight):
    """Roll out point mass dynamics for batched MPC trajectory sampling.

    states: [n_samples, 5] initial states (x, y, vx, vy, s)
    actions: [n_samples, 2, tsteps] action sequences (fx, fy)
    force_table_fx, force_table_fy: flattened [n_grid*n_grid] force lookup
    force_n_grid, force_extent: grid parameters for force table
    curve_dist_table: flattened [n_grid*n_grid] curve distance lookup
    curve_n_grid, curve_extent: grid parameters for curve distance table
    mass, dt: physics parameters
    tracking_speed: rate of s advancement
    curve_scale: figure-eight spatial scale
    cost_*_weight: cost function weights

    Returns (states_out, costs_out) where:
        states_out: [tsteps+1, n_samples, 5] state trajectories
        costs_out: [tsteps+1, n_samples] per-step costs
    """
    cdef int n_samples = states.shape[0]
    cdef int tsteps = actions.shape[2]

    # Pre-allocate output arrays
    cdef np.ndarray[DTYPE_t, ndim=3] states_out = np.empty(
        (tsteps + 1, n_samples, 5), dtype=np.float64,
    )
    cdef np.ndarray[DTYPE_t, ndim=2] costs_out = np.empty(
        (tsteps + 1, n_samples), dtype=np.float64,
    )

    # Typed memoryviews for lookup tables
    cdef double[::1] ff_x = force_table_fx
    cdef double[::1] ff_y = force_table_fy
    cdef double[::1] cd_table = curve_dist_table

    # Precompute lookup step inverses
    cdef double force_inv_step = (force_n_grid - 1) / (2.0 * force_extent)
    cdef double curve_inv_step = (curve_n_grid - 1) / (2.0 * curve_extent)
    cdef double inv_mass = 1.0 / mass

    # Working variables
    cdef double x, y, vx, vy, s, ux, uy
    cdef double fx_field, fy_field, ax_total, ay_total
    cdef double next_x, next_y, next_vx, next_vy, next_s
    cdef double c_curve, c_track
    cdef double tx, ty, tdx, tdy
    cdef int i, t

    # Copy initial states and compute initial costs
    for i in range(n_samples):
        states_out[0, i, 0] = states[i, 0]
        states_out[0, i, 1] = states[i, 1]
        states_out[0, i, 2] = states[i, 2]
        states_out[0, i, 3] = states[i, 3]
        states_out[0, i, 4] = states[i, 4]

        # Initial cost
        x = states[i, 0]
        y = states[i, 1]
        s = states[i, 4]
        c_curve = lookup_2d(x, y, cd_table, curve_n_grid, curve_extent, curve_inv_step)
        figure_eight(s, curve_scale, &tx, &ty)
        tdx = x - tx
        tdy = y - ty
        c_track = sqrt(tdx * tdx + tdy * tdy)
        costs_out[0, i] = cost_curve_weight * c_curve + cost_tracking_weight * c_track

    # Roll out dynamics
    for t in range(tsteps):
        for i in range(n_samples):

            # Read current state
            x = states_out[t, i, 0]
            y = states_out[t, i, 1]
            vx = states_out[t, i, 2]
            vy = states_out[t, i, 3]
            s = states_out[t, i, 4]
            ux = actions[i, 0, t]
            uy = actions[i, 1, t]

            # Force field lookup
            fx_field = lookup_2d(x, y, ff_x, force_n_grid, force_extent, force_inv_step)
            fy_field = lookup_2d(x, y, ff_y, force_n_grid, force_extent, force_inv_step)

            # Total acceleration
            ax_total = (fx_field + ux) * inv_mass
            ay_total = (fy_field + uy) * inv_mass

            # Euler integration
            next_vx = vx + dt * ax_total
            next_vy = vy + dt * ay_total
            next_x = x + dt * next_vx
            next_y = y + dt * next_vy

            # Hard boundary: clamp position, zero velocity at walls
            if next_x < -force_extent:
                next_x = -force_extent
                next_vx = 0.0
            elif next_x > force_extent:
                next_x = force_extent
                next_vx = 0.0
            if next_y < -force_extent:
                next_y = -force_extent
                next_vy = 0.0
            elif next_y > force_extent:
                next_y = force_extent
                next_vy = 0.0

            next_s = fmod(s + dt * tracking_speed, 1.0)

            # Store next state
            states_out[t + 1, i, 0] = next_x
            states_out[t + 1, i, 1] = next_y
            states_out[t + 1, i, 2] = next_vx
            states_out[t + 1, i, 3] = next_vy
            states_out[t + 1, i, 4] = next_s

            # Cost: curve distance + tracking distance + control effort
            c_curve = lookup_2d(next_x, next_y, cd_table, curve_n_grid, curve_extent, curve_inv_step)
            figure_eight(next_s, curve_scale, &tx, &ty)
            tdx = next_x - tx
            tdy = next_y - ty
            c_track = sqrt(tdx * tdx + tdy * tdy)
            costs_out[t + 1, i] = (cost_curve_weight * c_curve
                    + cost_tracking_weight * c_track
                    + cost_control_weight * (ux * ux + uy * uy))

    return states_out, costs_out
