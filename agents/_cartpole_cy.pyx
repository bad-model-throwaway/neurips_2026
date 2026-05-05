# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""Cython-accelerated cartpole MPC rollout.

Replaces the Python _forward_stateless loop for batched MPC trajectory sampling.
Physics matches CartPoleDynamics._step_stateless exactly (Euler integration).
Cost weights (w_theta, w_x, w_u) must match CartPoleDynamics.cost_weights.
All state costs are quadratic: w_theta * theta^2 + w_x * x^2.
"""

import numpy as np
cimport numpy as np
from libc.math cimport cos, sin

ctypedef np.float64_t DTYPE_t
np.import_array()


def cartpole_rollout(
        np.ndarray[DTYPE_t, ndim=2] states,
        np.ndarray[DTYPE_t, ndim=3] actions,
        double gravity, double masscart, double masspole,
        double length, double dt,
        double w_theta, double w_x, double w_u):
    """Roll out cartpole dynamics for batched MPC trajectory sampling.

    Replicates the _forward_stateless + _step_stateless loop for CartPoleDynamics,
    returning full state trajectories and per-step costs.

    states: [n_samples, 4] initial states (x, x_dot, theta, theta_dot)
    actions: [n_samples, 1, horizon] action sequences
    gravity, masscart, masspole, length, dt: physics parameters
    w_theta, w_x, w_u: cost weights for theta^2, x^2, u^2

    Returns (states_out, costs_out) where:
        states_out: [horizon+1, n_samples, 4] state trajectories
        costs_out: [horizon+1, n_samples] per-step costs
    """
    cdef int n_samples = states.shape[0]
    cdef int horizon = actions.shape[2]

    # Pre-allocate output arrays
    cdef np.ndarray[DTYPE_t, ndim=3] states_out = np.empty(
        (horizon + 1, n_samples, 4), dtype=np.float64,
    )
    cdef np.ndarray[DTYPE_t, ndim=2] costs_out = np.empty(
        (horizon + 1, n_samples), dtype=np.float64,
    )

    # Precompute constants
    cdef double total_mass = masscart + masspole
    cdef double polemass_length = masspole * length

    # Working variables
    cdef double x, x_dot, theta, theta_dot, u
    cdef double costheta, sintheta, temp, thetaacc, xacc
    cdef int i, t

    # Copy initial states into output and compute initial costs
    for i in range(n_samples):
        states_out[0, i, 0] = states[i, 0]
        states_out[0, i, 1] = states[i, 1]
        states_out[0, i, 2] = states[i, 2]
        states_out[0, i, 3] = states[i, 3]

        # Initial state cost (no control effort)
        costs_out[0, i] = w_theta * states[i, 2] * states[i, 2] + w_x * states[i, 0] * states[i, 0]

    # Roll out dynamics
    for t in range(horizon):
        for i in range(n_samples):

            # Read current state
            x = states_out[t, i, 0]
            x_dot = states_out[t, i, 1]
            theta = states_out[t, i, 2]
            theta_dot = states_out[t, i, 3]
            u = actions[i, 0, t]

            # Compute accelerations (matches CartPoleDynamics._step_stateless)
            costheta = cos(theta)
            sintheta = sin(theta)
            temp = (u + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
            thetaacc = (gravity * sintheta - costheta * temp) / (
                length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass)
            )
            xacc = temp - polemass_length * thetaacc * costheta / total_mass

            # Euler integration
            states_out[t + 1, i, 0] = x + dt * x_dot
            states_out[t + 1, i, 1] = x_dot + dt * xacc
            states_out[t + 1, i, 2] = theta + dt * theta_dot
            states_out[t + 1, i, 3] = theta_dot + dt * thetaacc

            # Pre-step state cost plus control effort
            costs_out[t + 1, i] = w_theta * theta * theta + w_x * x * x + w_u * u * u

    return states_out, costs_out
