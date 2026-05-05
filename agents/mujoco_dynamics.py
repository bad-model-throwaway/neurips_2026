"""MuJoCo-backed Dynamics subclasses for environments requiring contact physics.

Provides a generic base class (MuJoCoDynamics) that wraps any MJCF XML via the
mujoco C library, plus concrete subclasses for CartPole, Walker, and HumanoidStand.
"""

import os
import warnings
import numpy as np
import mujoco

from agents.base import Dynamics
from agents.rewards import tolerance
from configs import MUJOCO_XML_DIR as _XML_DIR


MISMATCH_FACTORS = {
    'CartPole':              [1.0, 1.5, 2.5, 3.0],   # pole-length factor
    'CartPoleQuadratic':     [1.0, 1.5, 2.5, 3.0],   # pole-length factor, quadratic cost
    'Walker':                [1.0, 1.6, 2.0, 2.6],   # torso-mass factor
    'HumanoidStand':         [1.0, 1.5, 2.0, 2.5],   # torso-mass factor
    'HumanoidBalance':       [1.0, 1.2, 1.4, 1.6],   # planner-model gravity factor
    'HumanoidStandGravity':  [1.0, 1.25, 1.5, 1.75], # planner-model gravity factor
}


class MuJoCoDynamics(Dynamics):
    """Base class for MuJoCo-backed environment dynamics.

    Wraps an MJCF XML model and steps physics via the mujoco C API.
    Subclasses must define:
        xml_path:           path to MJCF XML file
        _state_dim:         size of flat state vector
        _action_dim:        size of action vector
        _state_from_data:   extract flat numpy state from MjData
        _set_data_state:    set MjData qpos/qvel from flat state
        cost_function:      compute cost from state
    """

    def __init__(self, xml_path, n_substeps=1, noise_std=0.0,
                 stateless=False, initial_state=None, pool_size=64):
        super().__init__(stateless)
        self.noise_std = noise_std

        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_data = mujoco.MjData(self._mj_model)
        self._n_substeps = n_substeps

        # Pre-allocate a pool of MjData for batched rollouts
        self._data_pool = [mujoco.MjData(self._mj_model) for _ in range(pool_size)]

        if initial_state is not None:
            self.reset(initial_state)

    @property
    def dt(self):
        """Control timestep = sim_dt * n_substeps."""
        return self._mj_model.opt.timestep * self._n_substeps

    def apply_mismatch(self, factor: float) -> None:
        """Mutate planning model to inject parameter mismatch. Override in subclass."""
        raise NotImplementedError

    def _state_from_data(self, data):
        """Extract flat state vector from MjData. Override in subclass."""
        raise NotImplementedError

    def _set_data_state(self, data, state):
        """Set MjData qpos/qvel from flat state vector. Override in subclass."""
        raise NotImplementedError

    def _step_single(self, data, state, action):
        """Step one environment forward by one control step using given MjData."""
        self._set_data_state(data, state)
        data.ctrl[:] = action
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._mj_model, data)
        next_state = self._state_from_data(data)
        return next_state

    def _step_stateless(self, state, action, params=None):
        """Step physics for single or batched inputs."""
        if state.ndim == 1:
            next_state = self._step_single(self._mj_data, state, action)
            if self.noise_std > 0:
                next_state = next_state + np.random.randn(*next_state.shape) * self.noise_std
            cost = self.cost_function(next_state)
            return next_state, cost

        # Batched: loop over samples
        n_samples = state.shape[0]
        next_states = np.empty_like(state)
        for i in range(n_samples):
            d = self._data_pool[i % len(self._data_pool)]
            next_states[i] = self._step_single(d, state[i], action[i])
        if self.noise_std > 0:
            next_states = next_states + np.random.randn(*next_states.shape) * self.noise_std
        costs = self.cost_function(next_states)
        return next_states, costs

    def _forward_stateless(self, state, actions, params=None):
        """Override base to roll out full trajectories per-sample for efficiency.

        For batched inputs, loops over samples first, then timesteps for each,
        reusing a single MjData per sample to avoid repeated state pack/unpack.
        """
        if state.ndim == 1:
            return self._rollout_single(self._mj_data, state, actions)

        n_samples = state.shape[0]
        tsteps = actions.shape[-1]

        # Pre-allocate output: states [tsteps+1, n_samples, state_dim], costs [tsteps+1, n_samples]
        state_dim = state.shape[1]
        all_states = np.empty((tsteps + 1, n_samples, state_dim))
        all_costs = np.empty((tsteps + 1, n_samples))

        for i in range(n_samples):
            d = self._data_pool[i % len(self._data_pool)]
            s, c = self._rollout_single(d, state[i], actions[i])
            all_states[:, i, :] = s
            all_costs[:, i] = c

        return all_states, all_costs

    def _rollout_single(self, data, state, actions):
        """Roll out a single trajectory. actions shape: [action_dim, tsteps]."""
        tsteps = actions.shape[-1]
        state_dim = state.shape[0]

        states = np.empty((tsteps + 1, state_dim))
        costs = np.empty(tsteps + 1)

        states[0] = state
        costs[0] = self.cost_function(state)

        s = state
        for t in range(tsteps):
            a = actions[..., t]
            s = self._step_single(data, s, a)
            states[t + 1] = s
            costs[t + 1] = self.cost_function(s)

        return states, costs


class MuJoCoCartPoleDynamics(MuJoCoDynamics):
    """CartPole via MuJoCo C API. For benchmarking against native NumPy/Cython.

    State: [x, x_dot, theta, theta_dot] (4D)
    Action: [ctrl] (1D), range [-1, 1] (gear=10 → force [-10, 10] N)
    theta=0 is upright. Contacts disabled.

    Parameters from cartpole.xml:
        cart mass    = 1.0 kg
        pole mass    = 0.1 kg
        pole length  = 1.0 m (fromto 0 0 0 0 0 1)
        cart damping = 5e-4
        pole damping = 2e-6
        gear         = 10
    """

    # Quadratic LQR-balancing weights, state order [x, x_dot, theta, theta_dot].
    _Q_DIAG_DEFAULT = (1.0, 0.1, 10.0, 1.0)
    _R_SCALAR_DEFAULT = 0.1

    def __init__(self, cost_weights=None, noise_std=0.0, stateless=False,
                 initial_state=None, cost_type='tolerance',
                 Q_diag=None, R_scalar=None, **kwargs):
        xml_path = os.path.join(_XML_DIR, 'cartpole.xml')

        # sim_dt=0.01, ctrl_dt=0.02 → n_substeps=2
        super().__init__(xml_path, n_substeps=2, noise_std=noise_std,
                         stateless=stateless, initial_state=initial_state, **kwargs)

        if cost_weights is not None:
            warnings.warn(
                "cost_weights is deprecated for dm_control-aligned CartPole cost",
                DeprecationWarning,
                stacklevel=2,
            )
        self.cost_weights = None

        if cost_type not in ('tolerance', 'quadratic'):
            raise ValueError(f"cost_type must be 'tolerance' or 'quadratic', got {cost_type!r}")
        self.cost_type = cost_type
        self._Q_diag = np.asarray(
            Q_diag if Q_diag is not None else self._Q_DIAG_DEFAULT, dtype=float,
        )
        if self._Q_diag.shape != (4,):
            raise ValueError(f"Q_diag must have shape (4,), got {self._Q_diag.shape}")
        self._R_scalar = float(R_scalar if R_scalar is not None else self._R_SCALAR_DEFAULT)

    def _state_from_data(self, data):
        x = data.qpos[0]         # slider position
        theta = data.qpos[1]     # hinge angle
        x_dot = data.qvel[0]
        theta_dot = data.qvel[1]
        return np.array([x, x_dot, theta, theta_dot])

    def _set_data_state(self, data, state):
        data.qpos[0] = state[0]  # x
        data.qpos[1] = state[2]  # theta
        data.qvel[0] = state[1]  # x_dot
        data.qvel[1] = state[3]  # theta_dot
        mujoco.mj_forward(self._mj_model, data)

    def apply_mismatch(self, factor: float) -> None:
        if factor == 1.0:
            return
        mj = self._mj_model
        pole_bid = mj.body('pole_1').id
        gid = mj.geom('pole_1').id
        mj.body_ipos[pole_bid, 2] = mj.body_ipos[pole_bid, 2] * factor
        mj.body_inertia[pole_bid, 0] = mj.body_inertia[pole_bid, 0] * factor ** 2
        mj.body_inertia[pole_bid, 1] = mj.body_inertia[pole_bid, 1] * factor ** 2
        mj.geom_pos[gid, 2] = mj.geom_pos[gid, 2] * factor
        mj.geom_size[gid, 1] = mj.geom_size[gid, 1] * factor
        mujoco.mj_setConst(mj, mujoco.MjData(mj))

    def cost_function(self, state, ctrl=None):
        state = np.asarray(state, dtype=float)

        if self.cost_type == 'quadratic':
            # g(x,u) = x^T Q x + u^T R u, x = [x, x_dot, theta, theta_dot], theta=0 upright.
            cost = np.einsum('...i,i,...i->...', state, self._Q_diag, state)
            if ctrl is not None:
                ctrl = np.asarray(ctrl, dtype=float)
                cost = cost + self._R_scalar * np.sum(ctrl ** 2, axis=-1)
            return cost

        x = state[..., 0]
        theta = state[..., 2]
        theta_dot = state[..., 3]

        upright = (np.cos(theta) + 1.0) / 2.0
        centered = tolerance(x, bounds=(0.0, 0.0), margin=2.0)
        small_velocity = tolerance(theta_dot, bounds=(0.0, 0.0), margin=5.0)
        small_velocity_rescaled = (1.0 + small_velocity) / 2.0

        reward = upright * centered * small_velocity_rescaled
        return 1.0 - reward


class WalkerDynamics(MuJoCoDynamics):
    """Planar bipedal walker via MuJoCo C API (MJPC walker task residual cost).

    State: [qpos(9), qvel(9), torso_z, torso_zaxis_z, com_vx] = 21D.
    qpos layout is [rootz, rootx, rooty, r_hip, r_knee, r_ankle, l_hip,
    l_knee, l_ankle]. The three trailing slots hold the quantities MJPC's
    walker task reads via the `torso_position`, `torso_zaxis`, and
    `torso_subtreelinvel` sensors — see `_state_from_data` for how they
    are derived from MjData. Carrying them in the state keeps
    `cost_function` a pure function of `state` (plus the applied ctrl).

    Action: 6D motor ctrl ∈ [-1, 1]. Walker XML timestep 0.0025 s →
    n_substeps=4 → ctrl_dt=0.01 (the MJPC anchor).

    MJPC residuals (walker.cc + task.xml, all type=kQuadratic):
        ctrl       (dim=6, weight=0.1)
        torso_z - height_goal      (weight=10)
        torso_zaxis_z - 1          (weight=3)
        com_vx - speed_goal        (weight=1)
    """

    _SENSOR_SLOTS = 3  # torso_z, torso_zaxis_z, com_vx appended to qpos+qvel

    def __init__(self, noise_std=0.0, stateless=False, initial_state=None,
                 speed_goal=0.0, height_goal=1.2, **kwargs):
        xml_path = os.path.join(_XML_DIR, 'walker.xml')

        # sim_dt=0.0025, ctrl_dt=0.01 → n_substeps=4
        super().__init__(xml_path, n_substeps=4, noise_std=noise_std,
                         stateless=stateless, initial_state=initial_state, **kwargs)

        self.speed_goal  = float(speed_goal)
        self.height_goal = float(height_goal)

        self._torso_id = self._mj_model.body('torso').id
        self._nq = self._mj_model.nq  # 9
        self._nv = self._mj_model.nv  # 9

        # Default standing torso_z (MJCF places torso at pos=(0, 0, 1.3)).
        # Kept only as a debug reference — cost now reads the live sensor value.
        data = mujoco.MjData(self._mj_model)
        mujoco.mj_resetData(self._mj_model, data)
        mujoco.mj_forward(self._mj_model, data)
        self._default_torso_z = float(data.xpos[self._torso_id, 2])

    def _state_from_data(self, data):
        """Pack qpos, qvel, and the three MJPC sensor scalars into a flat state.

        Direct MjData access mirrors MJPC's named-sensor reads:
            torso_z       ← data.xpos[torso, 2]              (framepos torso_position[2])
            torso_zaxis_z ← data.xmat[torso, 8]              (framezaxis torso_zaxis[2])
            com_vx        ← data.subtree_linvel[torso, 0]    (subtreelinvel torso_subtreelinvel[0])
        `subtree_linvel` is populated by mj_forward whenever a subtreelinvel
        sensor is present; walker.xml ships one on `torso`.
        """
        torso_z       = float(data.xpos[self._torso_id, 2])
        torso_zaxis_z = float(data.xmat[self._torso_id, 8])
        com_vx        = float(data.subtree_linvel[self._torso_id, 0])
        return np.concatenate([
            data.qpos.copy(), data.qvel.copy(),
            np.array([torso_z, torso_zaxis_z, com_vx]),
        ])

    def _set_data_state(self, data, state):
        """Restore qpos/qvel; mj_forward refreshes the sensor slots."""
        data.qpos[:] = state[:self._nq]
        data.qvel[:] = state[self._nq:self._nq + self._nv]
        mujoco.mj_forward(self._mj_model, data)

    def cost_function(self, state, ctrl=None):
        """MJPC walker residual cost.

        MJPC sums `weight_i * Norm_i(residual_i)` where Norm type 0
        (kQuadratic) is `0.5 * x·x`. Weights come from walker/task.xml:
        ctrl 0.1, height 10, rotation 3, speed 1.

        `ctrl` is optional: during rollout it's the action applied to
        produce `state`; at t=0 (no action yet) pass None and the
        control term drops, matching MJPC's post-reset evaluation where
        `data->ctrl` is zero.
        """
        state = np.asarray(state, dtype=float)
        h_res = state[..., 18] - self.height_goal
        u_res = state[..., 19] - 1.0
        v_res = state[..., 20] - self.speed_goal

        cost = 0.5 * (10.0 * h_res ** 2 + 3.0 * u_res ** 2 + 1.0 * v_res ** 2)

        if ctrl is not None:
            ctrl = np.asarray(ctrl, dtype=float)
            cost = cost + 0.5 * 0.1 * np.sum(ctrl ** 2, axis=-1)

        return cost

    def apply_mismatch(self, factor: float, kind: str = 'torso_mass') -> None:
        """Mutate the planning model along one misspecification axis.

        Axes:
            'torso_mass'     — scale torso body mass.
            'foot_friction'  — scale tangential friction on both foot geoms.
                               MuJoCo combines contact friction as max-of-pair,
                               and only the feet touch the floor, so this is
                               equivalent to scaling floor friction.
        """
        if factor == 1.0:
            return
        mj = self._mj_model
        if kind == 'torso_mass':
            mj.body_mass[self._torso_id] = mj.body_mass[self._torso_id] * factor
        elif kind == 'foot_friction':
            for name in ('right_foot', 'left_foot'):
                gid = mj.geom(name).id
                mj.geom_friction[gid, 0] = mj.geom_friction[gid, 0] * factor
        else:
            raise ValueError(f"unknown mismatch kind: {kind!r}")

    def get_default_initial_state(self):
        """Zero-pose standing: all joint angles at 0, torso at default height.

        Walker knee range is [-150°, 0°] so zero is legal, and the MJCF
        places torso at pos=(0, 0, 1.3), which is already above the
        height goal of 1.2 — no foot-solve needed.
        """
        data = mujoco.MjData(self._mj_model)
        mujoco.mj_resetData(self._mj_model, data)
        mujoco.mj_forward(self._mj_model, data)
        return self._state_from_data(data)

    # Rollout overrides thread the applied ctrl into cost_function so the
    # control-effort term is included during planning (the base class calls
    # cost_function on state alone).

    def _step_stateless(self, state, action, params=None):
        if state.ndim == 1:
            next_state = self._step_single(self._mj_data, state, action)
            if self.noise_std > 0:
                next_state = self._apply_noise_single(self._mj_data, next_state)
            cost = self.cost_function(next_state, ctrl=action)
            return next_state, cost

        n = state.shape[0]
        next_states = np.empty_like(state)
        for i in range(n):
            d = self._data_pool[i % len(self._data_pool)]
            next_states[i] = self._step_single(d, state[i], action[i])
        if self.noise_std > 0:
            for i in range(n):
                d = self._data_pool[i % len(self._data_pool)]
                next_states[i] = self._apply_noise_single(d, next_states[i])
        costs = self.cost_function(next_states, ctrl=action)
        return next_states, costs

    def _rollout_single(self, data, state, actions):
        tsteps = actions.shape[-1]
        state_dim = state.shape[0]

        states = np.empty((tsteps + 1, state_dim))
        costs = np.empty(tsteps + 1)

        states[0] = state
        costs[0] = self.cost_function(state)

        s = state
        for t in range(tsteps):
            a = actions[..., t]
            s = self._step_single(data, s, a)
            states[t + 1] = s
            costs[t + 1] = self.cost_function(s, ctrl=a)

        return states, costs

    def _apply_noise_single(self, data, state):
        """Noise qpos/qvel only; re-derive sensor slots via mj_forward."""
        noise = np.random.randn(self._nq + self._nv) * self.noise_std
        state[:self._nq + self._nv] += noise
        self._set_data_state(data, state)
        return self._state_from_data(data)


class HumanoidStandDynamics(MuJoCoDynamics):
    """27-DOF humanoid with MJPC humanoid/stand residual cost.

    Two initial-state modes selectable via ``mode``:
        'standup' (default) — supine on the floor, planner must stand up.
        'balance'           — MJCF upright pose, planner maintains balance.
    Both modes share the same residual cost; only the default pose differs.

    State: [qpos(28), qvel(27), head_z, feet_avg_z, com_x, com_y,
            com_vel_x, com_vel_y, feet_avg_x, feet_avg_y] = 63D.
    The 8 trailing slots cache quantities MJPC's stand.cc reads via
    SensorByName (head_position, sp0..sp3, torso_subtreecom,
    torso_subtreelinvel) — see _state_from_data for the per-slot source.
    Carrying them in the state keeps cost_function a pure function of
    (state, ctrl), mirroring WalkerDynamics' pattern at 8 slots instead of 3.

    Action: 21D motor ctrl ∈ [-1, 1]. XML timestep 0.005 s → n_substeps=3 →
    ctrl_dt=0.015 s (MJPC's agent_timestep for stand).

    MJPC residuals (stand.cc + task.xml user sensors — 5 terms):
        Height    (dim=1,  norm=kSmoothAbsLoss p=0.1, w=100)
                  r = head_z − 0.25·Σᵢ sp_i_z − height_goal
        Balance   (dim=1,  norm=kSmoothAbsLoss p=0.1, w=50)
                  r = ‖ feet_avg_xy − (com_xy + 0.2·com_vel_xy) ‖₂
        CoM Vel.  (dim=2,  norm=kQuadratic,           w=10)
                  r = com_vel_xy
        Joint Vel (dim=21, norm=kQuadratic,           w=0.01)
                  r = qvel[6:]
        Control   (dim=21, norm=kCosh p=0.3,          w=0.025)
                  r = ctrl

    Total step cost: C = Σᵢ wᵢ · nᵢ(rᵢ)  (risk-neutral R=0 — paper eq. 4).

    Howell 2022 Appendix B lists 6 humanoid cost terms (including a
    "torso-position + CoM alignment" term w=1); current MJPC stand.cc has 5.
    We follow live stand.cc, not the paper.
    """

    _SENSOR_SLOTS = 8  # head_z, feet_avg_z, com_xy(2), com_vel_xy(2), feet_avg_xy(2)
    _SP_SITES = ('sp0', 'sp1', 'sp2', 'sp3')

    def __init__(self, noise_std=0.0, stateless=False, initial_state=None,
                 height_goal=1.4, mode='standup', **kwargs):
        if mode not in ('standup', 'balance'):
            raise ValueError(
                f"mode must be 'standup' or 'balance', got {mode!r}"
            )
        self.mode = mode

        xml_path = os.path.join(_XML_DIR, 'humanoid', 'stand', 'task.xml')

        # sim_dt=0.005, n_substeps=3 → ctrl_dt=0.015 (MJPC agent_timestep).
        super().__init__(xml_path, n_substeps=3, noise_std=noise_std,
                         stateless=stateless, initial_state=initial_state, **kwargs)

        self.height_goal = float(height_goal)

        self._torso_id = self._mj_model.body('torso').id
        self._head_id  = self._mj_model.body('head').id
        self._sp_ids   = np.array(
            [self._mj_model.site(name).id for name in self._SP_SITES]
        )
        self._nq = self._mj_model.nq  # 28
        self._nv = self._mj_model.nv  # 27

        # Sensor-slot indices into the flat state vector — shared between
        # cost_function and any external consumer (e.g. test predicates).
        _base = self._nq + self._nv
        self.HEAD_Z_IDX      = _base + 0
        self.FEET_AVG_Z_IDX  = _base + 1
        self.COM_X_IDX       = _base + 2
        self.COM_Y_IDX       = _base + 3
        self.COM_VEL_X_IDX   = _base + 4
        self.COM_VEL_Y_IDX   = _base + 5
        self.FEET_AVG_X_IDX  = _base + 6
        self.FEET_AVG_Y_IDX  = _base + 7

        # Default standing torso_z (MJCF places torso at pos=(0, 0, 1.282)).
        # Debug reference only; cost_function reads the live sensor values.
        data = mujoco.MjData(self._mj_model)
        mujoco.mj_resetData(self._mj_model, data)
        mujoco.mj_forward(self._mj_model, data)
        self._default_torso_z = float(data.xpos[self._torso_id, 2])

    def _state_from_data(self, data):
        """Pack qpos, qvel, and the 8 MJPC sensor scalars into a flat state.

        Direct MjData access mirrors MJPC's SensorByName reads in stand.cc:
            head_z        ← data.xpos[head, 2]                (framepos head_position[2])
            feet_avg_z    ← 0.25 · Σᵢ data.site_xpos[sp_i, 2]  (framepos sp0..sp3[2])
            com_xy        ← data.subtree_com[torso, :2]       (subtreecom torso_subtreecom[:2])
            com_vel_xy    ← data.subtree_linvel[torso, :2]    (subtreelinvel torso_subtreelinvel[:2])
            feet_avg_xy   ← 0.25 · Σᵢ data.site_xpos[sp_i, :2] (framepos sp0..sp3[:2])
        subtree_com / subtree_linvel are populated by mj_forward when a
        matching sensor is declared in the MJCF; our stand/task.xml ships
        both on `torso`.
        """
        head_z      = float(data.xpos[self._head_id, 2])
        sp_xyz      = data.site_xpos[self._sp_ids]         # (4, 3)
        feet_avg    = 0.25 * sp_xyz.sum(axis=0)            # (3,) — x, y, z
        com_xyz     = data.subtree_com[self._torso_id]     # (3,)
        com_vel_xyz = data.subtree_linvel[self._torso_id]  # (3,)

        extras = np.array([
            head_z,
            feet_avg[2],
            com_xyz[0], com_xyz[1],
            com_vel_xyz[0], com_vel_xyz[1],
            feet_avg[0], feet_avg[1],
        ])
        return np.concatenate([data.qpos.copy(), data.qvel.copy(), extras])

    def _set_data_state(self, data, state):
        """Restore qpos/qvel; mj_forward refreshes the 8 sensor slots."""
        data.qpos[:] = state[:self._nq]
        data.qvel[:] = state[self._nq:self._nq + self._nv]
        mujoco.mj_forward(self._mj_model, data)

    def cost_function(self, state, ctrl=None):
        """MJPC humanoid-stand residual cost (stand.cc::Residual, 5 terms).

        Weights and norm types from task.xml `user=` sensor fields; norm
        formulas from mjpc/norm.cc. `ctrl` is optional: at t=0 (no action
        yet) pass None and the control term drops, matching MJPC's
        post-reset evaluation where `data->ctrl` is zero.
        """
        state = np.asarray(state, dtype=float)
        qvel     = state[..., self._nq:self._nq + self._nv]
        head_z      = state[..., self._nq + self._nv + 0]
        feet_avg_z  = state[..., self._nq + self._nv + 1]
        com_xy      = state[..., self._nq + self._nv + 2:self._nq + self._nv + 4]
        com_vel_xy  = state[..., self._nq + self._nv + 4:self._nq + self._nv + 6]
        feet_avg_xy = state[..., self._nq + self._nv + 6:self._nq + self._nv + 8]

        # Residual 0 — Height: head_z − 0.25·Σ sp_i_z − height_goal  (scalar)
        r_height = head_z - feet_avg_z - self.height_goal
        # kSmoothAbsLoss(p=0.1): Σ (√(x² + p²) − p), per-element; here x is scalar
        p_h = 0.1
        n_height = np.sqrt(r_height ** 2 + p_h ** 2) - p_h

        # Residual 1 — Balance: ‖ feet_avg_xy − (com_xy + kFallTime·com_vel_xy) ‖₂
        # kFallTime = 0.2 (stand.cc).
        capture_point_xy = com_xy + 0.2 * com_vel_xy
        balance_vec = feet_avg_xy - capture_point_xy          # (..., 2)
        r_balance   = np.linalg.norm(balance_vec, axis=-1)     # (...,)  — scalar residual
        p_b = 0.1
        n_balance = np.sqrt(r_balance ** 2 + p_b ** 2) - p_b

        # Residual 2 — CoM velocity xy.  kQuadratic: 0.5·Σ x²
        n_com_vel = 0.5 * np.sum(com_vel_xy ** 2, axis=-1)

        # Residual 3 — Joint velocity qvel[6:] (skip 6 free-joint dof).  kQuadratic.
        n_joint_vel = 0.5 * np.sum(qvel[..., 6:] ** 2, axis=-1)

        cost = (
            100.0  * n_height     # Height weight
            + 50.0 * n_balance    # Balance weight
            + 10.0 * n_com_vel    # CoM Vel. weight
            + 0.01 * n_joint_vel  # Joint Vel. weight
        )

        # Residual 4 — Control.  kCosh(p=0.3): Σ p²·(cosh(x_i/p) − 1), per-element.
        if ctrl is not None:
            ctrl = np.asarray(ctrl, dtype=float)
            p_c = 0.3
            n_ctrl = np.sum(p_c ** 2 * (np.cosh(ctrl / p_c) - 1.0), axis=-1)
            cost = cost + 0.025 * n_ctrl  # Control weight

        return cost

    def apply_mismatch(self, factor: float, kind: str = 'torso_mass') -> None:
        """Mutate the planning model along one misspecification axis.

        Axes:
            'torso_mass'     — scale torso body mass.
            'foot_friction'  — scale tangential friction on all four foot geoms.
            'gravity'        — scale gravity magnitude in the planner's model.
                               factor>1 = planner expects heavier world.
        """
        if factor == 1.0:
            return
        mj = self._mj_model
        if kind == 'torso_mass':
            mj.body_mass[self._torso_id] = mj.body_mass[self._torso_id] * factor
        elif kind == 'foot_friction':
            for name in ('foot1_right', 'foot2_right', 'foot1_left', 'foot2_left'):
                gid = mj.geom(name).id
                mj.geom_friction[gid, 0] = mj.geom_friction[gid, 0] * factor
        elif kind == 'gravity':
            mj.opt.gravity[:] = mj.opt.gravity[:] * factor
        else:
            raise ValueError(f"unknown mismatch kind: {kind!r}")

    def get_default_initial_state(self):
        """Pose depends on ``self.mode``.

        'standup': supine (lying on back) on the floor — torso at z≈0.10 m,
            rotated -90° about Y so the humanoid lies face-up, all joint
            angles and velocities zero.
        'balance': MJCF default upright pose — torso at z=1.282, head at
            1.472, feet sp sites at 0.027; head-feet height 1.445 m so
            the Height residual is +0.045 m at init.
        """
        data = mujoco.MjData(self._mj_model)
        mujoco.mj_resetData(self._mj_model, data)
        if self.mode == 'standup':
            # Supine free-joint pose: low z, rotated onto back
            data.qpos[0] = 0.0   # x
            data.qpos[1] = 0.0   # y
            data.qpos[2] = 0.10  # z — torso capsule r=0.07, small margin
            # Quaternion for -90° about Y: (cos(-π/4), 0, sin(-π/4), 0)
            c = np.cos(np.pi / 4)
            s = np.sin(np.pi / 4)
            data.qpos[3] = c     # qw
            data.qpos[4] = 0.0   # qx
            data.qpos[5] = -s    # qy
            data.qpos[6] = 0.0   # qz
            # Joint angles qpos[7:28] already zero from mj_resetData
        # 'balance' uses mj_resetData's MJCF default (upright) pose as-is
        mujoco.mj_forward(self._mj_model, data)
        return self._state_from_data(data)

    # Rollout overrides thread the applied ctrl into cost_function so the
    # control-effort term is included during planning (the base class calls
    # cost_function on state alone).

    def _step_stateless(self, state, action, params=None):
        if state.ndim == 1:
            next_state = self._step_single(self._mj_data, state, action)
            if self.noise_std > 0:
                next_state = self._apply_noise_single(self._mj_data, next_state)
            cost = self.cost_function(next_state, ctrl=action)
            return next_state, cost

        n = state.shape[0]
        next_states = np.empty_like(state)
        for i in range(n):
            d = self._data_pool[i % len(self._data_pool)]
            next_states[i] = self._step_single(d, state[i], action[i])
        if self.noise_std > 0:
            for i in range(n):
                d = self._data_pool[i % len(self._data_pool)]
                next_states[i] = self._apply_noise_single(d, next_states[i])
        costs = self.cost_function(next_states, ctrl=action)
        return next_states, costs

    def _rollout_single(self, data, state, actions):
        tsteps = actions.shape[-1]
        state_dim = state.shape[0]

        states = np.empty((tsteps + 1, state_dim))
        costs = np.empty(tsteps + 1)

        states[0] = state
        costs[0] = self.cost_function(state)

        s = state
        for t in range(tsteps):
            a = actions[..., t]
            s = self._step_single(data, s, a)
            states[t + 1] = s
            costs[t + 1] = self.cost_function(s, ctrl=a)

        return states, costs

    def _apply_noise_single(self, data, state):
        """Noise qpos/qvel only; re-derive sensor slots via mj_forward."""
        noise = np.random.randn(self._nq + self._nv) * self.noise_std
        state[:self._nq + self._nv] += noise
        self._set_data_state(data, state)
        return self._state_from_data(data)
