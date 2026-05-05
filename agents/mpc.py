"""MPC components: spline-PS proposal, evaluation, decision, and factory."""
import numpy as np

from configs import DT, ENV_DT
from agents.base import Proposal, Evaluation, Decision, Agent
from agents.adaptation import ODEStepAdaptation, CostErrorAdaptation, make_adapter
from agents.mujoco_dynamics import (
    MuJoCoCartPoleDynamics, WalkerDynamics, HumanoidStandDynamics,
)
from agents.spline import TimeSpline


class MPCEvaluation(Evaluation):
    """Sum-of-costs trajectory evaluation (generic, works for any environment)."""

    def __call__(self, trajectories, proposals):
        _, costs = trajectories
        return np.sum(costs, axis=0)


class SplinePSProposal(Proposal):
    """MJPC-aligned Predictive Sampling proposal: spline-parameterized nominal + per-knot Gaussian noise."""

    def __init__(self, action_dim, tsteps, n_samples, dt,
                 ctrl_low, ctrl_high,
                 P=3, sigma=0.1, interp='cubic',
                 sigma2=0.0, mix_prob=0.0,
                 include_nominal=True, clip=True):
        self.action_dim = int(action_dim)
        self.tsteps = int(tsteps)
        self.n_samples = int(n_samples)
        self.dt = float(dt)
        self.ctrl_low = np.asarray(ctrl_low, dtype=float).reshape(-1)
        self.ctrl_high = np.asarray(ctrl_high, dtype=float).reshape(-1)
        if self.ctrl_low.shape != (self.action_dim,) or self.ctrl_high.shape != (self.action_dim,):
            raise ValueError("ctrl_low/ctrl_high must have shape (action_dim,)")
        self.P = int(P)
        self.sigma = float(sigma)
        self.interp = interp
        self.sigma2 = float(sigma2)
        self.mix_prob = float(mix_prob)
        self.include_nominal = bool(include_nominal)
        self.clip = bool(clip)
        self.recompute_interval = 1

        self._scale = 0.5 * (self.ctrl_high - self.ctrl_low)  # MJPC planner.cc:342-345
        self._knot_times = self._compute_knot_times(self.tsteps, self.dt, self.P, self.interp)
        self.plan = self._zero_spline()
        self._candidate_splines = []

    @staticmethod
    def _compute_knot_times(tsteps, dt, P, interp):
        time_horizon = (tsteps - 1) * dt
        if interp == 'zero':
            shift = time_horizon / P if P > 0 else 0.0  # MJPC planner.cc:304
        else:
            shift = time_horizon / (P - 1) if P > 1 else 0.0  # MJPC planner.cc:306
        return np.array([i * shift for i in range(P)], dtype=float)

    def _zero_spline(self):
        s = TimeSpline(self.action_dim, self.interp)
        for t in self._knot_times:
            s.add_knot(float(t), np.zeros(self.action_dim))
        return s

    def update_parameters(self, parameters):
        if 'recompute_interval' in parameters:
            self.recompute_interval = int(parameters['recompute_interval'])
        if 'horizon' in parameters and int(parameters['horizon']) != self.tsteps:
            old_plan = self.plan
            old_t_max = float(self._knot_times[-1])
            self.tsteps = int(parameters['horizon'])
            self._knot_times = self._compute_knot_times(self.tsteps, self.dt, self.P, self.interp)
            new_plan = TimeSpline(self.action_dim, self.interp)
            for t in self._knot_times:
                new_plan.add_knot(float(t), old_plan.sample(min(float(t), old_t_max)))
            self.plan = new_plan

    def _render(self, spline):
        grid = np.arange(self.tsteps) * self.dt
        out = np.empty((self.action_dim, self.tsteps))
        for t_idx, t in enumerate(grid):
            out[:, t_idx] = spline.sample(float(t))
        if self.clip:
            out = np.clip(out, self.ctrl_low[:, None], self.ctrl_high[:, None])  # policy.cc:58
        return out

    def __call__(self, state):
        N = self.n_samples
        actions = np.empty((N, self.action_dim, self.tsteps))
        self._candidate_splines = [None] * N

        for i in range(N):
            if self.include_nominal and i == 0:  # MJPC planner.cc:374; paper Alg. 5 line 9
                cand = self.plan.copy()
            else:
                sigma_eff = self.sigma
                if self.sigma2 > 0.0 and np.random.rand() < self.mix_prob:  # MJPC planner.cc:335-338
                    sigma_eff = self.sigma2
                cand = TimeSpline(self.action_dim, self.interp)
                base_knots = self.plan.knots
                for k_idx, t in enumerate(self._knot_times):
                    noise = np.random.randn(self.action_dim) * (self._scale * sigma_eff)  # planner.cc:342-345
                    perturbed = base_knots[k_idx] + noise
                    if self.clip:
                        perturbed = np.clip(perturbed, self.ctrl_low, self.ctrl_high)  # planner.cc:347
                    cand.add_knot(float(t), perturbed)
            self._candidate_splines[i] = cand
            actions[i] = self._render(cand)

        return actions

    def advance_nominal(self, winner_spline):
        """Non-sliding resample (MJPC planner.cc:295-322). Paper Alg. 5 line 7."""
        R = self.recompute_interval
        t_shift = R * self.dt
        new_plan = TimeSpline(self.action_dim, self.interp)
        for t in self._knot_times:
            new_plan.add_knot(float(t), winner_spline.sample(float(t + t_shift)))
        self.plan = new_plan


class SplinePSArgminDecision(Decision):
    """Argmin update: pick min-cost rollout and warm-start via non-sliding resample."""

    def __init__(self, proposal):
        self.proposal = proposal

    def __call__(self, proposals, trajectories, evaluations, n_actions=1):
        best_idx = int(np.argmin(np.asarray(evaluations)))  # MJPC planner.cc:184-188,204; paper Alg. 5 line 17
        winner_spline = self.proposal._candidate_splines[best_idx]
        self.proposal.advance_nominal(winner_spline)

        actions = [proposals[best_idx, :, t] for t in range(n_actions)]
        return actions, best_idx


PROPOSAL_CONFIGS = {
    'cartpole': {
        'proposal': 'spline_ps',
        'N': 30,
        'proposal_kwargs': {
            'P': 3,
            'sigma': 0.3,
            'interp': 'cubic',
            'include_nominal': True,
            'clip': True,
        },
        'decision': 'spline_ps_argmin',
    },
    'cartpole_quadratic': {
        'proposal': 'spline_ps',
        'N': 30,
        'proposal_kwargs': {
            'P': 3,
            'sigma': 0.3,
            'interp': 'cubic',
            'include_nominal': True,
            'clip': True,
        },
        'decision': 'spline_ps_argmin',
        'env_kwargs': {
            'cost_type': 'quadratic',
            'Q_diag':   (1.0, 0.1, 3.0, 1.0),
            'R_scalar': 0.1,
        },
    },
    'walker': {
        'proposal': 'spline_ps',
        'N': 30,
        'proposal_kwargs': {
            'P': 3,
            'sigma': 0.5,
            'interp': 'cubic',
            'include_nominal': True,
            'clip': True,
        },
        'decision': 'spline_ps_argmin',
        'env_kwargs': {'speed_goal': 1.5},
    },
    'humanoid_stand': {
        'proposal': 'spline_ps',
        'N': 30,
        'proposal_kwargs': {
            'P': 3,
            'sigma': 0.25,
            'interp': 'cubic',
            'include_nominal': True,
            'clip': True,
        },
        'decision': 'spline_ps_argmin',
    },
    'humanoid_balance': {
        'proposal': 'spline_ps',
        'N': 30,
        'proposal_kwargs': {
            'P': 3,
            'sigma': 0.25,
            'interp': 'cubic',
            'include_nominal': True,
            'clip': True,
        },
        'decision': 'spline_ps_argmin',
        'env_kwargs': {'mode': 'balance'},
        'mismatch_kind': 'gravity',
    },
    'humanoid_stand_gravity': {
        'proposal': 'spline_ps',
        'N': 30,
        'proposal_kwargs': {
            'P': 3,
            'sigma': 0.25,
            'interp': 'cubic',
            'include_nominal': True,
            'clip': True,
        },
        'decision': 'spline_ps_argmin',
        'mismatch_kind': 'gravity',
    },
}

_ENV_CLASSES = {
    'cartpole':               MuJoCoCartPoleDynamics,
    'cartpole_quadratic':     MuJoCoCartPoleDynamics,
    'walker':                 WalkerDynamics,
    'humanoid_stand':         HumanoidStandDynamics,
    'humanoid_balance':       HumanoidStandDynamics,
    'humanoid_stand_gravity': HumanoidStandDynamics,
}

_ACTION_DIMS = {
    'cartpole':               1,
    'cartpole_quadratic':     1,
    'walker':                 6,
    'humanoid_stand':         21,
    'humanoid_balance':       21,
    'humanoid_stand_gravity': 21,
}

_ENV_BASE = {
    'cartpole':               'cartpole',
    'cartpole_quadratic':     'cartpole_quadratic',
    'walker':                 'walker',
    'humanoid_stand':         'humanoid_stand',
    'humanoid_balance':       'humanoid_balance',
    'humanoid_stand_gravity': 'humanoid_stand_gravity',
}


def make_mpc(env_name, H, R, N=None, mismatch_factor=1.0, proposal=None,
             proposal_kwargs=None, decision=None, **kwargs):
    """Build a MuJoCo planning agent for env_name.

    Proposal config is resolved as: explicit args > PROPOSAL_CONFIGS[env_name].
    Unknown env_name raises ValueError — no silent fallback.
    """
    if env_name not in _ENV_CLASSES:
        raise ValueError(f"Unknown env '{env_name}'. Choose from {list(_ENV_CLASSES)}")

    cfg = PROPOSAL_CONFIGS[env_name]
    _proposal  = proposal        if proposal        is not None else cfg['proposal']
    _N         = N               if N               is not None else cfg['N']
    if proposal_kwargs is not None:
        _prop_kw = dict(proposal_kwargs)
    elif proposal is not None:
        _prop_kw = {}
    else:
        _prop_kw = dict(cfg['proposal_kwargs'])
    _decision = decision if decision is not None else cfg.get('decision', 'spline_ps_argmin')

    env_kwargs = cfg.get('env_kwargs', {}) or {}
    model = _ENV_CLASSES[env_name](stateless=True, **env_kwargs)

    if mismatch_factor != 1.0:
        mismatch_kind = cfg.get('mismatch_kind')
        if mismatch_kind is not None:
            model.apply_mismatch(mismatch_factor, kind=mismatch_kind)
        else:
            model.apply_mismatch(mismatch_factor)

    action_dim = _ACTION_DIMS[env_name]

    if _proposal != 'spline_ps':
        raise ValueError(f"Unknown proposal '{_proposal}'. Only 'spline_ps' is supported.")

    mj = model._mj_model
    ctrl_low = np.asarray(mj.actuator_ctrlrange[:, 0], dtype=float)
    ctrl_high = np.asarray(mj.actuator_ctrlrange[:, 1], dtype=float)
    env_dt = ENV_DT.get(_ENV_BASE[env_name], DT)
    proposal_obj = SplinePSProposal(
        action_dim=action_dim, tsteps=H, n_samples=_N, dt=env_dt,
        ctrl_low=ctrl_low, ctrl_high=ctrl_high, **_prop_kw
    )
    if _decision != 'spline_ps_argmin':
        raise ValueError(f"proposal='spline_ps' requires decision='spline_ps_argmin', got {_decision!r}")
    decision_obj = SplinePSArgminDecision(proposal=proposal_obj)

    parameters = {'recompute_interval': R, 'horizon': H}
    return Agent(proposal_obj, model, MPCEvaluation(), decision_obj,
                 parameters=parameters)
