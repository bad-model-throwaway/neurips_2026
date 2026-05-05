"""Derived metrics from sweep data and MPC cost landscape probes."""

import numpy as np
from configs import DT, SEED



def compute_cost_rates(sweep, labels, mismatches):
    """Per-episode cost rate: total_cost / max(duration, DT)."""
    return {
        lab: {
            m: [c / max(d, DT) for c, d in zip(
                sweep['sweep_cost'][lab][m], sweep['sweep_len'][lab][m]
            )]
            for m in mismatches
        }
        for lab in labels
    }


def compute_recompute_intervals(sweep, labels, mismatches):
    """Per-episode average recompute interval: duration / max(n_recomputations, 1)."""
    return {
        lab: {
            m: [d / max(n, 1) for n, d in zip(
                sweep['sweep_recomp'][lab][m], sweep['sweep_len'][lab][m]
            )]
            for m in mismatches
        }
        for lab in labels
    }


def compute_efficiency(sweep, labels, mismatches):
    """Per-episode efficiency score: cost_rate * recompute_rate."""
    efficiency = {}
    for lab in labels:
        efficiency[lab] = {}
        for m in mismatches:
            efficiency[lab][m] = [
                (c / max(d, DT)) * (r / max(d, DT))
                for c, r, d in zip(
                    sweep['sweep_cost'][lab][m],
                    sweep['sweep_recomp'][lab][m],
                    sweep['sweep_len'][lab][m],
                )
            ]
    return efficiency


def compute_rh_traces_sec(sweep):
    """Convert recompute interval traces from steps to seconds."""
    return {
        lab: np.array(traces) * DT
        for lab, traces in sweep['sweep_rh_traces'].items()
    }


def get_episode_lengths(sweep):
    """Episode length: failure time if available, else full duration."""
    return sweep.get('sweep_failure', sweep['sweep_len'])


def probe_cost_landscape(state, model, proposal, evaluation):
    """Run one proposal/evaluate cycle, return first actions and total costs.

    state: current state vector
    model: Dynamics instance (stateless planning model)
    proposal: Proposal instance
    evaluation: Evaluation instance

    Returns (first_actions, evaluations) where first_actions is [n_samples]
    for 1D actions or [n_samples, action_dim] for multi-dimensional.
    """
    proposals = proposal(state)
    trajectories = model.query(state, proposals)
    evals = evaluation(trajectories, proposals)

    # proposals is [n_samples, action_dim, tsteps]
    if proposals.shape[1] == 1:
        first_actions = proposals[:, 0, 0]
    else:
        first_actions = proposals[:, :, 0]

    return first_actions, evals
