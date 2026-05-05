"""Load, pack, and save MPC simulation results."""

import numpy as np
import pickle
import os
import glob
from datetime import datetime

from configs import DT, IP_FAILURE_ANGLE, IP_CART_BOUND


def pack_mpc_results(states_list, actions_list, costs_list,
                     horizon, recompute_interval, model_length_factor,
                     n_steps, seed):
    """Build results dict for one grid point (compatible with load_all_results)."""
    return {
        'states': states_list,
        'actions': actions_list,
        'costs': costs_list,
        'metadata': {
            'agent_type': 'MPC',
            'n_episodes': len(states_list),
            'n_steps': n_steps,
            'horizon': horizon,
            'n_samples': 500,
            'action_bounds': (-10.0, 10.0),
            'seed': seed,
            'model_length_factor': model_length_factor,
            'recompute_interval': recompute_interval,
            'env_params': {
                'gravity': 9.8,
                'masscart': 1.0,
                'masspole': 0.1,
                'length': 0.5,
                'dt': DT,
            },
            'model_params': {
                'length': 0.5 * model_length_factor,
            },
        },
    }


def save_mpc_results(results, output_dir, timestamp=None):
    """Save one grid point's results dict to a pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    m = results['metadata']
    filename = f"mpc_h{m['horizon']}_r{m['recompute_interval']}_lf{m['model_length_factor']}_n{m['n_episodes']}_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    return filepath


def load_all_results(results_dir):
    """Load all pickle files from results directory.

    Each pickle contains episode states, actions, costs, and metadata
    (produced by pack_mpc_results). Returns a list of summary dicts with
    aggregated statistics per parameter combination.
    """
    pattern = os.path.join(results_dir, "mpc_*.pkl")
    files = glob.glob(pattern)

    all_data = []
    for filepath in files:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)

        metadata = results['metadata']
        total_costs = [np.sum(c) for c in results['costs']]
        dt = metadata['env_params']['dt']

        # Compute episode durations (steps before angle exceeds failure threshold)
        from configs import FAILURE_ANGLE
        durations = []
        for states in results['states']:
            thetas = np.degrees(states[:, 2])
            exceeded = np.where(np.abs(thetas) > FAILURE_ANGLE)[0]
            duration = exceeded[0] if len(exceeded) > 0 else len(thetas)
            durations.append(duration)

        durations_sec = [d * dt for d in durations]
        cost_rates = [c / max(d, dt) for c, d in zip(total_costs, durations_sec)]
        n = len(total_costs)

        all_data.append({
            'horizon': metadata['horizon'],
            'horizon_sec': metadata['horizon'] * dt,
            'recompute_interval': metadata.get('recompute_interval', 1),
            'recompute_sec': metadata.get('recompute_interval', 1) * dt,
            'model_length_factor': metadata['model_length_factor'],
            'dt': dt,
            'total_costs': total_costs,
            'mean_cost': np.mean(total_costs),
            'std_cost': np.std(total_costs),
            'sem_cost': np.std(total_costs) / np.sqrt(n),
            'cost_rates': cost_rates,
            'mean_cost_rate': np.mean(cost_rates),
            'sem_cost_rate': np.std(cost_rates) / np.sqrt(n),
            'durations': durations,
            'durations_sec': durations_sec,
            'mean_duration': np.mean(durations),
            'mean_duration_sec': np.mean(durations_sec),
            'std_duration': np.std(durations),
            'std_duration_sec': np.std(durations_sec),
            'sem_duration_sec': np.std(durations_sec) / np.sqrt(n),
            'n_episodes': metadata['n_episodes'],
        })

    return all_data


def pack_ip_mpc_results(states_list, actions_list, costs_list,
                        horizon, recompute_interval, pole_mass_factor,
                        n_steps, seed):
    """Build results dict for one IP grid point (compatible with load_all_ip_results)."""
    return {
        'states': states_list,
        'actions': actions_list,
        'costs': costs_list,
        'metadata': {
            'agent_type': 'IP_MPC',
            'n_episodes': len(states_list),
            'n_steps': n_steps,
            'horizon': horizon,
            'n_samples': 500,
            'action_bounds': (-3.0, 3.0),
            'seed': seed,
            'pole_mass_factor': pole_mass_factor,
            'recompute_interval': recompute_interval,
            'env': 'InvertedPendulumMJX',
        },
    }


def save_ip_mpc_results(results, output_dir, timestamp=None):
    """Save one IP grid point's results dict to a pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    m = results['metadata']
    filename = (
        f"ip_mpc_h{m['horizon']}_r{m['recompute_interval']}"
        f"_pmf{m['pole_mass_factor']}_n{m['n_episodes']}_{timestamp}.pkl"
    )
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    return filepath


def load_all_ip_results(results_dir):
    """Load all IP pickle files from results directory.

    Mirrors load_all_results but uses IP-specific failure conditions:
      - Pole angle: |states[:, 1]| > IP_FAILURE_ANGLE (theta at index 1)
      - Cart position: |states[:, 0]| > IP_CART_BOUND
    Whichever triggers first defines the episode end.

    Returns a list of summary dicts with aggregated statistics per parameter
    combination, using the same keys as load_all_results for compatibility.
    """
    pattern = os.path.join(results_dir, "ip_mpc_*.pkl")
    files = glob.glob(pattern)

    all_data = []
    for filepath in files:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)

        metadata = results['metadata']
        total_costs = [np.sum(c) for c in results['costs']]

        durations = []
        for states in results['states']:
            angle_fail = np.where(np.abs(states[:, 1]) > IP_FAILURE_ANGLE)[0]
            cart_fail  = np.where(np.abs(states[:, 0]) > IP_CART_BOUND)[0]
            first_fail = min(
                angle_fail[0] if len(angle_fail) else len(states),
                cart_fail[0]  if len(cart_fail)  else len(states),
            )
            durations.append(first_fail)

        durations_sec = [d * DT for d in durations]
        cost_rates = [c / max(d, DT) for c, d in zip(total_costs, durations_sec)]
        n = len(total_costs)

        all_data.append({
            'horizon':            metadata['horizon'],
            'horizon_sec':        metadata['horizon'] * DT,
            'recompute_interval': metadata['recompute_interval'],
            'recompute_sec':      metadata['recompute_interval'] * DT,
            'pole_mass_factor':   metadata['pole_mass_factor'],
            'dt':                 DT,
            'total_costs':        total_costs,
            'mean_cost':          np.mean(total_costs),
            'std_cost':           np.std(total_costs),
            'sem_cost':           np.std(total_costs) / np.sqrt(n),
            'cost_rates':         cost_rates,
            'mean_cost_rate':     np.mean(cost_rates),
            'sem_cost_rate':      np.std(cost_rates) / np.sqrt(n),
            'durations':          durations,
            'durations_sec':      durations_sec,
            'mean_duration':      np.mean(durations),
            'mean_duration_sec':  np.mean(durations_sec),
            'std_duration':       np.std(durations),
            'std_duration_sec':   np.std(durations_sec),
            'sem_duration_sec':   np.std(durations_sec) / np.sqrt(n),
            'n_episodes':         metadata['n_episodes'],
        })

    return all_data


def create_ip_heatmap_data(all_data, pole_mass_factor, value_key='mean_cost_rate'):
    """Create 2D array for IP heatmap from data filtered by pole mass factor.

    Mirrors create_heatmap_data but filters on pole_mass_factor instead of
    model_length_factor.

    Returns (data, horizons_sec, recomputes_sec) shaped [n_recomputes, n_horizons].
    """
    filtered = [d for d in all_data if d['pole_mass_factor'] == pole_mass_factor]
    if not filtered:
        return None, None, None

    horizons_sec   = sorted(set(d['horizon_sec']   for d in filtered))
    recomputes_sec = sorted(set(d['recompute_sec'] for d in filtered))

    data = np.full((len(recomputes_sec), len(horizons_sec)), np.nan)
    for d in filtered:
        h_idx = horizons_sec.index(d['horizon_sec'])
        r_idx = recomputes_sec.index(d['recompute_sec'])
        data[r_idx, h_idx] = d[value_key]

    return data, horizons_sec, recomputes_sec


def filter_by_mpc_settings(all_data, horizon_sec, recompute_sec, tolerance=0.001):
    """Filter data to only include specific MPC settings."""
    return [
        d for d in all_data
        if abs(d['horizon_sec'] - horizon_sec) < tolerance
        and abs(d['recompute_sec'] - recompute_sec) < tolerance
    ]


def create_heatmap_data(all_data, length_factor, value_key='mean_cost'):
    """Create 2D array for heatmap from data filtered by length factor.

    Returns (data, horizons_sec, recomputes_sec) where data is shaped
    [n_recomputes, n_horizons]. Returns (None, None, None) if no matching data.
    """
    filtered = [d for d in all_data if d['model_length_factor'] == length_factor]

    if not filtered:
        return None, None, None

    horizons_sec = sorted(set(d['horizon_sec'] for d in filtered))
    recomputes_sec = sorted(set(d['recompute_sec'] for d in filtered))

    data = np.full((len(recomputes_sec), len(horizons_sec)), np.nan)
    for d in filtered:
        h_idx = horizons_sec.index(d['horizon_sec'])
        r_idx = recomputes_sec.index(d['recompute_sec'])
        data[r_idx, h_idx] = d[value_key]

    return data, horizons_sec, recomputes_sec
