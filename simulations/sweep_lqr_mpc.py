"""(H, R) grid sweep over scalar-LQR MPC under mismatch + noise + sampling.

Tests the hypothesis that the upper-H stability boundary observed in the
MuJoCo cost-rate heatmaps (but absent from the deterministic-Riccati LQR
heatmap) emerges once the planner uses sample-based action search and/or
noisy lookahead rollouts.

Three mechanism configurations are run, each at two action-search dimensions:

    (no_noise,    P=3 cubic)   (no_noise,    P=H zero)
    (model_noise, P=3 cubic)   (model_noise, P=H zero)
    (both_noise,  P=3 cubic)   (both_noise,  P=H zero)

Mismatch is fixed at (a_hat, b_hat) = (1.2, 0.9), matching one panel from
the analytic Row A in simulate_LQR_mismatch_regions.py.

Run all six sweeps and save pickles:

    python -m simulations.sweep_lqr_mpc

Render the 3x2 figure from cached pickles:

    python -m simulations.sweep_lqr_mpc --plot
"""

import os
import pickle
import argparse
import numpy as np

from configs import RESULTS_DIR, FIGURES_DIR, FIG_FMT, SEED
from agents.lqr import make_lqr_mpc
from simulations.simulation import run_pool


# True scalar LQR system and cost
A_TRUE, B_TRUE = 1.5, 1.0
Q_COST, R_COST = 1.0, 1.0

# Single mismatch case
A_HAT, B_HAT = 1.2, 0.9

# (H, R) grid; H starts at 2 to keep cubic-P=3 splines well-defined
H_VALUES = list(range(2, 16))
R_VALUES = list(range(1, 11))

# Episode and replication settings
N_STEPS   = 200
N_REPS    = 30
N_SAMPLES = 30
SIGMA     = 0.5

# Noise levels when "on"
MODEL_NOISE = 0.05
ENV_NOISE   = 0.05

# Divergence early-stop. Stable LQR trajectories sit near zero; once |state|
# exceeds DIVERGENCE_THRESHOLD the cell is unrecoverable and remaining steps
# are charged at SATURATION_COST_PER_STEP rather than simulated.
DIVERGENCE_THRESHOLD     = 100.0
SATURATION_COST_PER_STEP = Q_COST * DIVERGENCE_THRESHOLD ** 2

# Mechanism configurations: (label, env_noise_std, model_noise_std)
MECHANISMS = [
    ('no_noise',    0.0,       0.0        ),
    ('model_noise', 0.0,       MODEL_NOISE),
    ('both_noise',  ENV_NOISE, MODEL_NOISE),
]

# Action-search dimensionality configurations: (label, P_setting, interp)
# P_setting=None means "set P = current H" (independent per-step samples)
P_CONFIGS = [
    ('P3cubic', 3,    'cubic'),
    ('PHzero',  None, 'zero' ),
]


def _cell_worker(args):
    """Run one (H, R, seed) episode and return the cost rate.

    Inlined interaction loop with divergence early-stop. The remaining steps
    after a divergence event are charged at SATURATION_COST_PER_STEP so the
    cell appears hot in the heatmap rather than masking out as NaN, while
    avoiding wasted iterations on float-overflow trajectories.
    """
    H, R_int, env_noise, model_noise, P_setting, interp, seed = args

    # P=H means independent per-step samples (random-shooter equivalent)
    P_eff = H if P_setting is None else P_setting

    # Per-worker reproducibility for both noise streams and proposal sampling
    np.random.seed(seed)
    rng = np.random.RandomState(seed + 1)
    initial_state = np.array([rng.uniform(-0.1, 0.1)])

    agent, env = make_lqr_mpc(
        a=A_TRUE, b=B_TRUE, q=Q_COST, r=R_COST,
        a_hat=A_HAT, b_hat=B_HAT,
        horizon=H, recompute_interval=R_int,
        n_samples=N_SAMPLES, P=P_eff, interp=interp,
        sigma=SIGMA, dt=1.0,
        env_noise_std=env_noise, model_noise_std=model_noise,
        initial_state=initial_state, seed=seed,
    )

    total_cost = 0.0
    for i in range(N_STEPS):
        action = agent.interact(env.state, env.cost)
        env.step(action)
        step_cost = float(env.cost)
        diverged = (not np.isfinite(step_cost)
                    or np.abs(env.state).max() > DIVERGENCE_THRESHOLD)
        if diverged:
            steps_remaining = N_STEPS - i
            total_cost += SATURATION_COST_PER_STEP * steps_remaining
            break
        total_cost += step_cost

    return H, R_int, seed, total_cost / N_STEPS


def run_one_sweep(env_noise, model_noise, P_setting, interp, mech_label, p_label):
    """Run a single (H, R) grid sweep at fixed (noise, P) configuration."""
    rng = np.random.RandomState(SEED)
    seeds = rng.randint(0, 2**31, size=N_REPS)

    args_list = [
        (H, R_int, env_noise, model_noise, P_setting, interp, int(seeds[i]))
        for H in H_VALUES
        for R_int in R_VALUES if R_int <= H
        for i in range(N_REPS)
    ]

    print(f"LQR-MPC sweep [{mech_label} | {p_label}]: {len(args_list)} jobs")
    raw = run_pool(_cell_worker, args_list)

    # Aggregate per (H, R) cell
    cells = {}
    for H, R_int, _, cr in raw:
        cells.setdefault((H, R_int), []).append(cr)

    n_H, n_R = len(H_VALUES), len(R_VALUES)
    mean_cost = np.full((n_H, n_R, 1), np.nan)
    sem_cost  = np.full((n_H, n_R, 1), np.nan)
    sweep     = [[None for _ in range(n_R)] for _ in range(n_H)]

    for i, H in enumerate(H_VALUES):
        for j, R_int in enumerate(R_VALUES):
            key = (H, R_int)
            if key not in cells:
                continue
            vals = np.array(cells[key], dtype=float)
            sweep[i][j] = vals.tolist()
            finite = vals[np.isfinite(vals)]
            if finite.size > 0:
                mean_cost[i, j, 0] = finite.mean()
                sem_cost [i, j, 0] = finite.std() / max(np.sqrt(finite.size), 1.0)

    return {
        'env': 'lqr_scalar',
        'horizons':            H_VALUES,
        'recompute_intervals': R_VALUES,
        'mismatch_factors':    [A_HAT],   # single case carried in the canonical slot
        'dt':                  1.0,
        'mean_cost':           mean_cost,
        'sem_cost':            sem_cost,
        'sweep':               sweep,
        'params': {
            'a': A_TRUE, 'b': B_TRUE, 'q': Q_COST, 'r': R_COST,
            'a_hat': A_HAT, 'b_hat': B_HAT,
            'env_noise_std':   env_noise,
            'model_noise_std': model_noise,
            'P_setting':       P_setting,
            'interp':          interp,
            'N':               N_SAMPLES,
            'sigma':           SIGMA,
            'n_steps':         N_STEPS,
            'n_reps':          N_REPS,
            'seed':            SEED,
        },
    }


def _pickle_path(mech_label, p_label):
    return os.path.join(RESULTS_DIR, f'grid_lqr_{mech_label}_{p_label}.pkl')


def run_all():
    """Run the 6-configuration sweep matrix and save each pickle."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for mech_label, env_n, model_n in MECHANISMS:
        for p_label, P_setting, interp in P_CONFIGS:
            result = run_one_sweep(env_n, model_n, P_setting, interp, mech_label, p_label)
            path = _pickle_path(mech_label, p_label)
            with open(path, 'wb') as f:
                pickle.dump(result, f)
            print(f"  saved {path}")


def plot_grid(savefig=True):
    """Render the 3x2 (mechanism x P) heatmap grid from cached pickles."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Load the six pickles
    grids = {}
    for mech_label, _, _ in MECHANISMS:
        for p_label, _, _ in P_CONFIGS:
            path = _pickle_path(mech_label, p_label)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing sweep pickle: {path}. Run run_all() first.")
            with open(path, 'rb') as f:
                grids[(mech_label, p_label)] = pickle.load(f)

    # Shared color range across all six panels (log scale, ignoring inf cells)
    finite_means = []
    for g in grids.values():
        m = g['mean_cost'][..., 0]
        finite_means.append(m[np.isfinite(m)])
    finite_means = np.concatenate(finite_means) if finite_means else np.array([1.0])
    vmin = max(np.nanmin(finite_means), 1e-3)
    vmax = np.nanmax(finite_means)
    norm = LogNorm(vmin=vmin, vmax=max(vmax, 10 * vmin))

    n_rows, n_cols = len(MECHANISMS), len(P_CONFIGS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols + 1.4, 2.4 * n_rows + 0.6),
                             constrained_layout=True)

    H_arr = np.array(H_VALUES)
    R_arr = np.array(R_VALUES)
    extent = [R_arr.min() - 0.5, R_arr.max() + 0.5,
              H_arr.min() - 0.5, H_arr.max() + 0.5]

    im = None
    for i, (mech_label, _, _) in enumerate(MECHANISMS):
        for j, (p_label, _, _) in enumerate(P_CONFIGS):
            ax = axes[i, j]
            mean = grids[(mech_label, p_label)]['mean_cost'][..., 0]

            # Replace inf with vmax for display; LogNorm handles the rest
            disp = np.where(np.isfinite(mean), mean, vmax)

            im = ax.imshow(disp, origin='lower', aspect='auto',
                           extent=extent, cmap='viridis_r', norm=norm)

            if i == 0:
                ax.set_title(p_label)
            if j == 0:
                ax.set_ylabel(f'{mech_label}\nHorizon $H$')
            else:
                ax.tick_params(axis='y', labelleft=False)
            if i == n_rows - 1:
                ax.set_xlabel(r'Replan interval $R$')
            else:
                ax.tick_params(axis='x', labelbottom=False)

    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label='Cost per step')
    fig.suptitle(rf'Scalar LQR-MPC: $\hat{{a}}={A_HAT},\ \hat{{b}}={B_HAT}$',
                 fontsize=11)

    if savefig:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        out_stem = os.path.join(FIGURES_DIR, 'fig_lqr_mpc_mechanisms')
        fig.savefig(out_stem + FIG_FMT, dpi=300)
        fig.savefig(out_stem + '.pdf', dpi=300)
        print(f"  saved {out_stem}{FIG_FMT} and .pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true',
                        help='Render figure from cached pickles instead of running sweeps')
    args = parser.parse_args()

    if args.plot:
        plot_grid()
    else:
        run_all()


if __name__ == '__main__':
    main()
