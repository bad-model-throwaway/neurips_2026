"""Cost-vs-mismatch summary sweep — cartpole Panel A for Figure 3.

Three controllers over a dense mismatch grid, all at the same H=53 planning
horizon:

    Fixed (R=1)           — performance ceiling (always replan)
    Fixed (R=4)           — no-adaptation counterfactual at adaptive's initial R
    Adaptive (R=4 init)   — paper's proposed method

Comparing Fixed (R=4) against Adaptive (R=4 init) isolates the effect of
adaptation (same starting compute budget, same horizon). Fixed (R=1) provides
the ceiling context.

Time-resolved traces (recompute and running-cost) are saved at MISMATCH_A = 1.3,
which the v2 probe identified as the differentiated regime.

Run from worktree root:

    python -m simulations.sweep_cartpole_summary
    python -m simulations.sweep_cartpole_summary --plot-only   # skip re-sweep
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import visualization.style  # noqa: F401

import simulations.sweep_cartpole_adaptive as sca
from configs import DT, PLOTS_DIR, RESULTS_DIR


CONDITIONS = [
    dict(recompute=3, adaptive=False, label='Fixed (R·dt=0.06s)'),
    dict(recompute=3, adaptive=True,  label='Adaptive'),
]

MISMATCHES = [1.0, 1.5, 2.5, 3.0]
N_EPISODES = 30
MISMATCH_A = 1.3

PKL_NAME = 'summary_cartpole.pkl'
PNG_NAME = 'summary_cartpole.png'
PDF_NAME = 'summary_cartpole.pdf'

# Shared palette across all Figure 3 rows: red = ceiling, orange = no-adapt, blue = adaptive.
COND_COLORS = {
    'Fixed (R·dt=0.06s)': '#E69F00',   # orange    — no-adapt at white-star init
    'Adaptive':           '#0072B2',   # blue      — TheoryStepAdaptation (joint R+H)
}


def run_sweep():
    sca.CONDITIONS = CONDITIONS
    return sca.run_adaptive_sweep(
        n_episodes=N_EPISODES,
        mismatches=MISMATCHES,
        mismatch_a=MISMATCH_A,
        adapt_class='TheoryStepAdaptation',
    )


def _aggregate(sweep):
    duration_sec = sca.N_STEPS * DT
    labels = [c['label'] for c in CONDITIONS]
    mism = list(sweep['mismatches'])
    means = {lab: [] for lab in labels}
    stderrs = {lab: [] for lab in labels}
    fail_rate = {lab: [] for lab in labels}
    for r in mism:
        for lab in labels:
            costs = np.asarray(sweep['sweep_cost'][lab][r]) / duration_sec
            fails = np.asarray(sweep['sweep_failure'][lab][r])
            means[lab].append(float(costs.mean()))
            stderrs[lab].append(
                float(costs.std(ddof=1) / np.sqrt(len(costs))) if len(costs) > 1 else 0.0
            )
            fail_rate[lab].append(float((fails < duration_sec - 1e-6).mean()))
    return mism, labels, means, stderrs, fail_rate, duration_sec


def plot_panel(sweep, save=True):
    mism, labels, means, stderrs, fail_rate, duration_sec = _aggregate(sweep)

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for lab in labels:
        m = np.asarray(means[lab])
        s = np.asarray(stderrs[lab])
        color = COND_COLORS[lab]
        ax.plot(mism, m, marker='o', lw=1.5, color=color, label=lab)
        ax.fill_between(mism, np.maximum(m - s, 1e-4), m + s,
                        color=color, alpha=0.2, linewidth=0)

    ax.set_yscale('log')
    ax.set_xlabel('Mismatch ratio $r$ (pole length)')
    ax.set_ylabel('Cost per second')
    ax.set_title('Cartpole: adaptive vs. frozen vs. fixed baselines')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, which='both', alpha=0.25)

    fig.tight_layout()

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        for name in (PNG_NAME, PDF_NAME):
            path = os.path.join(PLOTS_DIR, name)
            fig.savefig(path)
            print(f"Saved: {path}")
    return fig, ax


def print_summary(sweep):
    mism, labels, means, stderrs, fail_rate, duration_sec = _aggregate(sweep)

    print("\nCost per second — mean ± stderr (n={}):".format(N_EPISODES))
    header = f"{'r':>6} | " + " | ".join(f"{lab:^24}" for lab in labels)
    print(header)
    print('-' * len(header))
    for i, r in enumerate(mism):
        cells = [f"{means[lab][i]:>10.3f} ± {stderrs[lab][i]:>5.3f}  " for lab in labels]
        print(f"{r:>6.2f} | " + " | ".join(cells))

    print("\nFailure rate (fraction of episodes that fell before 20s):")
    print(f"{'r':>6} | " + " | ".join(f"{lab:^24}" for lab in labels))
    for i, r in enumerate(mism):
        cells = [f"{fail_rate[lab][i]:>10.2f}{'':>13}" for lab in labels]
        print(f"{r:>6.2f} | " + " | ".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip sweep; load existing pkl and re-render figure.')
    args = parser.parse_args()

    path = os.path.join(RESULTS_DIR, PKL_NAME)
    if args.plot_only:
        with open(path, 'rb') as f:
            sweep = pickle.load(f)
        print(f"Loaded: {path}")
    else:
        sweep = run_sweep()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(sweep, f)
        print(f"Saved: {path}")

    print_summary(sweep)
    plot_panel(sweep)


if __name__ == '__main__':
    main()
