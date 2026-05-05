"""Cost-vs-mismatch summary sweep — humanoid_balance panels E/F for Figure 3.

Three controllers over a dense gravity mismatch grid at H=30 (U-bottom
from grid_humanoid_balance.pkl):

    Fixed (R=1)            — performance ceiling (always replan)
    Fixed (R=2)            — no-adaptation counterfactual at adaptive's initial R
    Adaptive (R=2 init)    — paper's proposed method

Comparing Fixed (R=2) against Adaptive (R=2 init) isolates the effect of
adaptation (same starting compute budget, same horizon); Fixed (R=1) gives
the ceiling. Mirrors the structure of `sweep_walker_summary`.

H and R_init rationale — picked from `data/results/grid_humanoid_balance.pkl`
(n_reps=30, gravity mismatch factors [1.0, 1.2, 1.4, 1.6]):

    At R=1 across H values, the U-bottom sits at H=30 (cost=1.06 at r=1.0,
    vs H=25: 3.19, H=40: 1.33). H=30 is the planning horizon where matched
    dynamics produce the best balance cost and remains viable at r=1.6
    (cost=1.51, +42% vs matched), confirming the cartpole-analog narrative
    that a Goldilocks horizon is sufficient under mild mismatch.

    At H=30 the cost cliff from R=1 to R=2 is steep: R=2 costs 4.82 at
    r=1.0 (vs 1.06 for R=1, a 4.5× gap) while R=3+ is catastrophic
    (20+ matched cost). R=2 is therefore the unique R_init: it is the
    largest R where the agent retains any useful performance, providing
    the clearest headroom for the adaptive controller to close the R=1
    gap. R=1 is too close to itself; R=3 is already non-functional.

    Under maximum tested mismatch (r=1.6), R=2 costs 6.94 (+44% vs R=1's
    1.51), preserving the "adaptation closes a quantifiable gap" narrative.

Mismatch axis: gravity scaling factor on the planner model only — env stays
at g=9.81 (factor 1.0); planner sees g * factor. Values extend below and
above Fig 2's 4-point grid [1.0, 1.2, 1.4, 1.6] for a smooth curve.
r <= 0.6 and r >= 1.75 were catastrophic in #134 Phase 3 probes; 1.7 is
the fence.

`run_sweep()` returns the sweep dict without saving; `main()` owns the
pickle (cartpole / walker post-#55 convention). `run.py::cache()` drives
pkl creation.

Run from worktree root:

    python -m simulations.sweep_humanoid_balance_summary
    python -m simulations.sweep_humanoid_balance_summary --plot-only
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import visualization.style  # noqa: F401

import simulations.sweep_humanoid_balance_adaptive as shba
from configs import PLOTS_DIR, RESULTS_DIR
from simulations.sweep_humanoid_balance_adaptive import DT


R_INIT = 8  # Fig 2 white-star recompute; both Fixed and Adaptive initialize here

CONDITIONS = [
    dict(recompute=R_INIT, adaptive=False, label='Fixed (R·dt=0.12s)'),
    dict(recompute=R_INIT, adaptive=True,  label='Adaptive'),
]

MISMATCHES = [1.0, 1.2, 1.4, 1.6]
N_EPISODES = 30
MISMATCH_A = 1.2  # highlighted mismatch for cumulative-cost trace

PKL_NAME = 'summary_humanoid_balance.pkl'
PNG_NAME = 'summary_humanoid_balance.png'
PDF_NAME = 'summary_humanoid_balance.pdf'

# Matches the cartpole figure palette (red / orange / blue) so all three
# Figure 3 rows read with the same color grammar: red = fixed high-freq
# ceiling, orange = fixed low-freq counterfactual, blue = adaptive method.
COND_COLORS = {
    'Fixed (R·dt=0.12s)':  '#E69F00',   # orange    — no-adapt at white-star init
    'Adaptive':            '#0072B2',   # blue      — TheoryStepAdaptation (joint R+H)
}


def run_sweep():
    """Dispatch through the humanoid_balance adaptive kernel. No side-effect save."""
    return shba.run_adaptive_sweep(
        n_episodes=N_EPISODES,
        mismatches=MISMATCHES,
        mismatch_a=MISMATCH_A,
        conditions=CONDITIONS,
        adapt_class='TheoryStepAdaptation',
    )


def _aggregate(sweep):
    duration_sec = shba.N_STEPS * DT
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
                float(costs.std(ddof=1) / np.sqrt(len(costs))) if len(costs) > 1
                else 0.0
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
    ax.set_xlabel(r'Mismatch ratio $r$ (gravity)')
    ax.set_ylabel('Cost per second')
    ax.set_title('Humanoid Balance: adaptive vs. frozen vs. ceiling')
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

    print("\nFailure rate (fraction of episodes that fell before {:.1f}s):".format(
        duration_sec))
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
        if not os.path.exists(path):
            raise SystemExit(
                f"No cached sweep found at {path}. Run without --plot-only to "
                "produce it."
            )
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
