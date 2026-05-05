"""Cost-vs-mismatch summary sweep — walker Panels E/F for Figure 3.

Three controllers over a dense torso-mass mismatch grid at the MJPC-aligned
walker horizon H=82 (on the grid_walker.pkl H-axis; nearest to the matched U-bottom):

    Fixed (R=1)            — performance ceiling (always replan)
    Fixed (R=8)            — no-adaptation counterfactual at adaptive's initial R
    Adaptive (R=8 init)    — paper's proposed method

Comparing Fixed (R=8) against Adaptive (R=8 init) isolates the effect of
adaptation (same starting compute budget, same horizon); Fixed (R=1) gives
the ceiling. Mirrors the structure of `sweep_cartpole_summary`.

R_init rationale — picked from `data/results/grid_walker.pkl` (n_reps=3, H=82):

    At H=82 on `grid_walker.pkl`, R=8 is the largest R where matched-dynamics
    cost stays near the R=1 ceiling (0.32 vs 0.23, ~40% gap) while the matched
    run doesn't show any sign of the instability cliff that opens at R=10
    (2.05 cost at r=1.6 signals collapse). Under mismatch, R=8 shows a clean
    monotone degradation (0.32 → 0.57 from r=1.0 to r=2.0, a 78% gap vs the
    R=1 ceiling's 62% gap) — adequate headroom for adaptation to contract τ
    and close the gap. R=4 is too close to the ceiling (adaptation would have
    nothing to fix); R=10 is past the cliff (frozen baseline collapses,
    making the adaptive-vs-frozen contrast trivially one-sided rather than a
    clean "adaptation closes a quantifiable gap" story).

The grid is n_reps=3 so there is sampling noise (R=6's 0.68 at r=1.6 is
within 1σ of R=8's 0.49); if the 7-seed sweep here reveals R=8 is not
well-separated from R=1, fall back to R=10.

`run_sweep()` returns the sweep dict without saving; `main()` owns the
pickle (cartpole post-#55 convention). `run.py::cache()` drives pkl creation.

Run from worktree root:

    python -m simulations.sweep_walker_summary
    python -m simulations.sweep_walker_summary --plot-only   # skip re-sweep
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import visualization.style  # noqa: F401

import simulations.sweep_walker_adaptive as swa
from configs import PLOTS_DIR, RESULTS_DIR
from simulations.sweep_walker_adaptive import DT


R_INIT = 7  # Fig 2 white-star recompute; both Fixed and Adaptive initialize here

CONDITIONS = [
    dict(recompute=R_INIT, adaptive=False, label='Fixed (R·dt=0.07s)'),
    dict(recompute=R_INIT, adaptive=True,  label='Adaptive'),
]

MISMATCHES = [1.0, 1.6, 2.0, 2.6]
N_EPISODES = 30
MISMATCH_A = 1.5

PKL_NAME = 'summary_walker.pkl'
PNG_NAME = 'summary_walker.png'
PDF_NAME = 'summary_walker.pdf'

# Shared palette across all Figure 3 rows: red = ceiling, orange = no-adapt, blue = adaptive.
COND_COLORS = {
    'Fixed (R·dt=0.07s)': '#E69F00',   # orange    — no-adapt at white-star init
    'Adaptive':           '#0072B2',   # blue      — TheoryStepAdaptation (joint R+H)
}


def run_sweep():
    """Dispatch through the walker adaptive kernel. No side-effect save."""
    return swa.run_adaptive_sweep(
        n_episodes=N_EPISODES,
        mismatches=MISMATCHES,
        mismatch_a=MISMATCH_A,
        conditions=CONDITIONS,
        adapt_class='TheoryStepAdaptation',
    )


def _aggregate(sweep):
    duration_sec = swa.N_STEPS * DT
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
    ax.set_xlabel('Mismatch ratio $r$ (torso mass)')
    ax.set_ylabel('Cost per second')
    ax.set_title('Walker: adaptive vs. frozen vs. fixed baselines')
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
                "produce it, or wait until Stage 2 of the walker plan."
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
