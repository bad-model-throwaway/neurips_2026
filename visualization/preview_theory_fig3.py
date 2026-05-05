"""Render Fig 3 with theory-adaptive replacing ODE-adaptive.

Top row: Σ ℓₖ (lookahead steps per episode) — machine-independent compute
proxy that responds to BOTH knobs the theory adapter moves (R and H).
Bottom row: per-seed physical-success rate (Fig 2 convention).

Loads:
  - data/results/summary_<env>.pkl       (existing 3-condition production;
                                          we use only the 2 Fixed lines)
  - data/results/preview_theory_<env>.pkl (theory-adapter preview, has the
                                           new sweep_rollout_steps field)

Saves: data/plots/preview_theory_fig3.{svg,png}
"""
import os
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import visualization.style  # noqa: F401  (registers manuscript rcParams)
from visualization import heatmaps
from visualization.figures import (
    _FIG3_ENV_STYLES, _PARETO_MARKERS,
    _physical_success_from_last_states,
)
from configs import RESULTS_DIR, PLOTS_DIR
from simulations import sweep_cartpole_adaptive as sca
from simulations import sweep_walker_adaptive as swa
from simulations import sweep_humanoid_balance_adaptive as sha

THEORY_COLOR = '#009E73'
THEORY_MARKER = 'D'
THEORY_LABEL = 'Adaptive (theory)'

ENV_TITLES = {
    'cartpole': 'Cartpole (pole length)',
    'walker': 'Walker (torso mass)',
    'humanoid_balance': 'Humanoid balance (gravity)',
}
ENV_STYLE_KEY = {
    'cartpole': 'cartpole',
    'walker': 'walker',
    'humanoid_balance': 'humanoid',
}
H_INIT = {
    'cartpole': sca.H_CARTPOLE,
    'walker': swa.H_WALKER,
    'humanoid_balance': sha.H_HUMANOID_BALANCE,
}


def _fixed_palette(env_style_key):
    """Drop the third (Adaptive) entry from the summary palette."""
    full = _FIG3_ENV_STYLES[env_style_key]['palette']
    return {k: v for k, v in full.items() if not k.startswith('Adaptive')}


def _rollout_steps_per_episode_summary(sweep, label, mismatch, h_init):
    """Σ ℓₖ for a summary-pkl Fixed condition: H is constant ⇒ n_replans × H."""
    n_replans = np.asarray(sweep['sweep_recomp'][label][mismatch])
    return n_replans * float(h_init) / 1000.0  # convert to ×10³


def _rollout_steps_per_episode_theory(theory_sweep, label, mismatch):
    """Read directly from the new sweep_rollout_steps field (theory varies H)."""
    arr = np.asarray(theory_sweep['sweep_rollout_steps'][label][mismatch])
    return arr.astype(float) / 1000.0


def _draw_compute_panel(ax, sweep, env_name, theory_sweep,
                        show_legend=False, show_xlabel=False, show_ylabel=True):
    style_key = ENV_STYLE_KEY[env_name]
    palette = _fixed_palette(style_key)
    labels = list(palette.keys())
    mism = list(sweep['mismatches'])
    h_init = H_INIT[env_name]

    # Two Fixed lines from existing summary
    for li, lab in enumerate(labels):
        means, ses = [], []
        for r in mism:
            per_seed = _rollout_steps_per_episode_summary(sweep, lab, r, h_init)
            n = len(per_seed)
            means.append(float(per_seed.mean()))
            ses.append(float(per_seed.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
        means = np.asarray(means); ses = np.asarray(ses)
        color = palette[lab]
        marker = _PARETO_MARKERS[li % len(_PARETO_MARKERS)]
        ax.fill_between(mism, np.maximum(means - ses, 0.0), means + ses,
                        color=color, alpha=0.2, linewidth=0)
        ax.plot(mism, means, '-', marker=marker, ms=4.5, lw=1.3,
                color=color, label=lab)

    # Theory line from preview
    if theory_sweep is not None:
        t_lab = next(iter(theory_sweep['sweep_rollout_steps']))
        t_mism = list(theory_sweep['mismatches'])
        t_means, t_ses = [], []
        for r in t_mism:
            per_seed = _rollout_steps_per_episode_theory(theory_sweep, t_lab, r)
            n = len(per_seed)
            t_means.append(float(per_seed.mean()))
            t_ses.append(float(per_seed.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
        t_means = np.asarray(t_means); t_ses = np.asarray(t_ses)
        ax.fill_between(t_mism, np.maximum(t_means - t_ses, 0.0), t_means + t_ses,
                        color=THEORY_COLOR, alpha=0.2, linewidth=0)
        ax.plot(t_mism, t_means, '-', marker=THEORY_MARKER, ms=4.5, lw=1.3,
                color=THEORY_COLOR, label=THEORY_LABEL)

    if show_xlabel:
        ax.set_xlabel('Mismatch')
    if show_ylabel:
        ax.set_ylabel('Lookahead steps / episode (×10³)')
    ax.set_ylim(bottom=0)
    if show_legend:
        ax.legend(loc='upper right', fontsize=6, framealpha=0.9)
    ax.grid(True, alpha=0.2)


def _draw_success_panel(ax, sweep, env_name, theory_sweep,
                        show_legend=False, show_xlabel=True, show_ylabel=True):
    style_key = ENV_STYLE_KEY[env_name]
    palette = _fixed_palette(style_key)
    labels = list(palette.keys())
    mism = list(sweep['mismatches'])
    sweep_ls = sweep.get('last_states') or sweep['sweep_last_states']
    phys_specs = heatmaps.PHYSICAL_SUCCESS[env_name]

    for li, lab in enumerate(labels):
        rates, ses = [], []
        for r in mism:
            success = _physical_success_from_last_states(sweep_ls[lab][r], phys_specs)
            n = len(success)
            p = float(success.mean())
            rates.append(p)
            ses.append(float(np.sqrt(p * (1.0 - p) / n)) if n > 0 else 0.0)
        rates = np.asarray(rates); ses = np.asarray(ses)
        color = palette[lab]
        marker = _PARETO_MARKERS[li % len(_PARETO_MARKERS)]
        ax.fill_between(mism, np.clip(rates - ses, 0, 1), np.clip(rates + ses, 0, 1),
                        color=color, alpha=0.2, linewidth=0)
        ax.plot(mism, rates, '-', marker=marker, ms=4.5, lw=1.3,
                color=color, label=lab)

    if theory_sweep is not None:
        t_lab = next(iter(theory_sweep['sweep_rollout_steps']))
        t_mism = list(theory_sweep['mismatches'])
        t_ls = theory_sweep.get('last_states') or theory_sweep['sweep_last_states']
        t_rates, t_ses = [], []
        for r in t_mism:
            success = _physical_success_from_last_states(t_ls[t_lab][r], phys_specs)
            n = len(success)
            p = float(success.mean())
            t_rates.append(p)
            t_ses.append(float(np.sqrt(p * (1.0 - p) / n)) if n > 0 else 0.0)
        t_rates = np.asarray(t_rates); t_ses = np.asarray(t_ses)
        ax.fill_between(t_mism, np.clip(t_rates - t_ses, 0, 1), np.clip(t_rates + t_ses, 0, 1),
                        color=THEORY_COLOR, alpha=0.2, linewidth=0)
        ax.plot(t_mism, t_rates, '-', marker=THEORY_MARKER, ms=4.5, lw=1.3,
                color=THEORY_COLOR, label=THEORY_LABEL)

    if show_xlabel:
        ax.set_xlabel('Mismatch')
    if show_ylabel:
        ax.set_ylabel('Success rate')
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.9, color='0.6', ls=':', lw=0.7, zorder=0)
    if show_legend:
        ax.legend(loc='lower left', fontsize=6, frameon=True, framealpha=0.9,
                  borderpad=0.4, handletextpad=0.5)
    ax.grid(True, alpha=0.2)


def main():
    envs = ['cartpole', 'walker', 'humanoid_balance']

    panel_w, panel_h = 2.7, 2.1
    fig, axes = plt.subplots(
        2, 3, figsize=(panel_w * 3 + 0.3, panel_h * 2 + 0.4),
        layout='constrained', sharex='col',
    )
    for c in range(1, 3):
        axes[1, c].sharey(axes[1, 0])

    for col, env_name in enumerate(envs):
        with open(os.path.join(RESULTS_DIR, f'summary_{env_name}.pkl'), 'rb') as f:
            sweep = pickle.load(f)

        theory_path = os.path.join(RESULTS_DIR, f'preview_theory_{env_name}.pkl')
        theory_sweep = None
        if os.path.exists(theory_path):
            with open(theory_path, 'rb') as f:
                theory_sweep = pickle.load(f)

        axes[0, col].set_title(ENV_TITLES[env_name], fontsize=10)
        _draw_compute_panel(
            axes[0, col], sweep, env_name, theory_sweep,
            show_legend=False, show_xlabel=False, show_ylabel=(col == 0),
        )
        _draw_success_panel(
            axes[1, col], sweep, env_name, theory_sweep,
            show_legend=(col == 0),
            show_xlabel=True, show_ylabel=(col == 0),
        )
        if col != 0:
            axes[0, col].tick_params(axis='y', labelleft=False)
            axes[1, col].tick_params(axis='y', labelleft=False)
        plt.setp(axes[0, col].get_xticklabels(), visible=False)

    fig.suptitle(
        'Figure 3 preview - TheoryStepAdaptation replaces ODE (N=30, lookahead-steps axis)',
        fontsize=10,
    )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_svg = os.path.join(PLOTS_DIR, 'preview_theory_fig3.svg')
    out_png = os.path.join(PLOTS_DIR, 'preview_theory_fig3.png')
    fig.savefig(out_svg, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_svg}')
    print(f'wrote {out_png}')


if __name__ == '__main__':
    main()
