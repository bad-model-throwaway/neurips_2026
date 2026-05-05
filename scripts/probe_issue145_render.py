"""Figure rendering:
  - Quadratic-cost 1x4 heatmap row for r ∈ {1.0, 1.5, 2.5, 3.0}
  - Side-by-side comparison of tolerance vs quadratic at matched r=1.0

Outputs (exploratory, not manuscript):
  data/plots/fig2A_quadratic_probe.svg
  data/plots/fig2A_tolerance_vs_quadratic_matched.svg
"""
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from visualization.heatmaps import plot_heatmap_row
from configs import RESULTS_DIR, PLOTS_DIR

TOL_PKL  = os.path.join(RESULTS_DIR, 'grid_cartpole.pkl')
QUAD_PKL = os.path.join(RESULTS_DIR, 'grid_cartpole_quadratic_probe.pkl')

os.makedirs(PLOTS_DIR, exist_ok=True)


def _subset(pkl, r_keep):
    """Return a copy of `pkl` with mismatch factors restricted to r_keep."""
    factors = list(pkl['mismatch_factors'])
    idx = [factors.index(r) for r in r_keep]
    out = dict(pkl)
    out['mismatch_factors'] = [factors[i] for i in idx]
    out['mean_cost'] = pkl['mean_cost'][idx]
    out['std_cost']  = pkl['std_cost'][idx]
    return out


def render_quadratic_panel():
    pkl = pickle.load(open(QUAD_PKL, 'rb'))
    sub = _subset(pkl, [1.0, 1.5, 2.5, 3.0])

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
    cbar_label = (f"mean cost / step  (quadratic: "
                  f"Q=diag{tuple(pkl['cost_kwargs']['Q_diag'])}, "
                  f"R={pkl['cost_kwargs']['R_scalar']})")
    plot_heatmap_row(axes, sub, 'CartPole', dt=pkl['dt'], cbar_label=cbar_label)
    out = os.path.join(PLOTS_DIR, 'fig2A_quadratic_probe.svg')
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def render_matched_comparison():
    """2-panel side-by-side: tolerance | quadratic at r=1.0."""
    tol  = pickle.load(open(TOL_PKL,  'rb'))
    quad = pickle.load(open(QUAD_PKL, 'rb'))

    tol_fi  = list(tol['mismatch_factors']).index(1.0)
    quad_fi = list(quad['mismatch_factors']).index(1.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    from matplotlib.colors import LogNorm
    import matplotlib, copy
    cmap = copy.copy(matplotlib.colormaps['viridis'])
    cmap.set_bad(color='#cccccc')

    for ax, pkl, fi, title, cbar_label in [
        (axes[0], tol,  tol_fi,  'Tolerance cost  (r=1.0)',
         'mean cost / step  (tolerance product)'),
        (axes[1], quad, quad_fi, 'Quadratic cost  (r=1.0)',
         f"mean cost / step  (Q=diag{tuple(quad['cost_kwargs']['Q_diag'])}, "
         f"R={quad['cost_kwargs']['R_scalar']})"),
    ]:
        M = np.asarray(pkl['mean_cost'][fi])
        H = np.asarray(pkl['H_values']); R = np.asarray(pkl['R_values'])
        dt = pkl['dt']
        pos = M[np.isfinite(M) & (M > 0)]
        norm = LogNorm(vmin=float(pos.min()), vmax=float(pos.max()))
        mat = np.ma.array(M, mask=~np.isfinite(M))
        im = ax.imshow(mat.T, aspect='auto', cmap=cmap, norm=norm, origin='lower')
        ax.set_xticks(range(len(H)))
        ax.set_xticklabels([f'{h*dt:.2f}' for h in H], rotation=45, ha='right')
        ax.set_yticks(range(len(R)))
        ax.set_yticklabels([f'{r*dt:.2f}' for r in R])
        ax.set_xlabel('H·dt [s]')
        ax.set_ylabel('R·dt [s]')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=cbar_label)

    out = os.path.join(PLOTS_DIR, 'fig2A_tolerance_vs_quadratic_matched.svg')
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == '__main__':
    render_quadratic_panel()
    render_matched_comparison()
