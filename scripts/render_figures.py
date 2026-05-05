"""Lightweight figure renderer — loads cached pkls and calls figure functions.

Does NOT run any sweeps, diagnostics, or multiprocessing.
Safe to run on a login node.

Usage:
    python render_figures.py          # render all available figures
    python render_figures.py --fig 2  # render only Figure 2
    python render_figures.py --fig 3  # render only Figure 3 (3x3 stack)
"""
import argparse
import os
import pickle

from configs import RESULTS_DIR
from visualization import figures as figs
from visualization import compile as comp
from visualization import supplement_fig2 as supp_fig2
from visualization import supplement_robustness as supp_rob


def _load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def render_figure_2():
    cartpole_grid         = _load('grid_cartpole.pkl')
    walker_grid           = _load('grid_walker.pkl')
    humanoid_balance_grid = _load('grid_humanoid_balance.pkl')
    if any(x is not None for x in (cartpole_grid, walker_grid, humanoid_balance_grid)):
        figs.figure_2(cartpole_grid, walker_grid, humanoid_balance_grid)
        comp.compile_figure_2()
        supp_fig2.render_all({
            'cartpole':         cartpole_grid,
            'walker':           walker_grid,
            'humanoid_balance': humanoid_balance_grid,
        })
        comp.compile_supplement_fig2()
        print('Figure 2 + supps rendered.')
    else:
        print('Figure 2: no grid pkls found, skipped.')


def render_figure_3():
    cartpole_summary   = _load('summary_cartpole.pkl')
    walker_summary     = _load('summary_walker.pkl')
    humanoid_summary   = _load('summary_humanoid_balance.pkl')
    cartpole_midswitch = _load('midswitch_cartpole.pkl')
    walker_midswitch   = _load('midswitch_walker.pkl')
    humanoid_midswitch = _load('midswitch_humanoid_balance.pkl')
    inputs = (cartpole_summary, walker_summary, humanoid_summary,
              cartpole_midswitch, walker_midswitch, humanoid_midswitch)
    if any(x is not None for x in inputs):
        figs.figure_3(
            cartpole_summary=cartpole_summary,
            walker_summary=walker_summary,
            humanoid_summary=humanoid_summary,
            cartpole_midswitch=cartpole_midswitch,
            walker_midswitch=walker_midswitch,
            humanoid_midswitch=humanoid_midswitch,
        )
        comp.compile_figure_3()
        print('Figure 3 rendered.')
        if any(x is not None for x in (cartpole_summary, walker_summary, humanoid_summary)):
            figs.print_fig3_table(cartpole_summary, walker_summary, humanoid_summary)
            figs.print_fig3_stats(cartpole_summary, walker_summary, humanoid_summary)
    else:
        print('Figure 3: no summary or midswitch pkls found, skipped.')


def render_supp_fig3():
    cartpole_summary = _load('summary_cartpole.pkl')
    walker_summary   = _load('summary_walker.pkl')
    humanoid_summary = _load('summary_humanoid_balance.pkl')
    if any(x is not None for x in (cartpole_summary, walker_summary, humanoid_summary)):
        figs.figure_supp3(cartpole_summary, walker_summary, humanoid_summary)
        comp.compile_supplement_fig3()
        print('Figure 3 supplement rendered.')
    else:
        print('Figure 3 supplement: no summary pkls found, skipped.')


def render_supp_robustness():
    supp_rob.figure_supp_rob_grid()
    supp_rob.figure_supp_rob_summary()
    comp.compile_supplement_rob()
    print('Supplement robustness figures rendered.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fig', type=str, choices=['2', '3', 'rob'],
                        help='Render only this figure (default: all)')
    args = parser.parse_args()

    if args.fig == '2':
        render_figure_2()
    elif args.fig == '3':
        render_figure_3()
        render_supp_fig3()
    elif args.fig == 'rob':
        render_supp_robustness()
    else:
        render_figure_2()
        render_figure_3()
        render_supp_fig3()
        render_supp_robustness()


if __name__ == '__main__':
    main()
