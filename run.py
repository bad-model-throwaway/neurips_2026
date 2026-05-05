"""Run simulations and generate manuscript figures."""

import os
import glob
import json
import pickle
import shutil

import numpy as np

from configs import *

from simulations.sweep_cartpole_adaptive import run_adaptive_sweep
from simulations import sweep_cartpole_summary, sweep_cartpole_perturbation
from simulations import sweep_walker_summary, sweep_walker_perturbation
from simulations import sweep_humanoid_balance_summary
from simulations import sweep_cartpole_midswitch, sweep_walker_midswitch
from simulations import sweep_humanoid_balance_midswitch
from simulations.sweep_grid import run_grid_sweep, DEFAULT_GRIDS, SMOKE_GRIDS
from simulations import diagnostics
from visualization import figures as figs, supplement as sups, compile as comp


def _to_serializable(obj):
    """Recursively convert obj to a JSON-serializable form."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return repr(obj)


def cache(filename, func, **kwargs):
    """Load cached results from RESULTS_DIR, or run func and save.

    Writes a .meta.json sidecar alongside each pickle recording the kwargs
    and sweep-affecting globals (SEED, N_WORKERS). On subsequent loads,
    warns if the current parameters differ from what produced the cache.
    """
    # Check if cache exists
    path = os.path.join(RESULTS_DIR, filename)
    meta_path = path + '.meta.json'
    current_meta = {
        'kwargs': _to_serializable(kwargs),
        'SEED': SEED,
        'N_WORKERS': N_WORKERS,
    }

    if os.path.exists(path):
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                saved_meta = json.load(f)
            if saved_meta != current_meta:
                print(
                    f"WARNING: cached '{filename}' was generated with different "
                    f"parameters. Delete it or run clean(results=True) to regenerate."
                )
        else:
            print(
                f"WARNING: cached '{filename}' has no metadata sidecar; "
                f"cannot verify freshness. Run clean(results=True) to regenerate."
            )
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        # Run the function to get data
        data = func(**kwargs)

        # Then save the data
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        with open(meta_path, 'w') as f:
            json.dump(current_meta, f, indent=2)

    # Yield results
    return data


def clean(results=False, figures=False, plots=False):
    """Remove generated data from output directories.

    results: clear simulations/results/ (pickle files)
    figures: clear data/figures/ (manuscript SVGs)
    plots: clear data/plots/ (test/exploratory SVGs)
    """
    dirs = []
    if results:
        dirs.append(RESULTS_DIR)
    if figures:
        dirs.append(FIGURES_DIR)
    if plots:
        dirs.append(PLOTS_DIR)

    for d in dirs:
        if not os.path.isdir(d):
            continue
        files = glob.glob(os.path.join(d, '*'))
        for f in files:
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)
        print(f"Cleaned {d} ({len(files)} items)")


def clean_metagrid():
    """Remove metagrid sweep results (grid pkl caches and their sidecars)."""
    removed = 0
    for pattern in ('grid_*.pkl', 'grid_*.pkl.meta.json'):
        for f in glob.glob(os.path.join(RESULTS_DIR, pattern)):
            os.remove(f)
            removed += 1
    print(f"Cleaned {removed} metagrid files from {RESULTS_DIR}")


def main(
    diagnostics_cartpole=True,
    metagrid_cartpole=True, metagrid_walker=True, metagrid_humanoid_balance=True,
    adaptive_cartpole=False, adaptive_walker=False,
    adaptive_humanoid_balance=False,
    midswitch_cartpole=False, midswitch_walker=False,
    midswitch_humanoid_balance=False,
):
    smoke = bool(os.environ.get('SMOKE'))
    grids = SMOKE_GRIDS if smoke else DEFAULT_GRIDS

    if diagnostics_cartpole:
        diagnostics.run_control_quality_probe('cartpole')
        diagnostics.run_timing_model_probe('cartpole')
        diagnostics.run_mismatch_sensitivity_probe('cartpole')

    if metagrid_cartpole:
        g = grids['cartpole']
        cartpole_grid = cache(
            'grid_cartpole.pkl', run_grid_sweep,
            env_name='cartpole', H_values=g['H'], R_values=g['R'],
            mismatch_factors=g['mismatch'], n_reps=g['reps'],
            n_steps=g.get('n_steps', 1000),
            proposal=g.get('proposal'), N=g.get('N'),
            proposal_kwargs=g.get('proposal_kwargs'),
            decision=g.get('decision'),
        )
    else:
        cartpole_grid = None

    if metagrid_walker:
        g = grids['walker']
        walker_grid = cache(
            'grid_walker.pkl', run_grid_sweep,
            env_name='walker', H_values=g['H'], R_values=g['R'],
            mismatch_factors=g['mismatch'], n_reps=g['reps'],
            n_steps=g.get('n_steps', 1000),
        )
    else:
        walker_grid = None

    if metagrid_humanoid_balance:
        g = grids['humanoid_balance']
        humanoid_balance_grid = cache(
            'grid_humanoid_balance.pkl', run_grid_sweep,
            env_name='humanoid_balance', H_values=g['H'], R_values=g['R'],
            mismatch_factors=g['mismatch'], n_reps=g['reps'],
            n_steps=g.get('n_steps', 1000),
        )
    else:
        humanoid_balance_grid = None

    if metagrid_cartpole or metagrid_walker or metagrid_humanoid_balance:
        figs.figure_2(cartpole_grid, walker_grid, humanoid_balance_grid)
        comp.compile_figure_2()

    cartpole_summary = None
    cartpole_perturbation = None
    if adaptive_cartpole:
        cartpole_summary = cache(
            'summary_cartpole.pkl', sweep_cartpole_summary.run_sweep,
        )
        cartpole_perturbation = cache(
            'perturbation_cartpole.pkl',
            sweep_cartpole_perturbation.run_perturbation_sweep,
        )
        cartpole_adaptive = cache('adaptive_sweep.pkl', run_adaptive_sweep)
        sups.supplement_3(cartpole_adaptive)
        comp.compile_supplement_3()

    walker_summary = None
    walker_perturbation = None
    if adaptive_walker:
        walker_summary = cache(
            'summary_walker.pkl', sweep_walker_summary.run_sweep,
        )
        walker_perturbation = cache(
            'perturbation_walker.pkl',
            sweep_walker_perturbation.run_perturbation_sweep,
        )

    humanoid_balance_summary = None
    if adaptive_humanoid_balance:
        humanoid_balance_summary = cache(
            'summary_humanoid_balance.pkl',
            sweep_humanoid_balance_summary.run_sweep,
        )

    cartpole_midswitch_data = None
    if midswitch_cartpole:
        cartpole_midswitch_data = cache(
            'midswitch_cartpole.pkl',
            sweep_cartpole_midswitch.run_midswitch_sweep,
        )

    walker_midswitch_data = None
    if midswitch_walker:
        walker_midswitch_data = cache(
            'midswitch_walker.pkl',
            sweep_walker_midswitch.run_midswitch_sweep,
        )

    humanoid_midswitch_data = None
    if midswitch_humanoid_balance:
        humanoid_midswitch_data = cache(
            'midswitch_humanoid_balance.pkl',
            sweep_humanoid_balance_midswitch.run_midswitch_sweep,
        )

    # Figure 3 compose: fires when all available rows are on disk.
    _r = lambda f: os.path.join(RESULTS_DIR, f)
    if cartpole_summary is None and os.path.exists(_r('summary_cartpole.pkl')):
        cartpole_summary = cache('summary_cartpole.pkl', sweep_cartpole_summary.run_sweep)
    if walker_summary is None and os.path.exists(_r('summary_walker.pkl')):
        walker_summary = cache('summary_walker.pkl', sweep_walker_summary.run_sweep)
    if humanoid_balance_summary is None and os.path.exists(_r('summary_humanoid_balance.pkl')):
        humanoid_balance_summary = cache(
            'summary_humanoid_balance.pkl',
            sweep_humanoid_balance_summary.run_sweep,
        )
    if cartpole_midswitch_data is None and os.path.exists(_r('midswitch_cartpole.pkl')):
        cartpole_midswitch_data = cache(
            'midswitch_cartpole.pkl', sweep_cartpole_midswitch.run_midswitch_sweep,
        )
    if walker_midswitch_data is None and os.path.exists(_r('midswitch_walker.pkl')):
        walker_midswitch_data = cache(
            'midswitch_walker.pkl', sweep_walker_midswitch.run_midswitch_sweep,
        )
    if humanoid_midswitch_data is None and os.path.exists(_r('midswitch_humanoid_balance.pkl')):
        humanoid_midswitch_data = cache(
            'midswitch_humanoid_balance.pkl',
            sweep_humanoid_balance_midswitch.run_midswitch_sweep,
        )

    if any(x is not None for x in (
        cartpole_summary, walker_summary, humanoid_balance_summary,
        cartpole_midswitch_data, walker_midswitch_data, humanoid_midswitch_data,
    )):
        figs.figure_3(
            cartpole_summary=cartpole_summary,
            walker_summary=walker_summary,
            humanoid_summary=humanoid_balance_summary,
            cartpole_midswitch=cartpole_midswitch_data,
            walker_midswitch=walker_midswitch_data,
            humanoid_midswitch=humanoid_midswitch_data,
        )
        comp.compile_figure_3()

    if cartpole_perturbation is None and os.path.exists(_r('perturbation_cartpole.pkl')):
        cartpole_perturbation = cache('perturbation_cartpole.pkl',
                                      sweep_cartpole_perturbation.run_perturbation_sweep)
    if walker_perturbation is None and os.path.exists(_r('perturbation_walker.pkl')):
        walker_perturbation = cache('perturbation_walker.pkl',
                                    sweep_walker_perturbation.run_perturbation_sweep)
    if cartpole_perturbation is not None and walker_perturbation is not None:
        sups.supplement_4(cartpole_perturbation, walker_perturbation)
        comp.compile_supplement_4()


if __name__ == '__main__':
    main()
