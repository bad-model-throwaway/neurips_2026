"""TheoryStepAdaptation preview at the existing Fig 3 mismatch grid.

One condition only (theory-adaptive). N_EPISODES=30. Outputs pkls in the
same schema as sweep_*_adaptive.run_adaptive_sweep so the existing Fig 3
plotting machinery can consume them.

Usage:
    N_WORKERS=14 python -m simulations.preview_theory_sweep
"""
import os
import pickle

import simulations.sweep_cartpole_adaptive as sca
import simulations.sweep_walker_adaptive as swa
import simulations.sweep_humanoid_balance_adaptive as sha
from simulations import sweep_cartpole_summary as scs
from simulations import sweep_walker_summary as sws
from simulations import sweep_humanoid_balance_summary as shbs
from configs import RESULTS_DIR

N_EPISODES = 30
THEORY_LABEL = 'Adaptive (theory)'


def _theory_cond(adaptive_recompute):
    return dict(recompute=adaptive_recompute, adaptive=True, label=THEORY_LABEL)


def run_cartpole():
    """Cartpole adaptive uses module-level CONDITIONS; monkey-patch like summary does."""
    orig = sca.CONDITIONS
    sca.CONDITIONS = [_theory_cond(scs.CONDITIONS[2]['recompute'])]
    try:
        sweep = sca.run_adaptive_sweep(
            n_episodes=N_EPISODES,
            mismatches=scs.MISMATCHES,
            mismatch_a=scs.MISMATCH_A,
            adapt_class='TheoryStepAdaptation',
        )
    finally:
        sca.CONDITIONS = orig
    return sweep


def run_walker():
    return swa.run_adaptive_sweep(
        n_episodes=N_EPISODES,
        mismatches=sws.MISMATCHES,
        mismatch_a=sws.MISMATCH_A,
        conditions=[_theory_cond(sws.CONDITIONS[2]['recompute'])],
        adapt_class='TheoryStepAdaptation',
    )


def run_humanoid():
    return sha.run_adaptive_sweep(
        n_episodes=N_EPISODES,
        mismatches=shbs.MISMATCHES,
        mismatch_a=shbs.MISMATCH_A,
        conditions=[_theory_cond(shbs.CONDITIONS[2]['recompute'])],
        adapt_class='TheoryStepAdaptation',
    )


def main():
    runners = [
        ('cartpole', run_cartpole),
        ('walker', run_walker),
        ('humanoid_balance', run_humanoid),
    ]
    for env_name, run in runners:
        print(f'\n=== preview_theory_sweep: {env_name} (N_EPISODES={N_EPISODES}) ===',
              flush=True)
        sweep = run()
        out_path = os.path.join(RESULTS_DIR, f'preview_theory_{env_name}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(sweep, f)
        print(f'wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
