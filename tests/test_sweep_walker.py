"""Smoke test for walker adaptive sweep with TheoryStepAdaptation.

Pipeline sanity only — module imports, MJPC walker env, MPPI planner,
TheoryStepAdaptation hookup, result aggregation. Does NOT validate the
Figure 3 scientific claims.

n_episodes=2 × 2 mismatches × 3 conditions = 12 episodes (1000 steps each).
On 12 workers expect ~60-120 s.
"""
import numpy as np

from simulations import sweep_walker_adaptive as swa


def test_walker_adaptive_theory_smoke():
    """TheoryStepAdaptation: matched + sustained mismatch, schema check."""
    sweep = swa.run_adaptive_sweep(
        n_episodes=2, mismatches=[1.0, 1.5], mismatch_a=1.5,
        adapt_class='TheoryStepAdaptation',
    )

    for key in ('mismatches', 'mismatch_a', 'sweep_len', 'sweep_cost',
                'sweep_failure', 'sweep_recomp',
                'sweep_rh_traces', 'sweep_cum_traces'):
        assert key in sweep, f"missing key {key!r}"

    labels = [c['label'] for c in swa.CONDITIONS]
    for lab in labels:
        for m in (1.0, 1.5):
            costs = sweep['sweep_cost'][lab][m]
            assert len(costs) == 2, \
                f"{lab} m={m}: expected 2 episodes, got {len(costs)}"
            assert all(np.isfinite(c) and c > 0 for c in costs), \
                f"{lab} m={m}: non-finite/non-positive costs: {costs}"

            failures = sweep['sweep_failure'][lab][m]
            assert all(0.0 < f <= swa.N_STEPS * swa.DT + 1e-6 for f in failures), \
                f"{lab} m={m}: failure_sec out of range: {failures}"
