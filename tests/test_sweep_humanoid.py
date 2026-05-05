"""Smoke test for humanoid_balance adaptive sweep with TheoryStepAdaptation.

Pipeline sanity only — module imports, MJPC humanoid env, planner,
TheoryStepAdaptation hookup, result aggregation. Does NOT validate the
Figure 3 scientific claims.

n_episodes=2 × 2 mismatches × 3 conditions = 12 episodes (300 steps each).
On 12 workers expect ~30-90 s.
"""
import numpy as np

from simulations import sweep_humanoid_balance_adaptive as sha


def test_humanoid_balance_adaptive_theory_smoke():
    """TheoryStepAdaptation: matched + sustained mismatch, schema check."""
    sweep = sha.run_adaptive_sweep(
        n_episodes=2, mismatches=[1.0, 1.4], mismatch_a=1.4,
        adapt_class='TheoryStepAdaptation',
    )

    for key in ('mismatches', 'mismatch_a', 'sweep_len', 'sweep_cost',
                'sweep_failure', 'sweep_recomp',
                'sweep_rh_traces', 'sweep_cum_traces'):
        assert key in sweep, f"missing key {key!r}"

    labels = [c['label'] for c in sha.CONDITIONS]
    for lab in labels:
        for m in (1.0, 1.4):
            costs = sweep['sweep_cost'][lab][m]
            assert len(costs) == 2, \
                f"{lab} m={m}: expected 2 episodes, got {len(costs)}"
            assert all(np.isfinite(c) and c > 0 for c in costs), \
                f"{lab} m={m}: non-finite/non-positive costs: {costs}"

            failures = sweep['sweep_failure'][lab][m]
            assert all(0.0 <= f <= sha.N_STEPS * sha.DT + 1e-6 for f in failures), \
                f"{lab} m={m}: failure_sec out of range: {failures}"
