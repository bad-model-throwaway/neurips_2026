"""Walker adapter relaxation rate tuning probe.

Runs short adaptive-walker episodes across {r=1.0, r=1.5} with 5 seeds per
mismatch and reports tau (= recompute_interval) statistics at matched and
mismatched dynamics plus an optional r=1.0 impulse arm to check post-impulse
contraction.

Targets:
  r=1.0, 10 s: mean tau_bar across episode <= 9.0.
  r=1.5, 10 s: mean tau_bar <= 4.0 (must not regress).
  Post-impulse at r=1.0 with 300 N impulse at t=5 s: adaptive tau should
    dip to <= 7 for at least 0.5 s after the impulse.

Usage:
    python -m scripts.probe_issue_106_adapter
    python -m scripts.probe_issue_106_adapter --quick    # 3 seeds, skip impulse
    python -m scripts.probe_issue_106_adapter --impulse  # include impulse arm
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs import SEED
from simulations.simulation import run_pool


# Mirror the sweep config so the probe reflects actual production wiring.
H_WALKER = 80
N_STEPS = 1000            # 10 s at walker DT=0.01
DT = 0.01
R_INIT = 8
MIN_ERROR_THRESHOLD = 0.15
RELAX_STEP = float(os.environ.get('PROBE_RELAX_STEP', '0.05'))
# Slower than the 0.2 default to keep matched-dynamics tau_bar near tau_init on
# a 10 s horizon without damaging contraction under mismatch. Override via
# PROBE_RELAX_STEP=... to sweep during tuning.

SAMPLE_TIMES_SEC = (2.0, 5.0, 8.0, 10.0)

# Impulse config matches sweep_walker_perturbation defaults.
IMPULSE_N = 300.0
IMPULSE_DURATION_STEPS = 10
IMPULSE_START_FRAC = 0.5


def _worker(args):
    """Run one walker episode. Returns tau trace + fall time."""
    seed, mismatch, apply_impulse = args

    import numpy as _np

    repo_root = REPO_ROOT
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from agents.mpc import make_mpc
    from agents.mujoco_dynamics import WalkerDynamics
    from agents.adaptation import make_adapter
    from simulations.simulation import run_simulation

    _np.random.seed(int(seed))

    env = WalkerDynamics(stateless=False, speed_goal=1.5)
    agent = make_mpc('walker', H=H_WALKER, R=R_INIT, mismatch_factor=mismatch)
    adapt_args = {
        'adapt_class': 'ODEStepAdaptation',
        'adapt_params': ('recompute',),
        'adapt_kwargs': {
            'min_error_threshold': MIN_ERROR_THRESHOLD,
            'relax_step': RELAX_STEP,
        },
    }
    agent.adaptation = make_adapter(adapt_args)
    env.reset(env.get_default_initial_state())

    impulse_fn = None
    if apply_impulse:
        start_step = int(IMPULSE_START_FRAC * N_STEPS)
        end_step = start_step + IMPULSE_DURATION_STEPS
        torso_body_id = env._torso_id

        def _fn(e, i):
            e._mj_data.xfrc_applied[torso_body_id, :] = 0.0
            if start_step <= i < end_step:
                e._mj_data.xfrc_applied[torso_body_id, 0] = IMPULSE_N

        impulse_fn = _fn

    _, _, history = run_simulation(
        agent, env, n_steps=N_STEPS,
        env_perturbation_fn=impulse_fn, interval=None,
    )

    tau = _np.asarray(history.get_item_history('recompute_interval'),
                      dtype=float)
    states = history.get_item_history('state')
    torso_z = states[:, 18].astype(float)
    fell = _np.where(torso_z < 0.7)[0]
    fall_time = float(fell[0] * DT) if len(fell) > 0 else float('nan')

    return dict(
        seed=int(seed), mismatch=float(mismatch),
        apply_impulse=bool(apply_impulse),
        tau=tau, fall_time=fall_time,
    )


def _summarize(results, label):
    """Aggregate across seeds: mean/p10/p90 of episode-mean tau and tau at sample times."""
    taus = np.stack([r['tau'] for r in results])  # shape (n_seeds, n_steps)
    fall_times = np.array([r['fall_time'] for r in results])

    episode_means = taus.mean(axis=1)  # per-episode tau_bar

    sample_steps = [int(round(t / DT)) - 1 for t in SAMPLE_TIMES_SEC]
    # Clip to last step in case of rounding
    sample_steps = [min(s, taus.shape[1] - 1) for s in sample_steps]
    tau_at_times = taus[:, sample_steps]  # (n_seeds, n_times)

    lines = []
    lines.append(f'--- {label} ---')
    lines.append(f'  n_seeds: {len(results)}')
    lines.append(f'  episode-mean tau (tau_bar):')
    lines.append(f'    mean={episode_means.mean():.3f}  '
                 f'p10={np.percentile(episode_means, 10):.3f}  '
                 f'p90={np.percentile(episode_means, 90):.3f}  '
                 f'per-seed={np.round(episode_means, 2).tolist()}')
    lines.append(f'  tau at sample times (mean across seeds):')
    hdr = '    ' + '  '.join(f't={t:>4.1f}s' for t in SAMPLE_TIMES_SEC)
    lines.append(hdr)
    vals = tau_at_times.mean(axis=0)
    lines.append('    ' + '  '.join(f'{v:>7.3f}' for v in vals))
    fall_frac = np.mean(~np.isnan(fall_times))
    lines.append(f'  fall rate: {fall_frac:.2f}  fall_times={np.round(fall_times, 2).tolist()}')
    return '\n'.join(lines), dict(
        episode_means=episode_means,
        tau_at_times=vals,
        taus=taus,
        fall_times=fall_times,
    )


def _post_impulse_dip(taus_impulse, impulse_start_step):
    """Report min tau and sustained-dip duration after the impulse.

    Target: tau <= 7 for >= 0.5 s (>= 50 consecutive steps) after impulse.
    """
    mean_tau_over_time = taus_impulse.mean(axis=0)  # mean across seeds
    post = mean_tau_over_time[impulse_start_step:]
    min_post = float(post.min())
    # Longest run of post-impulse steps with tau <= 7
    below = (post <= 7.0).astype(int)
    best_run = 0
    cur = 0
    for b in below:
        cur = cur + 1 if b else 0
        best_run = max(best_run, cur)
    run_sec = best_run * DT
    return min_post, run_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='3 seeds per arm, skip impulse (~45 s)')
    parser.add_argument('--impulse', action='store_true',
                        help='Also run a 5-seed r=1.0 impulse arm for contraction check')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Seeds per arm (default 5, quick overrides to 3)')
    args = parser.parse_args()

    n_seeds = 3 if args.quick else args.seeds
    rng = np.random.RandomState(SEED)
    seeds = rng.randint(0, 2**31 - 1, size=n_seeds)

    arms = [
        ('r=1.0 (matched)', 1.0, False),
        ('r=1.5 (mismatched)', 1.5, False),
    ]
    if args.impulse and not args.quick:
        arms.append(('r=1.0 + 300N impulse @ t=5s', 1.0, True))

    all_jobs = []
    for label, m, imp in arms:
        for s in seeds:
            all_jobs.append((int(s), float(m), bool(imp)))

    print(f'[probe] {len(all_jobs)} episodes '
          f'({len(arms)} arms x {n_seeds} seeds)')

    raw = run_pool(_worker, all_jobs, verbose=1)

    by_arm = {label: [] for label, _, _ in arms}
    for r in raw:
        for label, m, imp in arms:
            if abs(r['mismatch'] - m) < 1e-9 and r['apply_impulse'] == imp:
                by_arm[label].append(r)
                break

    print()
    print('=' * 72)
    print('Walker adapter probe')
    print('=' * 72)
    print(f'Config: H={H_WALKER}, R_init={R_INIT}, N_STEPS={N_STEPS} ({N_STEPS*DT:.0f} s), '
          f'threshold={MIN_ERROR_THRESHOLD}')
    print(f'Seeds per arm: {n_seeds}')
    print()

    agg = {}
    for label, _, _ in arms:
        text, data = _summarize(by_arm[label], label)
        print(text)
        print()
        agg[label] = data

    print('=' * 72)
    print('TARGETS')
    print('=' * 72)

    t_r1 = agg['r=1.0 (matched)']['episode_means'].mean()
    t_r15 = agg['r=1.5 (mismatched)']['episode_means'].mean()
    print(f'  r=1.0 episode-mean tau_bar: {t_r1:>6.3f}  target <= 9.00  '
          f'{"OK" if t_r1 <= 9.0 else "FAIL"}')
    print(f'  r=1.5 episode-mean tau_bar: {t_r15:>6.3f}  target <= 4.00  '
          f'{"OK" if t_r15 <= 4.0 else "FAIL"}')

    if 'r=1.0 + 300N impulse @ t=5s' in agg:
        imp_taus = agg['r=1.0 + 300N impulse @ t=5s']['taus']
        imp_fall = agg['r=1.0 + 300N impulse @ t=5s']['fall_times']
        impulse_step = int(IMPULSE_START_FRAC * N_STEPS)

        min_post, dip_dur = _post_impulse_dip(imp_taus, impulse_step)
        print(f'  Post-impulse min mean tau: {min_post:>6.3f}  target <= 7.00  '
              f'{"OK" if min_post <= 7.0 else "FAIL"}')
        print(f'  Post-impulse <=7 duration: {dip_dur:>6.3f} s  target >= 0.50 s  '
              f'{"OK" if dip_dur >= 0.5 else "FAIL"}')

        # Per-seed post-impulse min tau (includes falling episodes whose tau
        # stays stuck). The mean-across-seeds check above is pessimistic when
        # many seeds fall; this surfaces per-seed contraction to diagnose.
        print(f'  Per-seed post-impulse min tau: '
              f'{np.round(imp_taus[:, impulse_step:].min(axis=1), 2).tolist()}')
        survivors = np.isnan(imp_fall)
        if survivors.any():
            surv_taus = imp_taus[survivors][:, impulse_step:]
            print(f'  Survivors-only post-impulse min tau: '
                  f'{np.round(surv_taus.min(axis=1), 2).tolist()}  '
                  f'(n={int(survivors.sum())})')

    print()


if __name__ == '__main__':
    main()
