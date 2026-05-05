import numpy as np

from agents.mpc import SplinePSProposal, SplinePSArgminDecision


def _make_prop(action_dim=4, H=50, N=32, P=3, sigma=0.1, interp='cubic',
               sigma2=0.0, mix_prob=0.2, include_nominal=True, clip=True):
    return SplinePSProposal(
        action_dim=action_dim, tsteps=H, n_samples=N, dt=0.02,
        ctrl_low=-np.ones(action_dim), ctrl_high=np.ones(action_dim),
        P=P, sigma=sigma, interp=interp, sigma2=sigma2, mix_prob=mix_prob,
        include_nominal=include_nominal, clip=clip,
    )


def test_proposal_shape():
    prop = _make_prop()
    prop.update_parameters({'recompute_interval': 1, 'horizon': 50})
    actions = prop(state=np.zeros(14))
    assert actions.shape == (32, 4, 50)


def test_nominal_in_sample_is_exact():
    np.random.seed(0)
    prop = _make_prop()
    prop.update_parameters({'recompute_interval': 1, 'horizon': 50})
    actions = prop(state=np.zeros(14))
    expected = prop._render(prop.plan)
    assert np.array_equal(actions[0], expected)


def test_noise_std_matches_formula():
    # sigma=0.5, ctrlrange=[-1,1] -> per-actuator std = 0.5*(high-low)*sigma = 0.5*2*0.5 = 0.5
    np.random.seed(42)
    N = 4000
    prop = _make_prop(N=N, sigma=0.5, include_nominal=True)
    prop.update_parameters({'recompute_interval': 1, 'horizon': 50})
    actions = prop(state=np.zeros(14))
    # compare to rendered nominal; drop sample 0 (nominal-in-sample)
    base = prop._render(prop.plan)
    # pick unclipped samples: use sigma=0.5 → rarely clipped around zero nominal
    resid = actions[1:] - base[None, :, :]
    std = resid[:, :, 0].std(axis=0)  # per-actuator std at t=0
    assert np.allclose(std, 0.5, atol=0.05), f"std {std} not near 0.5"


def test_clipping_saturates_at_bounds():
    np.random.seed(0)
    prop = _make_prop(sigma=100.0)
    prop.update_parameters({'recompute_interval': 1, 'horizon': 50})
    actions = prop(state=np.zeros(14))
    assert actions.min() >= -1.0 - 1e-12
    assert actions.max() <= 1.0 + 1e-12


def test_warm_start_non_sliding_resample():
    prop = _make_prop(P=3, interp='cubic', H=50)
    prop.update_parameters({'recompute_interval': 3, 'horizon': 50})
    from agents.spline import TimeSpline
    old = TimeSpline(4, 'cubic')
    knot_times = prop._knot_times
    old.add_knot(float(knot_times[0]), np.array([0.1, 0.2, 0.3, 0.4]))
    old.add_knot(float(knot_times[1]), np.array([0.5, 0.4, 0.3, 0.2]))
    old.add_knot(float(knot_times[2]), np.array([-0.1, -0.2, -0.3, -0.4]))
    prop.plan = old

    prop.advance_nominal(old)
    new_knots = prop.plan.knots
    # each new knot should equal old.sample(R*dt + k*shift)
    R_dt = prop.recompute_interval * prop.dt
    for k_idx, t in enumerate(knot_times):
        expected = old.sample(float(t + R_dt))
        assert np.allclose(new_knots[k_idx], expected)
    assert not np.allclose(new_knots, 0.0)
    assert not np.allclose(new_knots, new_knots[-1][None, :])


def test_warm_start_preserved_across_horizon_change():
    # When TheoryStepAdaptation changes horizon, update_parameters should
    # re-grid the existing plan onto the new knot times rather than zeroing
    # it (otherwise the next replan starts from a constant-zero nominal,
    # which trips humanoid_balance under sustained mismatch).
    prop = _make_prop(P=3, interp='cubic', H=50)
    prop.update_parameters({'recompute_interval': 1, 'horizon': 50})
    from agents.spline import TimeSpline
    seeded = TimeSpline(4, 'cubic')
    knot_times = prop._knot_times
    seeded.add_knot(float(knot_times[0]), np.array([0.1, 0.2, 0.3, 0.4]))
    seeded.add_knot(float(knot_times[1]), np.array([0.5, 0.4, 0.3, 0.2]))
    seeded.add_knot(float(knot_times[2]), np.array([-0.1, -0.2, -0.3, -0.4]))
    prop.plan = seeded
    old_t_max = float(knot_times[-1])

    # Increase horizon: new last knot exceeds old support; clamp policy holds last.
    prop.update_parameters({'recompute_interval': 1, 'horizon': 70})
    new_knot_times = prop._knot_times
    for k_idx, t in enumerate(new_knot_times):
        expected = seeded.sample(min(float(t), old_t_max))
        assert np.allclose(prop.plan.knots[k_idx], expected), (
            f"H-increase k={k_idx}: knot mismatch"
        )
    assert not np.allclose(prop.plan.knots, 0.0), "plan should not be zeroed"

    # Decrease horizon: new knots all inside old support; pure interpolation.
    prop.plan = seeded.copy()
    prop._knot_times = knot_times  # reset to H=50 grid
    prop.tsteps = 50
    prop.update_parameters({'recompute_interval': 1, 'horizon': 30})
    new_knot_times = prop._knot_times
    for k_idx, t in enumerate(new_knot_times):
        expected = seeded.sample(float(t))
        assert np.allclose(prop.plan.knots[k_idx], expected), (
            f"H-decrease k={k_idx}: knot mismatch"
        )

    # H unchanged: branch should not fire.
    prop.plan = seeded.copy()
    prop.update_parameters({'recompute_interval': 1, 'horizon': prop.tsteps})
    assert np.array_equal(prop.plan.knots, seeded.knots), "no-op H update mutated plan"


def test_two_sigma_mixture_samples_from_both():
    # MJPC planner.cc:335-338 mixes a second noise scale `sigma2` in with
    # probability `mix_prob`. Draw a large batch, classify each sample by
    # whether its empirical knot-noise std is closer to `sigma` or `sigma2`
    # (split at the log-midpoint), and assert both modes are populated.
    np.random.seed(0)
    N = 2000
    prop = _make_prop(action_dim=4, H=10, N=N, P=3,
                      sigma=0.1, sigma2=1.0, mix_prob=0.5,
                      include_nominal=False, clip=False)
    prop.update_parameters({'recompute_interval': 1, 'horizon': 10})
    prop(state=np.zeros(14))

    # Nominal plan is all zeros, so each sample's knots are pure noise at
    # scale (ctrl_high - ctrl_low)/2 * sigma_eff = 1.0 * sigma_eff.
    knots = np.stack([s.knots for s in prop._candidate_splines])  # (N, P, action_dim)
    per_sample_std = knots.reshape(N, -1).std(axis=1)

    mid = np.sqrt(0.1 * 1.0)  # geometric mean of the two σ scales
    near_sigma2 = per_sample_std > mid
    near_sigma  = per_sample_std <= mid

    frac_sigma2 = float(near_sigma2.mean())
    frac_sigma  = float(near_sigma.mean())
    assert frac_sigma2 >= 0.25, f"sigma2 mode under-sampled: {frac_sigma2:.3f}"
    assert frac_sigma  >= 0.25, f"sigma mode under-sampled: {frac_sigma:.3f}"


def test_argmin_decision_picks_min():
    prop = _make_prop(N=3)
    prop.update_parameters({'recompute_interval': 1, 'horizon': 50})
    _ = prop(state=np.zeros(14))

    dec = SplinePSArgminDecision(proposal=prop)
    proposals = np.random.randn(3, 4, 50)
    evaluations = np.array([5.0, 1.0, 3.0])
    actions, best_idx = dec(proposals, None, evaluations, n_actions=1)
    assert best_idx == 1
    assert np.array_equal(actions[0], proposals[1, :, 0])


