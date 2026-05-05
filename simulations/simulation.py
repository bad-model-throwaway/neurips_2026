"""Core simulation loop and multiprocessing utilities."""

import multiprocessing
import time
import numpy as np
from agents.history import History
from configs import N_WORKERS


def run_simulation(agent, env, n_steps=1000, perturbation=None,
                   env_perturbation_fn=None, env_mismatch_fn=None,
                   history=None, interval=50):
    """Run agent-environment interaction loop.

    perturbation: optional array of shape [n_steps] or [n_steps, action_dim].
        External force added to the environment action at each step.
        The agent does not observe the perturbation directly.
    env_perturbation_fn: optional callable `fn(env, step_idx)` invoked just
        before each `env.step(...)`. Intended for env-level physical
        perturbations (e.g. MuJoCo `xfrc_applied`) that cannot be expressed
        as an additive action — walker Supplement 5 uses this to apply a
        torso force during a stance window. The agent does not observe it.
    env_mismatch_fn: optional callable `fn(env, step_idx)` invoked just
        before each `env.step(...)`. Intended for one-time mid-episode
        parameter changes (e.g. pole length, gravity). The agent's planning
        model is never informed of the change.
    """

    # Track simulation history
    if history is None:
        history = History(agent, env)

    # Run interaction loop
    for i in range(n_steps):
        # Perform interaction
        action = agent.interact(env.state, env.cost)

        # Save history
        history.update(i, agent, env)

        # Perturbation is applied to the env only (agent does not observe it)
        env_action = action + perturbation[i] if perturbation is not None else action
        if env_perturbation_fn is not None:
            env_perturbation_fn(env, i)
        if env_mismatch_fn is not None:
            env_mismatch_fn(env, i)
        env.step(env_action)

        # Print status at specified intervals
        if interval is not None and i % interval == 0:
            history.print_last()

    return agent, env, history


def run_sequential(func, args_list, verbose=1, on_result=None):
    """Run func over args_list in the calling process (no multiprocessing).

    Use this on GPU where a single JAX process should own the device.
    Interface mirrors run_pool so callers can swap freely.
    """
    args_list = list(args_list)
    total = len(args_list)

    count_w = len(str(total))
    if verbose > 0:
        print(f"  Sequential: {total} jobs (GPU mode)", flush=True)

    t0 = time.perf_counter()
    time_w = 4
    results = []
    for i, args in enumerate(args_list):
        result = func(args)
        if on_result is not None:
            on_result(result)
        else:
            results.append(result)

        done = i + 1
        elapsed = time.perf_counter() - t0
        expected = elapsed / done * total
        time_w = max(time_w, len(str(int(expected))))

        should_print = (verbose >= 2
                        or (verbose >= 1 and (done % 10 == 0 or done == total)))
        if should_print:
            print(f"  {done:>{count_w}}/{total}"
                  f"   Elapsed = {elapsed:>{time_w}.0f}s"
                  f"   Expected ~ {expected:>{time_w}.0f}s")

    return results


def run_pool(func, args_list, n_processes=None, verbose=1, on_result=None,
             maxtasksperchild=20):
    """Run func over args_list using multiprocessing pool.

    func: callable taking a single argument
    args_list: iterable of arguments to map over
    n_processes: number of worker processes (defaults to N_WORKERS from configs,
        which is already clipped to cpu_count - 2)
    verbose: 0 = silent, 1 = progress every 10 jobs and at the end, 2 = same as 1
    on_result: optional callback invoked with each result as it arrives.
        When provided, results are not accumulated in memory.
    maxtasksperchild: recycle each worker after this many tasks. Defaults to 20
        to bound RSS — slow MuJoCo/numpy memory bloat can otherwise cause workers
        to be OOM-killed while the parent dispatcher waits indefinitely for
        results from dead workers. Pass None to disable.
    """
    args_list = list(args_list)
    total = len(args_list)

    if n_processes is None:
        n_processes = N_WORKERS

    count_w = len(str(total))
    if verbose > 0:
        print(f"  Pool: {total} jobs on {n_processes} workers "
              f"(maxtasksperchild={maxtasksperchild})", flush=True)

    t0 = time.perf_counter()
    time_w = 4
    results = []
    pool_kwargs = {'processes': n_processes}
    if maxtasksperchild is not None:
        pool_kwargs['maxtasksperchild'] = maxtasksperchild
    with multiprocessing.Pool(**pool_kwargs) as pool:
        for i, result in enumerate(pool.imap_unordered(func, args_list)):
            if on_result is not None:
                on_result(result)
            else:
                results.append(result)

            done = i + 1
            elapsed = time.perf_counter() - t0
            expected = elapsed / done * total
            time_w = max(time_w, len(str(int(expected))))

            if verbose >= 1 and (done % 10 == 0 or done == total):
                print(f"  {done:>{count_w}}/{total}"
                      f"   Elapsed = {elapsed:>{time_w}.0f}s"
                      f"   Expected ~ {expected:>{time_w}.0f}s",
                      flush=True)

    return results
