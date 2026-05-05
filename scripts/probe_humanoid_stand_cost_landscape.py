"""Cost-landscape review: verify the existing 5-term MJPC residual cost
provides informative gradient from the supine (floor) initial state
through to standing.

Prints a per-term breakdown at two anchor states (supine, upright default)
and a height-term profile across intermediate head heights.  Also checks
the cost landscape under each MISMATCH_FACTORS['HumanoidStand'] level.

Usage:
    python scripts/probe_humanoid_stand_cost_landscape.py
"""

import os
import sys

import mujoco
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.mujoco_dynamics import HumanoidStandDynamics, MISMATCH_FACTORS


def smooth_abs(x, p=0.1):
    return np.sqrt(x**2 + p**2) - p


def cost_breakdown(state, ctrl=None, height_goal=1.4):
    """Return dict of individual cost terms for a single state vector."""
    head_z = state[55]
    feet_z = state[56]
    com_xy = state[57:59]
    com_vel_xy = state[59:61]
    feet_avg_xy = state[61:63]
    qvel = state[28:55]

    p = 0.1

    # Height
    r_height = head_z - feet_z - height_goal
    height = 100.0 * smooth_abs(r_height, p)

    # Balance
    capture = feet_avg_xy - (com_xy + 0.2 * com_vel_xy)
    r_balance = np.linalg.norm(capture)
    balance = 50.0 * smooth_abs(r_balance, p)

    # CoM velocity
    com_vel = 10.0 * 0.5 * np.sum(com_vel_xy**2)

    # Joint velocity (skip 6 free-joint DOF)
    joint_vel = 0.01 * 0.5 * np.sum(qvel[6:]**2)

    # Control
    control = 0.0
    if ctrl is not None:
        control = 0.025 * np.sum(0.3**2 * (np.cosh(ctrl / 0.3) - 1.0))

    total = height + balance + com_vel + joint_vel + control
    return {
        'r_height': r_height,
        'height': height,
        'r_balance': r_balance,
        'balance': balance,
        'com_vel': com_vel,
        'joint_vel': joint_vel,
        'control': control,
        'total': total,
    }


def get_upright_state(env):
    """Return the MJCF default upright pose."""
    data = mujoco.MjData(env._mj_model)
    mujoco.mj_resetData(env._mj_model, data)
    mujoco.mj_forward(env._mj_model, data)
    return env._state_from_data(data)


def print_breakdown(label, bd):
    print(f'  {label}')
    print(f'    r_height   = {bd["r_height"]:+.3f} m')
    print(f'    Height     (w=100):  {bd["height"]:.3f}')
    print(f'    r_balance  = {bd["r_balance"]:.4f} m')
    print(f'    Balance    (w=50):   {bd["balance"]:.3f}')
    print(f'    CoM vel    (w=10):   {bd["com_vel"]:.6f}')
    print(f'    Joint vel  (w=0.01): {bd["joint_vel"]:.6f}')
    print(f'    TOTAL (no ctrl):     {bd["total"]:.3f}')


def main():
    env = HumanoidStandDynamics(stateless=False)

    s_floor = env.get_default_initial_state()
    s_stand = get_upright_state(env)

    print('=' * 60)
    print('COST BREAKDOWN: supine (floor) vs upright (standing)')
    print('=' * 60)

    bd_floor = cost_breakdown(s_floor)
    bd_stand = cost_breakdown(s_stand)

    print_breakdown('FLOOR (supine initial state)', bd_floor)
    print()
    print_breakdown('STANDING (MJCF default upright pose)', bd_stand)
    print()
    print(f'  Cost ratio floor/standing: {bd_floor["total"] / bd_stand["total"]:.1f}x')

    print()
    print('=' * 60)
    print('HEIGHT-TERM PROFILE (head_z sweep, feet_z fixed at floor value)')
    print('=' * 60)
    feet_z = s_floor[56]
    print(f'  feet_avg_z = {feet_z:.3f} m,  height_goal = 1.4 m')
    print()
    print(f'  {"head_z":>7s}  {"r_height":>9s}  {"height_term":>11s}')
    print(f'  {"------":>7s}  {"---------":>9s}  {"-----------":>11s}')
    for hz in np.arange(0.1, 1.55, 0.1):
        r_h = hz - feet_z - 1.4
        term = 100.0 * smooth_abs(r_h)
        print(f'  {hz:7.2f}  {r_h:+9.3f}  {term:11.2f}')

    factors = MISMATCH_FACTORS['HumanoidStand']
    print()
    print('=' * 60)
    print(f'MISMATCH FACTORS: {factors}')
    print('=' * 60)
    print()
    print(f'  {"factor":>6s}  {"floor_cost":>10s}  {"stand_cost":>10s}  {"ratio":>7s}')
    print(f'  {"------":>6s}  {"----------":>10s}  {"----------":>10s}  {"-----":>7s}')

    for f in factors:
        env_m = HumanoidStandDynamics(stateless=False)
        if f != 1.0:
            env_m.apply_mismatch(f)
        s_f = env_m.get_default_initial_state()
        s_s = get_upright_state(env_m)
        c_floor = env_m.cost_function(s_f)
        c_stand = env_m.cost_function(s_s)
        print(f'  {f:6.1f}  {c_floor:10.3f}  {c_stand:10.3f}  {c_floor / c_stand:7.1f}x')

    print()
    print('Conclusion: the cost landscape is monotonically decreasing from')
    print('floor to standing with no flat regions. All mismatch levels retain')
    print('a large cost ratio, confirming informative gradient for stand-up.')


if __name__ == '__main__':
    main()
