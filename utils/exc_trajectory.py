"""
CaDeLaC-style 5th-order Fourier joint excitation references (EXC).

Trajectory form (Schulze et al. / excitation_generator.py):
  q  = Σ_k [A_k/(kω) sin(kωt) − B_k/(kω) cos(kωt)] + q0
  qd = Σ_k [A_k cos(kωt) + B_k sin(kωt)]
"""
from __future__ import annotations

import numpy as np


PANDA_Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
PANDA_Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_QD_MAX = 2.0 * np.ones(7)

# CaDeLaC MPC home pose (not the MJCF keyframe)
PANDA_Q_HOME = np.array([0.0, -0.4, 0.0, -2.4, 0.0, 2.2, -0.7853])


def _constraints(order: int):
    cA = np.ones(order)
    cB = np.array([[1 / i for i in range(1, order + 1)], [i for i in range(1, order + 1)]])
    cB_red = np.array([(cB[0] - cB[1]) / (cB[0] - cB[1])[1], cB[1]])
    return cA, cB_red


def sample_fourier_params(order: int, njoints: int, joints_range: np.ndarray, rng: np.random.Generator):
    param_norm = 0.5 * joints_range / order
    rand = rng.uniform(-1.0, 1.0, size=(2, order, njoints))
    params = rand * param_norm[None, None, :]

    cA, cB = _constraints(order)
    A, B = params[0], params[1]
    A[-1] = -np.dot(cA[:-1], A[:-1])
    B[1] = -np.dot(cB[0][2:], B[2:])
    B[0] = -np.dot(cB[1][1:], B[1:])
    return np.array([A, B])


def evaluate_fourier(params, duration, q0, qd0, t):
    """Evaluate Fourier traj at times t. params shape (2, order, nq)."""
    A, B = params
    order, nq = A.shape
    omega = 2.0 * np.pi / duration
    t = np.asarray(t, dtype=float).reshape(-1)
    q = np.tile(np.asarray(q0, dtype=float), (len(t), 1))
    qd = np.tile(np.asarray(qd0, dtype=float), (len(t), 1))
    qdd = np.zeros_like(q)

    for k in range(1, order + 1):
        wk = omega * k
        sk = np.sin(wk * t)[:, None]
        ck = np.cos(wk * t)[:, None]
        Ak = A[k - 1][None, :]
        Bk = B[k - 1][None, :]
        q += (Ak / wk) * sk - (Bk / wk) * ck
        qd += Ak * ck + Bk * sk
        qdd += -Ak * wk * sk + Bk * wk * ck
    return q, qd, qdd


def is_traj_valid(q, qd, q_min, q_max, qd_max):
    return (
        np.all(q >= q_min[None, :])
        and np.all(q <= q_max[None, :])
        and np.all(np.abs(qd) <= qd_max[None, :])
    )


def sample_valid_exc_trajectory(
    duration=10.0,
    dt=0.02,
    order=5,
    q0=None,
    seed=None,
    max_tries=10000,
):
    """Return dict with t,q,qd,qdd and Fourier params for a valid Panda EXC traj."""
    rng = np.random.default_rng(seed)
    q0 = PANDA_Q_HOME.copy() if q0 is None else np.asarray(q0, dtype=float)
    qd0 = np.zeros_like(q0)
    nq = q0.shape[0]

    q_max = PANDA_Q_MAX.copy()
    q_min = PANDA_Q_MIN.copy()
    qd_max = PANDA_QD_MAX.copy()
    joints_range = q_max - q_min
    joints_range[0] = 2.0

    t = np.arange(0.0, duration + 1e-12, dt)
    for attempt in range(max_tries):
        params = sample_fourier_params(order, nq, joints_range, rng)
        q, qd, qdd = evaluate_fourier(params, duration, q0, qd0, t)
        if is_traj_valid(q, qd, q_min, q_max, qd_max):
            return {
                "t": t,
                "q": q,
                "qd": qd,
                "qdd": qdd,
                "params": params,
                "q0": q0,
                "attempts": attempt + 1,
            }
    raise RuntimeError(f"Failed to sample valid EXC trajectory in {max_tries} tries")


def mpc_reference_horizon(traj, t0, horizon, dt):
    """
    Build X_ref of shape (nx, horizon+1) with state [q; qd] for knode-mpc convention.
    Wraps circularly like CaDeLaC get_mpc_ref.
    """
    q = traj["q"]
    qd = traj["qd"]
    n = len(traj["t"])
    i0 = int(round(t0 / dt)) % n
    X = np.zeros((q.shape[1] * 2, horizon + 1))
    for k in range(horizon + 1):
        idx = (i0 + k) % n
        X[:, k] = np.concatenate([q[idx], qd[idx]])
    return X
