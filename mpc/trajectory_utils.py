import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.trajectory_generator import (
    circle_trajectory,
    lemniscate_trajectory,
    line_trajectory,
    rectangle_trajectory,
)


def get_joint_space_trajectory_mpc(start_time, mpc_horizon, dt):
    """ Not so visually pleasing trajectories. """
    nx = 4
    X_ref = np.zeros((nx, mpc_horizon + 1))

    for i in range(mpc_horizon + 1):
        t = start_time + (i * dt)
        q1_ref = 0.5 * np.sin(2.0 * t)
        q2_ref = 0.5 * np.cos(2.0 * t)
        qdot1_ref = 1.0 * np.cos(2.0 * t)
        qdot2_ref = -1.0 * np.sin(2.0 * t)

        # NOTE: Column based stuff for CasADi
        X_ref[:, i] = np.array([q1_ref, q2_ref, qdot1_ref, qdot2_ref])

    return X_ref


def task_space_circle_trajectory(t, r=0.25, cx=0.5, cy=0.5, omega=1.0):
    phi = omega * t
    c, s = np.cos(phi), np.sin(phi)
    x = cx + r * c
    y = cy + r * s
    xdot = -r * omega * s
    ydot = r * omega * c
    xddot = -r * omega**2 * c
    yddot = -r * omega**2 * s
    return x, y, xdot, ydot, xddot, yddot


def sample_task_space_trajectory(t, spec):
    """Sample desired task-space pose from a dataset trajectory spec."""
    trajectory = spec["type"]
    omega = spec.get("omega", 0.3)

    if trajectory == "circle":
        return circle_trajectory(t, r=spec["r"], cx=spec["cx"], cy=spec["cy"], omega=omega)
    if trajectory == "lemniscate":
        return lemniscate_trajectory(t, a=spec["a"], cx=spec["cx"], cy=spec["cy"], omega=omega)
    if trajectory == "line":
        return line_trajectory(t, spec["lx0"], spec["ly0"], spec["lx1"], spec["ly1"], omega=omega)
    if trajectory == "rectangle":
        return rectangle_trajectory(t, spec["lx0"], spec["ly0"], spec["lx1"], spec["ly1"], omega=omega)

    raise ValueError(f"Unknown trajectory type: {trajectory}")


def get_trajectory_start_position(spec):
    """Return the task-space start position for a trajectory spec."""
    trajectory = spec["type"]
    omega = spec.get("omega", 0.3)

    if trajectory == "line":
        return 0.5 * (spec["lx0"] + spec["lx1"]), 0.5 * (spec["ly0"] + spec["ly1"])
    if trajectory == "circle":
        return spec["cx"] + spec["r"], spec["cy"]
    if trajectory == "lemniscate":
        return spec["cx"] + spec["a"], spec["cy"]
    if trajectory == "rectangle":
        x, y, _, _, _, _ = rectangle_trajectory(
            0, spec["lx0"], spec["ly0"], spec["lx1"], spec["ly1"], omega=omega
        )
        return x, y

    raise ValueError(f"Unknown trajectory type: {trajectory}")


def ee_fk(q):
    """Planar 2R FK; returns EE (x, y) in the arm plane."""
    q1, q2 = q
    L1, L2 = 1.0, 1.0
    c1, c12 = q1, q1 + q2
    x = L1 * np.cos(c1) + L2 * np.cos(c12)
    y = L1 * np.sin(c1) + L2 * np.sin(c12)
    return np.array([x, y])


def ik_residual(q_seed, x_des, y_des):
    sol = ee_fk(q_seed)
    x, y = sol[0], sol[1]
    return x - x_des, y - y_des


def inverse_kinematics(robot, q_prev, x, y, xdot, ydot):
    q_des = fsolve(ik_residual, q_prev, args=(x, y))
    J = robot.ee_jacobian(q_des)
    J_inv = np.linalg.inv(J)
    qdot_des = J_inv @ np.array([xdot, ydot])
    return q_des, qdot_des


def get_task_space_trajectory_mpc(start_time, mpc_horizon, dt, robot, q_prev, spec=None):
    nx = 4
    X_ref = np.zeros((nx, mpc_horizon + 1))

    for i in range(mpc_horizon + 1):
        t = start_time + (i * dt)
        if spec is None:
            x, y, xdot, ydot, xddot, yddot = task_space_circle_trajectory(t)
        else:
            x, y, xdot, ydot, xddot, yddot = sample_task_space_trajectory(t, spec)

        q_des, qdot_des = inverse_kinematics(robot, q_prev, x, y, xdot, ydot)

        q1_des, q2_des, qdot1_des, qdot2_des = q_des[0], q_des[1], qdot_des[0], qdot_des[1]

        # NOTE: Column based stuff for CasADi
        X_ref[:, i] = np.array([q1_des, q2_des, qdot1_des, qdot2_des])

        q_prev = q_des.copy()

    return X_ref


def get_desired_task_space_from_history(history_x, dt, spec=None):
    """Reconstruct desired EE (x, y) at each logged state."""
    n = len(history_x)
    desired_xy = np.zeros((n, 2))

    for i in range(n):
        t = i * dt
        if spec is None:
            x, y, _, _, _, _ = task_space_circle_trajectory(t)
            desired_xy[i] = (x, y)
        else:
            x, y, _, _, _, _ = sample_task_space_trajectory(t, spec)
            desired_xy[i] = (x, y)

    return desired_xy


def dtw_distance(series_a, series_b):
    """Euclidean DTW between two (N, D) trajectories."""
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0, 0.0

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            step = np.linalg.norm(ai - b[j - 1])
            cost[i, j] = step + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    total = float(cost[n, m])
    return total, total / max(n, m)


def compute_task_space_tracking_stats(history_x, dt, spec=None):
    """Compute task-space tracking error statistics."""
    history_x = np.asarray(history_x)
    actual_xy = np.array([ee_fk(q) for q in history_x[:, :2]])
    desired_xy = get_desired_task_space_from_history(history_x, dt, spec=spec)
    errors = np.linalg.norm(actual_xy - desired_xy, axis=1)
    dtw_total, dtw_mean = dtw_distance(actual_xy, desired_xy)

    return {
        "task_space_rmse": float(np.sqrt(np.mean(errors**2))),
        "task_space_mean_error": float(np.mean(errors)),
        "task_space_max_error": float(np.max(errors)),
        "task_space_final_error": float(errors[-1]),
        "task_space_dtw": dtw_total,
        "task_space_dtw_mean": dtw_mean,
    }


def plot_task_space_tracking(history_x, robot, dt, spec=None, show=True, save_path=None):
    """Plot tracked vs desired end-effector (x, y) from closed-loop MPC history."""
    history_x = np.asarray(history_x)
    n = len(history_x)
    time = np.arange(n) * dt

    actual_xy = np.array([ee_fk(q) for q in history_x[:, :2]])
    desired_xy = get_desired_task_space_from_history(history_x, dt, spec=spec)

    title_suffix = ""
    if spec is not None:
        title_suffix = f" ({spec['type']})"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax_path = axes[0]
    ax_path.plot(desired_xy[:, 0], desired_xy[:, 1], "g--", label="desired", linewidth=1.5)
    ax_path.plot(actual_xy[:, 0], actual_xy[:, 1], "r-", label="tracked", linewidth=1.2)
    ax_path.set_xlabel("x [m]")
    ax_path.set_ylabel("y [m]")
    ax_path.set_title(f"Task-space path{title_suffix}")
    ax_path.axis("equal")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend()

    ax_time = axes[1]
    ax_time.plot(time, desired_xy[:, 0], "g--", label="desired x", linewidth=1.5)
    ax_time.plot(time, desired_xy[:, 1], "g--", alpha=0.7, label="desired y", linewidth=1.5)
    ax_time.plot(time, actual_xy[:, 0], "r-", label="tracked x", linewidth=1.2)
    ax_time.plot(time, actual_xy[:, 1], "r-", alpha=0.7, label="tracked y", linewidth=1.2)
    ax_time.set_xlabel("time [s]")
    ax_time.set_ylabel("position [m]")
    ax_time.set_title(f"Task-space tracking vs time{title_suffix}")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, actual_xy, desired_xy
