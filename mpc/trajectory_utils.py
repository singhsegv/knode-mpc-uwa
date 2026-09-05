import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.trajectory_generator import sample_trajectory, get_trajectory_start


def _task_dim(robot):
    return getattr(robot, "task_space_dim", 3)


def sample_task_space_trajectory(t, spec):
    """Sample desired task-space pose from a dataset trajectory spec."""
    return sample_trajectory(t, spec)


def get_trajectory_start_position(spec):
    """Return the task-space start position (x, y, z) for a trajectory spec."""
    return get_trajectory_start(spec)


def inverse_kinematics(robot, q_prev, pos_des, vel_des, damping=0.05, posture_weight=0.05):
    """
    Numerical IK using Pinocchio FK and damped least-squares velocity mapping.
    pos_des: (3,) desired EE position
    vel_des: (3,) desired EE linear velocity
    """
    pos_des = np.asarray(pos_des, dtype=float)
    vel_des = np.asarray(vel_des, dtype=float)
    task_dim = getattr(robot, "task_space_dim", 3)
    q_prev = np.asarray(q_prev, dtype=float)

    def fk_residual(q):
        pos_err = robot.ee_position(q) - pos_des
        if robot.nq > task_dim:
            posture_err = posture_weight * (q - q_prev)
            return np.concatenate([pos_err, posture_err])
        return pos_err

    result = least_squares(fk_residual, q_prev, method="trf")
    q_des = result.x

    J = robot.ee_jacobian(q_des)
    JJt = J @ J.T
    damp = damping ** 2 * np.eye(JJt.shape[0])
    J_pinv = J.T @ np.linalg.inv(JJt + damp)
    qdot_des = J_pinv @ vel_des

    return q_des, qdot_des


def get_task_space_trajectory_mpc(start_time, mpc_horizon, dt, robot, q_prev, spec=None):
    nq = robot.nq
    nx = 2 * nq
    X_ref = np.zeros((nx, mpc_horizon + 1))

    for i in range(mpc_horizon + 1):
        t = start_time + (i * dt)
        if spec is None:
            x, y, z = 0.4, 0.2, 0.4
            xdot = ydot = zdot = 0.0
        else:
            x, y, z, xdot, ydot, zdot, _, _, _ = sample_task_space_trajectory(t, spec)

        q_des, qdot_des = inverse_kinematics(
            robot, q_prev, np.array([x, y, z]), np.array([xdot, ydot, zdot]),
        )
        X_ref[:, i] = np.concatenate([q_des, qdot_des])
        q_prev = q_des.copy()

    return X_ref


def get_desired_task_space_from_history(history_x, dt, spec=None):
    """Reconstruct desired EE (x, y, z) at each logged state."""
    n = len(history_x)
    desired_xyz = np.zeros((n, 3))

    for i in range(n):
        t = i * dt
        if spec is None:
            desired_xyz[i] = (0.4, 0.2, 0.4)
        else:
            x, y, z, _, _, _, _, _, _ = sample_task_space_trajectory(t, spec)
            desired_xyz[i] = (x, y, z)

    return desired_xyz


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


def compute_task_space_tracking_stats(history_x, robot, dt, spec=None):
    """Compute task-space tracking error statistics."""
    history_x = np.asarray(history_x)
    nq = robot.nq
    actual_xyz = np.array([robot.ee_position(q) for q in history_x[:, :nq]])
    desired_xyz = get_desired_task_space_from_history(history_x, dt, spec=spec)
    errors = np.linalg.norm(actual_xyz - desired_xyz, axis=1)
    dtw_total, dtw_mean = dtw_distance(actual_xyz, desired_xyz)

    return {
        "task_space_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "task_space_mean_error": float(np.mean(errors)),
        "task_space_max_error": float(np.max(errors)),
        "task_space_final_error": float(errors[-1]),
        "task_space_dtw": dtw_total,
        "task_space_dtw_mean": dtw_mean,
    }


def plot_task_space_tracking(history_x, robot, dt, spec=None, show=True, save_path=None):
    """Plot tracked vs desired end-effector position from closed-loop MPC history."""
    history_x = np.asarray(history_x)
    n = len(history_x)
    time = np.arange(n) * dt
    nq = robot.nq
    task_dim = getattr(robot, "task_space_dim", 3)

    actual_xyz = np.array([robot.ee_position(q) for q in history_x[:, :nq]])
    desired_xyz = get_desired_task_space_from_history(history_x, dt, spec=spec)
    q_traj = history_x[:, :nq]

    title_suffix = ""
    if spec is not None:
        title_suffix = f" ({spec['type']})"

    if task_dim == 3:
        from utils.robot_viz import plot_task_space_3d_scene

        fig = plt.figure(figsize=(14, 6))
        ax3d = fig.add_subplot(121, projection="3d")
        plot_task_space_3d_scene(
            ax3d, robot, q_traj, actual_xyz, ee_desired=desired_xyz,
        )
        ax3d.set_title(f"Task-space path (3D){title_suffix}")

        ax_time = fig.add_subplot(122)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_path = axes[0]
        ax_path.plot(desired_xyz[:, 0], desired_xyz[:, 1], "g--", label="desired", linewidth=1.5)
        ax_path.plot(actual_xyz[:, 0], actual_xyz[:, 1], "r-", label="tracked", linewidth=1.2)
        ax_path.set_xlabel("x [m]")
        ax_path.set_ylabel("y [m]")
        ax_path.set_title(f"Task-space path (x-y){title_suffix}")
        ax_path.axis("equal")
        ax_path.grid(True, alpha=0.3)
        ax_path.legend()
        ax_time = axes[1]

    for idx, label in enumerate(["x", "y", "z"][:task_dim]):
        ax_time.plot(time, desired_xyz[:, idx], "--", alpha=0.7, label=f"desired {label}", linewidth=1.5)
        ax_time.plot(time, actual_xyz[:, idx], "-", label=f"tracked {label}", linewidth=1.2)
        
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

    return fig, actual_xyz, desired_xyz
