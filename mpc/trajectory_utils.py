import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


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


def task_space_circle_trajectory(t, r=0.25, cx=1.0, cy=1.0, omega=1.0):
    phi = omega * t
    c, s = np.cos(phi), np.sin(phi)
    x = cx + r * c
    y = cy + r * s
    xdot = -r * omega * s
    ydot = r * omega * c
    xddot = -r * omega**2 * c
    yddot = -r * omega**2 * s
    return x, y, xdot, ydot, xddot, yddot


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


def get_task_space_trajectory_mpc(start_time, mpc_horizon, dt, robot, q_prev):
    """ Hardcoded to a circle for now. """
    nx = 4
    X_ref = np.zeros((nx, mpc_horizon + 1))

    for i in range(mpc_horizon + 1):
        t = start_time + (i * dt)
        x, y, xdot, ydot, xddot, yddot = task_space_circle_trajectory(t)

        q_des, qdot_des = inverse_kinematics(robot, q_prev, x, y, xdot, ydot)

        q1_des, q2_des, qdot1_des, qdot2_des = q_des[0], q_des[1], qdot_des[0], qdot_des[1]
        
        # NOTE: Column based stuff for CasADi
        X_ref[:, i] = np.array([q1_des, q2_des, qdot1_des, qdot2_des])

        q_prev = q_des.copy()

    return X_ref


def get_desired_task_space_from_history(history_x, dt, robot):
    """Reconstruct desired EE (x, y) at each logged state from MPC joint references."""
    n = len(history_x)
    desired_xy = np.zeros((n, 2))

    for i in range(n):
        t = 0.0 if i == 0 else (i - 1) * dt
        q_prev = history_x[i, :2] if i == 0 else history_x[i - 1, :2]
        X_ref = get_task_space_trajectory_mpc(
            t, mpc_horizon=1, dt=dt, robot=robot, q_prev=q_prev
        )
        desired_xy[i] = ee_fk(X_ref[:2, 0])

    return desired_xy


def plot_task_space_tracking(history_x, robot, dt, show=True, save_path=None):
    """Plot tracked vs desired end-effector (x, y) from closed-loop MPC history."""
    history_x = np.asarray(history_x)
    n = len(history_x)
    time = np.arange(n) * dt

    actual_xy = np.array([ee_fk(q) for q in history_x[:, :2]])
    desired_xy = get_desired_task_space_from_history(history_x, dt, robot)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax_path = axes[0]
    ax_path.plot(desired_xy[:, 0], desired_xy[:, 1], "g--", label="desired", linewidth=1.5)
    ax_path.plot(actual_xy[:, 0], actual_xy[:, 1], "r-", label="tracked", linewidth=1.2)
    ax_path.set_xlabel("x [m]")
    ax_path.set_ylabel("y [m]")
    ax_path.set_title("Task-space path")
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
    ax_time.set_title("Task-space tracking vs time")
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