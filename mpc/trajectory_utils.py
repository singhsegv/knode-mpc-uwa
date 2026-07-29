import numpy as np
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