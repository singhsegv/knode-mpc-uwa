"""
Headless data collection for cartesian-space trajectory tracking.

Uses Pinocchio for rigid-body dynamics (M, C, G), Jacobian, and RK4 integration.
Records q, qdot, and tau at 500 Hz (dt=0.002 s) with no visualization.
"""
import argparse
import os

import numpy as np
from scipy.optimize import fsolve

from robot import PlanarArm
from utils.trajectory_generator import circle_trajectory, lemniscate_trajectory

DT = 0.002  # 500 Hz
Kp = np.array([50.0, 50.0, 50.0])
Kd = np.array([4.0, 4.0, 4.0])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "data", "cartesian_circle.npz")


def ee_fk(q):
    """Planar 3R FK; returns EE (x, y, theta) in the arm plane."""
    q1, q2, q3 = q
    L1, L2, L3 = 1.0, 1.0, 0.25
    c1, c12, c123 = q1, q1 + q2, q1 + q2 + q3
    x = L1 * np.cos(c1) + L2 * np.cos(c12) + L3 * np.cos(c123)
    y = L1 * np.sin(c1) + L2 * np.sin(c12) + L3 * np.sin(c123)
    theta = q1 + q2 + q3
    return np.array([x, y, theta])


def ik_residual(q_seed, x_des, y_des, theta_des):
    sol = ee_fk(q_seed)
    x, y, theta = sol[0], sol[1], sol[2]
    return x - x_des, y - y_des, theta - theta_des


def collect(duration=20.0, trajectory="circle", r=0.4, cx=0.7, cy=1.5, omega=0.8, a=0.5):
    robot = PlanarArm(enable_viz=False)

    n_steps = int(duration / DT)
    t_log = np.empty(n_steps)
    q_log = np.empty((n_steps, 3))
    qdot_log = np.empty((n_steps, 3))
    tau_log = np.empty((n_steps, 3))

    q = np.array([0.5, 0.5, 0.5])
    qdot = np.zeros(3)
    q_prev = q.copy() # Just as a seed for the solver
    traj_fn = circle_trajectory if trajectory == "circle" else lemniscate_trajectory

    for i in range(n_steps):
        t = i * DT
        t_log[i] = t

        if trajectory == "circle":
            x, y, xdot, ydot, xddot, yddot = traj_fn(t, r=r, cx=cx, cy=cy, omega=omega)
        else:
            x, y, xdot, ydot, xddot, yddot = traj_fn(t, a=a, cx=cx, cy=cy, omega=omega)

        # Holding a static pose
        # x, y, xdot, ydot, xddot, yddot = 0.5, 1.5, 0.0, 0.0, 0.0, 0.0

        # TODO: Theta implementation was sketchy
        theta, theta_dot, theta_ddot = 0.0, 0.0, 0.0

        # ---------------------------------------------
        # Computer Torque Controller Implementation
        # ---------------------------------------------
        q_des = fsolve(ik_residual, q_prev, args=(x, y, theta))
        J = robot.ee_jacobian(q_des)
        J_inv = np.linalg.inv(J)

        qdot_des = J_inv @ np.array([xdot, ydot, theta_dot])
        qddot_des = J_inv @ np.array([xddot, yddot, theta_ddot]) # TODO: Missing J dot thingy
        q_prev = q_des.copy()

        e = q_des - q
        ed = qdot_des - qdot
        qdd_fb = qddot_des + Kp * e + Kd * ed
        tau = robot.inverse_dynamics(q, qdot, qdd_fb)
        # tau = np.array([0.0, 0.0, 0.0])

        # TODO: We might be sketchy: Check data written at the beginning/end of the step
        q_log[i] = q
        qdot_log[i] = qdot
        tau_log[i] = tau

        # robot.display(q)

        q, qdot = robot.step_rk4(q, qdot, tau, DT)

    return {
        "t": t_log,
        "q": q_log,
        "qdot": qdot_log,
        "tau": tau_log,
        "dt": DT,
        "trajectory": trajectory,
        "r": r,
        "a": a,
        "cx": cx,
        "cy": cy,
        "omega": omega,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect cartesian-space trajectory data at 500 Hz (Pinocchio).")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--trajectory", choices=["circle", "lemniscate"], default="circle")
    parser.add_argument("--r", type=float, default=0.4)
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--cx", type=float, default=0.7)
    parser.add_argument("--cy", type=float, default=1.5)
    parser.add_argument("--omega", type=float, default=0.8)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    log = collect(
        duration=args.duration,
        trajectory=args.trajectory,
        r=args.r,
        cx=args.cx,
        cy=args.cy,
        omega=args.omega,
        a=args.a,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, **log)
    print(f"Saved {len(log['t'])} samples at {1.0 / DT:.0f} Hz -> {args.output}")


if __name__ == "__main__":
    main()
