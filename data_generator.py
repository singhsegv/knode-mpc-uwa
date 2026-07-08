"""
Headless data collection for cartesian-space trajectory tracking.

Uses Pinocchio for rigid-body dynamics (M, C, G), Jacobian, and RK4 integration.
Records q, qdot, and tau at 500 Hz (dt=0.002 s) with no visualization.
"""
import os
import time
import argparse

import numpy as np
from scipy.optimize import fsolve

from robot import PlanarArm
from utils.trajectory_generator import circle_trajectory, lemniscate_trajectory, line_trajectory


DOF = 2
DT = 0.002  # 500 Hz
Kp = np.array([50.0, 50.0])
Kd = np.array([4.0, 4.0])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "data", "unnamed.npz")


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


def collect(duration, trajectory, cx, cy, omega, a=None, r=None, lx0=None, ly0=None, lx1=None, ly1=None):
    """
    Common: omega, duration
    Circle: cx, cy, r
    Lemniscate: cx, cy, a
    Line: lx0, ly0, lx1, ly1
    """
    robot = PlanarArm(enable_viz=True)

    n_steps = int(duration / DT)
    t_log = np.empty(n_steps)
    q_log = np.empty((n_steps, DOF))
    qdot_log = np.empty((n_steps, DOF))
    tau_log = np.empty((n_steps, DOF))

    if trajectory == "line":
        x_start, y_start = lx1, ly1
    elif trajectory == "circle":
        x_start, y_start = cx + r, cy
    else:
        x_start, y_start = cx, cy

    q_initial = fsolve(ik_residual, [0.5, 0.5], args=(x_start, y_start))
    robot.move_to(q_initial)
    q = q_initial.copy()

    robot.display(q)

    time.sleep(2)

    qdot = np.zeros(DOF)
    q_prev = q.copy() # Just as a seed for the solver

    for i in range(n_steps):
        t = i * DT
        t_log[i] = t

        if trajectory == "circle":
            x, y, xdot, ydot, xddot, yddot = circle_trajectory(t, r=r, cx=cx, cy=cy, omega=omega)
        elif trajectory == "lemniscate":
            x, y, xdot, ydot, xddot, yddot = lemniscate_trajectory(t, a=a, cx=cx, cy=cy, omega=omega)
        elif trajectory == "line":
            x, y, xdot, ydot, xddot, yddot = line_trajectory(t, lx0, ly0, lx1, ly1, omega=omega)

        # Holding a static pose
        # x, y, xdot, ydot, xddot, yddot = cx, cy, 0.0, 0.0, 0.0, 0.0

        # ---------------------------------------------
        # Computer Torque Controller Implementation
        # Why q_des in J and not the current q? Dafuq?
        # ---------------------------------------------
        q_des = fsolve(ik_residual, q_prev, args=(x, y))
        J = robot.ee_jacobian(q_des)
        J_inv = np.linalg.inv(J)

        qdot_des = J_inv @ np.array([xdot, ydot])
        qddot_des = J_inv @ np.array([xddot, yddot]) # TODO: Missing J dot thingy
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

        robot.display(q)

        q, qdot = robot.step_rk4(q, qdot, tau, DT)
        # q, qdot = robot.imperfect_dynamics_step(q, qdot, tau, DT)

    return {
        "t": t_log,
        "q": q_log,
        "qdot": qdot_log,
        "tau": tau_log,
        "dt": DT,
        "meta": {
            "trajectory": trajectory,
            "duration": duration,
            "r": r,
            "a": a,
            "cx": cx,
            "cy": cy,
            "omega": omega,
            "lx0": lx0,
            "ly0": ly0,
            "lx1": lx1,
            "ly1": ly1,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Collect cartesian-space trajectory data at 500 Hz (Pinocchio).")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--trajectory", choices=["circle", "lemniscate", "line"], default="circle")
    parser.add_argument("--cx", type=float, default=0.7)
    parser.add_argument("--cy", type=float, default=1.5)
    parser.add_argument("--omega", type=float, default=0.8)
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--r", type=float, default=0.4)
    parser.add_argument("--lx0", type=float, default=0.3, help="Line start x")
    parser.add_argument("--ly0", type=float, default=1.0, help="Line start y")
    parser.add_argument("--lx1", type=float, default=1.1, help="Line end x")
    parser.add_argument("--ly1", type=float, default=1.6, help="Line end y")
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
        lx0=args.lx0,
        ly0=args.ly0,
        lx1=args.lx1,
        ly1=args.ly1,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        t=log["t"],
        q=log["q"],
        qdot=log["qdot"],
        tau=log["tau"],
        dt=np.array(log["dt"]),
        meta=np.array(log["meta"], dtype=object),
    )
    print(f"Saved {len(log['t'])} samples at {1.0 / DT:.0f} Hz -> {args.output}")


if __name__ == "__main__":
    main()
