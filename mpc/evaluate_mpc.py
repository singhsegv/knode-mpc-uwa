import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robots.simulator import Manipulator, load_robot_config

from trajectory_utils import (
    get_task_space_trajectory_mpc,
    inverse_kinematics,
    plot_task_space_tracking,
)
# from manipulator_mpc import ManipulatorControl
from fatrop_manipulator_mpc import FatropManipulatorControl as ManipulatorControl


def get_plant_and_controller(dt=0.001, mpc_horizon=20, enable_knode=False, knode_path=None, robot_config=None):
    plant = Manipulator(robot_config=robot_config, enable_viz=False)
    mpc = ManipulatorControl(
        horizon=mpc_horizon, dt=dt, enable_knode=enable_knode,
        run_dir=knode_path, robot_config=robot_config,
    )
    return plant, mpc


def run_closed_loop_mpc(knode_path=None, robot_config=None):
    DT = 0.01
    current_time = 0.0
    SIM_TIME = 10.0
    MPC_HORIZON = 25

    plant, mpc = get_plant_and_controller(
        dt=DT, mpc_horizon=MPC_HORIZON, enable_knode=False,
        knode_path=knode_path, robot_config=robot_config,
    )

    nq = plant.nq
    q0, _ = inverse_kinematics(
        plant, plant.get_default_config(),
        np.array([0.35, 0.15, 0.35]), np.zeros(3),
    )
    plant.move_to(q0)
    x_current_state = np.concatenate([q0, np.zeros(nq)])

    history_x = [x_current_state]
    history_u = []
    u_prev = None

    print("Starting closed-loop execution...")

    steps = int(SIM_TIME / DT)
    for step in range(steps):
        X_ref = get_task_space_trajectory_mpc(
            current_time, mpc_horizon=MPC_HORIZON, dt=DT,
            robot=plant, q_prev=x_current_state[:nq],
        )

        t_start = time.perf_counter()
        U_optimal, X_predicted = mpc.solve(x_current_state, X_ref, u_guess=u_prev)
        solve_time = time.perf_counter() - t_start

        u_actual = U_optimal[:, 0]
        q_next, qdot_next = plant.step_rk4(x_current_state[:nq], x_current_state[nq:], u_actual, DT)

        x_current_state = np.concatenate([q_next.flatten(), qdot_next.flatten()])
        current_time += DT

        u_prev = np.zeros_like(U_optimal)
        u_prev[:, :-1] = U_optimal[:, 1:]
        u_prev[:, -1] = U_optimal[:, -1]

        history_x.append(x_current_state)
        history_u.append(u_actual)

        if step % 100 == 0:
            print(f"Step {step}/{steps} | Time: {current_time:.2f}s | Solve Time: {solve_time*1000:.1f}ms")

    return np.array(history_x), np.array(history_u), DT


if __name__ == "__main__":
    robot_config, _ = load_robot_config("ur5-mpc")
    history_x, history_u, dt = run_closed_loop_mpc(knode_path=None, robot_config=robot_config)
    plant = Manipulator(robot_config=robot_config)
    image_save_path = os.path.join(SCRIPT_DIR, "perfect-ur5.png")
    plot_task_space_tracking(history_x, plant, dt=dt, save_path=image_save_path)
