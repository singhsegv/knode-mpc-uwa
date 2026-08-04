import os
import sys
import time

import casadi as ca
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robots.simulator import PlanarArm

from trajectory_utils import (
    get_joint_space_trajectory_mpc,
    get_task_space_trajectory_mpc,
    inverse_kinematics,
    plot_task_space_tracking,
)

# from manipulator_mpc import ManipulatorControl
from fatrop_manipulator_mpc import ManipulatorControl


def get_plant_and_controller(dt=0.001, mpc_horizon=20, enable_knode=False):
    if not enable_knode:
        plant = PlanarArm(enable_viz=False, gravity=np.array([0.0, -9.81, 0.0]))
        mpc = ManipulatorControl(horizon=mpc_horizon, dt=dt)
    else:
        plant = PlanarArm(enable_viz=False)
        knode_path = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/runs/20260731_151818_mpc_dataset_mpc_basic_mpc-basic-la-5-step-0p01"
        mpc = ManipulatorControl(horizon=mpc_horizon, dt=dt, enable_knode=True, run_dir=knode_path)

    return plant, mpc


def run_closed_loop_mpc():
    # DT = 0.002 # 500hz
    DT = 0.01 # 100hz
    current_time = 0.0
    SIM_TIME = 15.0
    MPC_HORIZON = 50

    # plant, mpc = get_plant_and_controller(dt=DT, mpc_horizon=MPC_HORIZON, enable_knode=False)
    plant, mpc = get_plant_and_controller(dt=DT, mpc_horizon=MPC_HORIZON, enable_knode=True)
    
    q0, _ = inverse_kinematics(plant, plant.get_random_config(), 0.7, 0.5, 0.0, 0.0)
    plant.move_to(q0)
    x_current_state = np.array([q0[0], q0[1], 0.0, 0.0])

    history_x = [x_current_state]
    history_u = []
    u_prev = None 

    print("Starting closed-loop execution...")

    steps = int(SIM_TIME / DT)
    for step in range(steps):
        # X_ref = get_joint_space_trajectory_mpc(current_time, mpc_horizon=MPC_HORIZON, dt=DT)
        X_ref = get_task_space_trajectory_mpc(current_time, mpc_horizon=MPC_HORIZON, dt=DT, robot=plant, q_prev=x_current_state[:2])

        t_start = time.perf_counter()
        # TODO: For perfect nominal model, this should be the case
        U_optimal, X_predicted = mpc.solve(x_current_state, X_ref, u_guess=u_prev)
        solve_time = time.perf_counter() - t_start

        u_actual = U_optimal[:, 0]
        # TODO: check both, use_torchdiffeq true and false
        q_next, qdot_next = plant.step_rk4(x_current_state[:2], x_current_state[2:], u_actual, DT)
        # plant.display(q_next)

        x_current_state = np.concatenate([q_next.flatten(), qdot_next.flatten()])
        current_time += DT

        # TODO: Do we need the whole history for making next MPC steps faster?
        u_prev = np.zeros_like(U_optimal)
        u_prev[:, :-1] = U_optimal[:, 1:]
        u_prev[:, -1] = U_optimal[:, -1]

        history_x.append(x_current_state)
        history_u.append(u_actual)        

        if step % 50 == 0:
            print(f"Step {step}/{steps} | Time: {current_time:.2f}s | Solve Time: {solve_time*1000:.1f}ms")

    return np.array(history_x), np.array(history_u), DT


def ablation_dynamics():
    """
    expand: False
    """
    pass


def ablation_integrators():
    """
    expand: False
    integrators: manual/rk4/euler
    """
    pass


if __name__ == "__main__":
    history_x, history_u, dt = run_closed_loop_mpc()
    robot = PlanarArm()
    plot_task_space_tracking(history_x, robot, dt=dt, save_path="/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/mpc/perfect.png")
