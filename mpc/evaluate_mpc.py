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
from manipulator_mpc import ManipulatorControl


def verify_pinn_and_casadi(robot, perfect_dynamics):
    """
    Comparison: pin and cpin
    Goal: Difference in pin and cpin ABA() should be zero
    """
    q = np.array([0.3, -0.5])
    v = np.array([0.1, -0.2])
    x = np.concatenate([q, v])
    tau = np.array([1.0, 0.5]) 

    if perfect_dynamics:
        # This is not torchdiffeq thingy, this is pinocchio
        pin_next = np.concatenate(robot.step_rk4(q, v, tau, DT))
    else:
        pin_next = np.concatenate(robot.imperfect_dynamics_step(q, v, tau, DT))

    Dynamics = get_nominal(perfect_dynamics=perfect_dynamics)
    cas_next = np.array(Dynamics(x, tau)).ravel()
    
    # pin_qdd = robot.forward_dynamics(q_test, v_test, tau_test)
    # aba_fn = ca.Function("aba", [q, v, tau], [nominal_qdd])
    # cas_qdd = np.array(aba_fn(q_test, v_test, tau_test)).ravel()
    
    print("----------------------------")
    print(f"Perfect Dynamics: {perfect_dynamics}")
    print("max |pin - cas|:", np.max(np.abs(pin_next - cas_next)))
    print("----------------------------")


def test_single_target_mpc():
    """
    Runs the MPC only once for the given target with a horizon of MPC_HORIZON
    """
    DT = 0.002
    MPC_HORIZON = 50
    
    x0 = np.array([0.08211392, 1.89652581, 0., 0.])
    xref = np.array([0.22925932, 1.92863033, -0.01787316, 0.05872287])

    # ----------------------
    # Dynamics
    # ----------------------
    Dynamics = get_nominal(perfect_dynamics=True)

    # -------------------------------------------------------
    # Forming the Optimization Problem: Cost and Constraints
    # NOTE: Row columns thingy, each row has all the values for a variable
    # -------------------------------------------------------
    N = MPC_HORIZON
    nx, nu = 4, 2

    opti = ca.Opti()
    X = opti.variable(nx, N+1) # [q1, q2, qdot1, qdot2]
    U = opti.variable(nu, N) # [torque]

    cost = 0
    for k in range(N):
        opti.subject_to(X[:, k+1] == Dynamics(X[:, k], U[:, k]))
        cost += ca.sumsqr(X[:, k] - xref) + 1e-3 * ca.sumsqr(U[:, k]) # intermediate cost thingy

    cost += 10 * ca.sumsqr(X[:, N] - xref) # final cost thingy

    # --------------------------------------------------
    # Solving the Optimization Problem
    # --------------------------------------------------
    opti.minimize(cost)
    opti.subject_to(X[:, 0] == x0) # TODO: Setting up the IVP??
    opti.subject_to(opti.bounded(-20, U, 20)) # TODO: Check with my URDF shit
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0})
    sol = opti.solve()

    # print(f"Sol: {sol}")
    # print(f"X: {sol.value(X)}")
    # print(f"U: {sol.value(U)}")

    X_sol, U_sol = sol.value(X), sol.value(U)

    q_mpc, qdot_mpc = X_sol[:2, :-1], X_sol[2:, :-1]

    robot = PlanarArm()
    q0, qdot0 = X_sol[:2, 0], X_sol[2:, 0]
    q_log, qdot_log = robot.simulate_rollout(q0, qdot0, U_sol.T, DT, use_torchdiffeq=False)
    q_log, qdot_log = q_log.T, qdot_log.T

    q_err = np.linalg.norm(q_mpc - q_log)
    qdot_err = np.linalg.norm(qdot_mpc - qdot_log)

    print(f"Error q: {q_err}")
    print(f"Error qdot: {qdot_err}")


def get_plant_and_controller(dt=0.001, mpc_horizon=20, enable_knode=False):
    if not enable_knode:
        plant = PlanarArm(enable_viz=False)
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
    SIM_TIME = 10.0
    MPC_HORIZON = 10

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

        if step % 100 == 0:
            print(f"Step {step}/{steps} | Time: {current_time:.2f}s | Solve Time: {solve_time*1000:.1f}ms")

    return np.array(history_x), np.array(history_u), DT


if __name__ == "__main__":
    history_x, history_u, dt = run_closed_loop_mpc()
    robot = PlanarArm()
    plot_task_space_tracking(history_x, robot, dt=dt, save_path="/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/mpc/perfect.png")
