import casadi as ca
import numpy as np
from casadi_models import get_nominal


DT = 0.002
MPC_HORIZON = 50


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


def single_shot_open_loop_mpc():
    """
    `compare_casadi_nominal()`
    SINGLE SHOT MPC SOLVER

    Comparison: Casadi manual model and original generated data
    Dynamics: Perfect and Imperfect

    -------
    return: casadi.OptiSol
    """
    # ----------------------
    # Load test data
    # ----------------------
    x0 = np.array([0.08211392, 1.89652581, 0., 0.]) # TODO: Get from test_data
    # NOTE: Reference is currently taken from 5th second of a circular trajectory
    xref = np.array([0.22925932, 1.92863033, -0.01787316, 0.05872287]) # TODO: How to change this to a continuous trajectory?

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
    X = opti.variable(nx, N+1) # [q1, q2, qdot1, qdot2 ]
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

    return sol.value(X), sol.value(U)


def closed_loop_real_mpc(x0, xref_fn, n_steps, mpc_horizon, dt, perfect=True):
    """
    TODO: Dafuq is xref_fn
    """
    robot = PlanarArm()
    Dynamics = get_nominal(perfect_dynamics=True)

    q, qdot = x0[:2], x0[2:]
    log = []

    # Trajectory total steps
    for step in range(n_steps):
        k0 = step
        # TODO: Is this for trajecotry tracking?
        X_ref = xref_fn(k0, mpc_horizon + 1) # (4, N+1) reference window

        opti = ca.Opti()
        X = opti.variable(4, mpc_horizon + 1)
        U = opti.variable(2, mpc_horizon)
        cost = 0
        for k in range(mpc_horizon):
            opti.subject_to(X[:, k+1] == Dynamics(X[:, k], U[:, k]))
            cost += ca.sumsqr(X[:, k] - X_ref[:, k]) + 1e-3 * ca.sumsqr(U[:, k])
        
        cost += 10 * ca.sumsqr(X[:, mpc_horizon] - X_ref[:, mpc_horizon])
        opti.minimize(cost)
        opti.subjecto_to(X[:, 0] == np.concatenate([q, qdot]))
        opti.subject_to(opti.bounded(-20, U, 20))
        opti.solver("ipopt", {"print_time": False}, {"print_level": 0})
        sol = opti.solve()

        tau = 



if __name__ == "__main__":
    # verify_pinn_and_casadi(perfect_dynamics=True)
    # verify_pinn_and_casadi(perfect_dynamics=False)

    single_shot_open_loop_mpc()
