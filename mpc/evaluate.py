import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robots.simulator import PlanarArm
from mpc import single_shot_open_loop_mpc


DT = 0.002


# def later():
#     from utils.data_utils import load_evaluation_data

#     traj = load_evaluation_data("data/benchmark_basic")[0]  # e.g. 00_circle
#     controls = traj["controls"]  # (N, 7): [t, q, qdot, tau]

#     x0 = controls[0, 1:5]                          # [q, qdot]
#     xref_traj = controls[:N+1, 1:5].T              # (4, N+1) time-varying joint ref
#     des_ee = traj["des_ee"]                        # (N, 6): task-space ref

#     # Tracking
#     cost += ca.sumsqr(X[:, k] - xref_traj[:, k])   # joint-space tracking



def test_open_loop_mpc():
    X_sol, U_sol = single_shot_open_loop_mpc()

    q_mpc, qdot_mpc = X_sol[:2, :-1], X_sol[2:, :-1]

    robot = PlanarArm()
    q0, qdot0 = X_sol[:2, 0], X_sol[2:, 0]
    q_log, qdot_log = robot.simulate_rollout(q0, qdot0, U_sol.T, DT, use_torchdiffeq=False)
    q_log, qdot_log = q_log.T, qdot_log.T

    q_err = np.linalg.norm(q_mpc - q_log)
    qdot_err = np.linalg.norm(qdot_mpc - qdot_log)

    print(f"Error q: {q_err}")
    print(f"Error qdot: {qdot_err}")


def compare_casadi_knode():
    """
    Comparison: KNODE Casadi implementation and original generated data
    Dynamics: Perfect and Imperfect
    """
    pass


if __name__ == "__main__":
    test_open_loop_mpc()
