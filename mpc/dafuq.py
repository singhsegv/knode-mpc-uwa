"""
Later Try
# Automatic
# import l4casadi as l4c
# l4c_model = l4c.L4CasADi(nn_model, device='cpu')
# state_input = ca.vertcat(q, qdot)
# residual_model = l4c_model(state_input) # Da Shizz
"""
import os
import sys
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robots.simulator import PlanarArm

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin


def tensor_to_dm(t: torch.tensor) -> ca.DM:
    """torch tensor to numpy to dm for CasADi"""
    return ca.DM(t.detach().cpu().numpy())


def open_loop_run():
    """ Pinocchio vs Pinocchio.Casadi """
    for _ in range(100000):
        # TODO: Test me open loop over multiple steps
        q_test = np.array([0.3, -0.5])
        v_test = np.array([0.1, -0.2])
        x_test = np.concatenate([q_test, v_test])
        tau_test = np.array([1.0, 0.5])

        pin_x_next = robot.step_rk4(q_test, v_test, tau_test, 0.002)
        pin_x_next = np.concatenate([pin_x_next[0], pin_x_next[1]])
        cas_x_next = np.array(Dynamics(x_test, tau_test)).ravel()
        print("----------------------------")
        print(np.max(np.abs(pin_x_next - cas_x_next)))
        print("----------------------------")


if __name__ == "__main__":
    urdf_path = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/robots/planar-2R.urdf"
    checkpoint_path = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/runs/20260717_171819_evaluation1_basic1_comparison-la-100/checkpoints/neural_ode_epoch_2100.pt"
    
    # Step 1
    Dynamics = get_dynamics_model(urdf_path, checkpoint_path)
    robot = PlanarArm()
    
    # Step 2
    # State in Opti should be 4D [q, qdot] (not the training 6D z). 
    # Torques are decision variables U, held ZOH over each MPC step — same as training’s constant-u ODE.
    opti = ca.Opti()
    X = opti.variable(4, N+1)
    U = opti.variable(2, N)

    opti.subject_to(X[:, k+1] == Dynamics(X[:, k], U[:, k]))
    
    # Step 3: Cost / constraints for the arm -> ‖q − q_ref‖² + ‖q̇ − q̇_ref‖² + R‖τ‖²
    # TODO: Joint space or task space??


    