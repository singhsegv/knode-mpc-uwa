"""
Later Try
# Automatic
# import l4casadi as l4c
# l4c_model = l4c.L4CasADi(nn_model, device='cpu')
# state_input = ca.vertcat(q, qdot)
# residual_model = l4c_model(state_input) # Da Shizz
"""
import torch
import numpy as np
from robot import PlanarArm

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin


def tensor_to_dm(t: torch.tensor) -> ca.DM:
    """torch tensor to numpy to dm for CasADi"""
    return ca.DM(t.detach().cpu().numpy())


def get_dynamics_model(urdf_path, model_path):
    """
    TODO: Add same knobs as the KNODE nominal model: gravity / mass / etc
    Returns symbolic dynamics in CasADi thingy

    TODO: Add configuration settings like NN layers, activations, etc used in KNODE
    TODO: Weight normalization stuff

    TODO: dt of 0.002 might just fail for MPC??

    nn.Linear(6, 128) -> nn.Tanh() -> nn.Linear(128, 128) -> nn.Tanh() -> nn.Linear(128, 4)
    """
    # ------------------------------------------------------
    # Analytical Physics: Move each torch fully connected layer to a casadi layer
    # ------------------------------------------------------
    model = pin.buildModelFromUrdf(urdf_path)
    model.gravity.linear = np.array([0.0, -9.81, 0.0])
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    # TODO: Check for SX vs MX here -> Dafuq are the side effects
    nq, nv = cmodel.nq, cmodel.nv
    q = ca.SX.sym("q", nq)
    v = ca.SX.sym("v", nv) # qdot
    tau = ca.SX.sym("tau", nv)

    # Perfect Model
    nominal_qdd = cpin.aba(cmodel, cdata, q, v, tau)

    # Imperfect Model
    # g = cpin.computeGeneralizedGravity(cmodel, cdata, q)
    # nominal_qdd = cpin.aba(cmodel, cdata, q, v, tau + g)

    # Moving from q to x naming convention for clear state derivative shizz
    x = ca.vertcat(q, v) # state for MPC: [q, qdot], size 4
    u = tau
    nominal_xdot = ca.vertcat(v, nominal_qdd)

    f = ca.Function("f", [x, u], [nominal_xdot])

    # ##################################################
    # Testing Stuff
    # aba_fn = ca.Function("aba", [q, v, tau], [nominal_qdd])

    # robot = PlanarArm()
    # q_test = np.array([0.3, -0.5])
    # v_test = np.array([0.1, -0.2])
    # tau_test = np.array([1.0, 0.5])

    # pin_qdd = robot.forward_dynamics(q_test, v_test, tau_test)
    # cas_qdd = np.array(aba_fn(q_test, v_test, tau_test)).ravel()
    # print("----------------------------")
    # print(np.max(np.abs(pin_qdd - cas_qdd)))
    # print("----------------------------")
    # ##################################################

    # ------------------------------------------------------
    # Learned Physics: Pinocchio Casadi for Articulated Body Algo shizz -> TODO: Manual like paper OR L4CasADi
    # ------------------------------------------------------
    # state_dict = torch.load(model_path)
    
    # # TODO: SKETCHY AF -> Should this be stored/loaded instead of being calculated on-the-fly??
    # mean = tensor_to_dm(state_dict["func.residual_input_mean"])
    # std  = tensor_to_dm(state_dict["func.residual_input_std"])
    
    # # W0, b0, W1, b1, W2, b2
    # params = [
    #     tensor_to_dm(state_dict[f"func.residual_dynamics_network.{i}.{attr}"])
    #     for i in (0, 2, 4)
    #     for attr in ("weight", "bias")
    # ]
    
    # z = ca.vertcat(q, v, tau)
    # z_n = (z - mean) / std

    # # TODO: Make sure this fucker matches
    # h = ca.tanh(ca.mtimes(params[0], z_n) + params[1])
    # h = ca.tanh(ca.mtimes(params[2], h) + params[3])
    # residual_xdot = ca.mtimes(params[4], h) + params[5] # (4,) -> [δq̇, δq̈]

    # # f = ca.Function("f", [x, u], [residual_xdot])
    
    # combined_xdot = nominal_xdot + residual_xdot
    # f = ca.Function("f", [x, u], [combined_xdot])

    # # Continuous shizz to Discrete shizz mapping
    intg = ca.integrator(
        "intg", "rk",
        {
            "x": x, 
            "p": u, 
            "ode": f(x, u)
        },
        dict(tf=0.002, simplify=True, number_of_finite_elements=4),
    )

    return ca.Function("F", [x, u], [intg(x0=x, p=u)["xf"]])


if __name__ == "__main__":
    urdf_path = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/planar-2R.urdf"
    checkpoint_path = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/archive-runs-data/look20-epoch2k/checkpoints/neural_ode_epoch_2000.pt"
    
    # Step 1
    Dynamics = get_dynamics_model(urdf_path, checkpoint_path)
    robot = PlanarArm()

    while True:
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
    
    import pdb; pdb.set_trace()

    # Step 2
    # State in Opti should be 4D [q, qdot] (not the training 6D z). 
    # Torques are decision variables U, held ZOH over each MPC step — same as training’s constant-u ODE.
    opti = ca.Opti()
    X = opti.variable(4, N+1)
    U = opti.variable(2, N)

    opti.subject_to(X[:, k+1] == Dynamics(X[:, k], U[:, k]))
    
    # Step 3: Cost / constraints for the arm -> ‖q − q_ref‖² + ‖q̇ − q̇_ref‖² + R‖τ‖²
    # TODO: Joint space or task space??


    