import torch
import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin


URDF_PATH = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/robots/planar-2R.urdf"


def get_casadi_pinn_model():
    model = pin.buildModelFromUrdf(URDF_PATH)
    model.gravity.linear = np.array([0.0, -9.81, 0.0])
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    return model, cmodel, cdata


def get_dynamics_model():
    """
    NOTE: CasADi dynamics model

    TODO: Add same knobs as the KNODE nominal model: gravity / mass / etc
    Returns symbolic dynamics in CasADi thingy

    TODO: Add configuration settings like NN layers, activations, etc used in KNODE
    TODO: Weight normalization stuff

    TODO: dt of 0.002 might just fail for MPC??

    nn.Linear(6, 128) -> nn.Tanh() -> nn.Linear(128, 128) -> nn.Tanh() -> nn.Linear(128, 4)
    """
    _, cmodel, cdata = get_casadi_pinn_model() 

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
    
    # combined_xdot = nominal_xdot + residual_xdot
    # f = ca.Function("f", [x, u], [combined_xdot])

    # # Continuous shizz to Discrete shizz mapping
    intg = ca.integrator(
        "intg", "rk",
        { "x": x, "p": u, "ode": f(x, u)},
        dict(tf=0.002, simplify=True, number_of_finite_elements=4),
    )

    return ca.Function("F", [x, u], [intg(x0=x, p=u)["xf"]])


def get_nominal(perfect_dynamics=False):
    """
    Comparison: Casadi manual model and original generated data
    Dynamics: Perfect and Imperfect
    """
    _, cmodel, cdata = get_casadi_pinn_model()

    # TODO: MX or SX ??
    nq, nv = cmodel.nq, cmodel.nv
    q = ca.SX.sym("q", nq)
    v = ca.SX.sym("v", nv) # qdot
    tau = ca.SX.sym("tau", nv)
    
    if perfect_dynamics:
        qdd = cpin.aba(cmodel, cdata, q, v, tau)
    else:
        g = cpin.computeGeneralizedGravity(cmodel, cdata, q)
        qdd = cpin.aba(cmodel, cdata, q, v, tau + g)

    # Moving from q to x naming convention for clear state derivative shizz
    x = ca.vertcat(q, v) # state for MPC: [q, qdot], size 4
    u = tau
    xdot = ca.vertcat(v, qdd)

    # TODO: Check if this should be different?
    MPC_DT = 0.002

    # TODO: Check us -> 'simplify' and 'number_of_finite_elements'
    f = ca.Function("f", [x, u], [xdot])
    intg = ca.integrator(
        "intg", "rk",
        { "x": x, "p": u, "ode": f(x, u)},
        dict(tf=MPC_DT, simplify=True, number_of_finite_elements=4),
    )

    return ca.Function("F", [x, u], [intg(x0=x, p=u)["xf"]])


def get_knode(model_path, nominal_only=False, perfect_dynamics=False):
    """
    Comparison: KNODE Casadi implementation and original generated data
    Dynamics: Perfect and Imperfect (NN + Residual)
    
    TODO: Manual like paper OR L4CasADi
    """
    _, cmodel, cdata = get_casadi_pinn_model(model_path) 

    nq, nv = cmodel.nq, cmodel.nv
    q = ca.SX.sym("q", nq)
    v = ca.SX.sym("v", nv) # qdot
    tau = ca.SX.sym("tau", nv)

    # Perfect Model
    nominal_qdd = cpin.aba(cmodel, cdata, q, v, tau)
    
    state_dict = torch.load(model_path)
    
    # TODO: SKETCHY AF -> Should this be stored/loaded instead of being calculated on-the-fly??
    mean = tensor_to_dm(state_dict["func.residual_input_mean"])
    std  = tensor_to_dm(state_dict["func.residual_input_std"])
    
    # W0, b0, W1, b1, W2, b2
    params = [
        tensor_to_dm(state_dict[f"func.residual_dynamics_network.{i}.{attr}"])
        for i in (0, 2, 4)
        for attr in ("weight", "bias")
    ]

    z = ca.vertcat(q, v, tau)
    z_n = (z - mean) / std

    # TODO: Make sure this fucker matches
    h = ca.tanh(ca.mtimes(params[0], z_n) + params[1])
    h = ca.tanh(ca.mtimes(params[2], h) + params[3])
    residual_xdot = ca.mtimes(params[4], h) + params[5] # (4,) -> [δq̇, δq̈]

    f = ca.Function("f", [x, u], [residual_xdot])