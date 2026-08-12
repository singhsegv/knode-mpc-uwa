import os
import sys

import torch
import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.model_utils import load_state_dict_from_run
from casadi_knode_callback import ResidualNNCallback
from robot_descriptions.loaders.pinocchio import load_robot_description


def tensor_to_dm(t):
    """torch tensor -> CasADi DM."""
    return ca.DM(t.detach().cpu().numpy())


def extract_mlp_layers(state_dict, prefix="func.residual_dynamics_network."):
    """Extract (weight, bias) pairs from a Sequential MLP state dict."""
    layer_indices = sorted({
        int(key[len(prefix):].split(".")[0])
        for key in state_dict
        if key.startswith(prefix) and key.endswith(".weight")
    })
    layers = []
    for idx in layer_indices:
        layers.append((
            state_dict[f"{prefix}{idx}.weight"].detach().cpu().numpy(),
            state_dict[f"{prefix}{idx}.bias"].detach().cpu().numpy(),
        ))
    return layers


class ManipulatorControl:
    """MPC solver for n-DOF manipulators."""

    def __init__(self, horizon, dt, enable_knode=False, run_dir=None, epoch=-1, robot_config=None):
        self.robot_config = robot_config

        self.nq = robot_config["nq"]
        self.nv = robot_config["nv"]

        self.nx = 2 * self.nq
        self.nu = self.nv

        self.torque_limits = robot_config["torque_limits"]

        self.mpc_horizon = horizon
        self.dt = dt

        if enable_knode:
            if run_dir is None:
                raise ValueError("run_dir is required when enable_knode=True")

            self.run_dir = run_dir
            self.knode_epoch = epoch
            self.dynamics = self.get_knode()
        else:
            self.dynamics = self.get_nominal()

        self.opti = ca.Opti()
        self.X = self.opti.variable(self.nx, self.mpc_horizon + 1)
        self.U = self.opti.variable(self.nu, self.mpc_horizon)

        self.x0 = self.opti.parameter(self.nx, 1)
        self.X_ref = self.opti.parameter(self.nx, self.mpc_horizon + 1)

        self._build_optimization_problem()

    def _build_optimization_problem(self):
        cost = 0

        q_weights = np.full(self.nq, 100.0)
        qdot_weights = np.full(self.nv, 10.0)
        Q = np.diag(np.concatenate([q_weights, qdot_weights]))
        R = np.diag(np.full(self.nu, 0.01))

        for k in range(self.mpc_horizon):
            self.opti.subject_to(self.X[:, k + 1] == self.dynamics(self.X[:, k], self.U[:, k]))

            tracking_error = self.X[:, k] - self.X_ref[:, k]
            cost += ca.mtimes([tracking_error.T, Q, tracking_error])
            cost += ca.mtimes([self.U[:, k].T, R, self.U[:, k]])

        terminal_error = self.X[:, self.mpc_horizon] - self.X_ref[:, self.mpc_horizon]
        cost += ca.mtimes([terminal_error.T, Q * 25, terminal_error])

        self.opti.minimize(cost)
        self.opti.subject_to(self.X[:, 0] == self.x0)

        for j in range(self.nu):
            self.opti.subject_to(self.opti.bounded(-self.torque_limits[j], self.U[j, :], self.torque_limits[j]))

        p_opts = {"expand": False, "print_time": False}
        s_opts = {
            "max_iter": 500,
            "print_level": 0,
            "print_user_options": "no",
            "print_timing_statistics": "no",
            "tol": 1e-4,
            "hessian_approximation": "limited-memory",
        }
        self.opti.solver("ipopt", p_opts, s_opts)

    def solve(self, current_state, reference_horizon, u_guess=None):
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.X_ref, reference_horizon)

        if u_guess is not None:
            self.opti.set_initial(self.U, u_guess)

        try:
            sol = self.opti.solve()
            return sol.value(self.U), sol.value(self.X)
        except Exception:
            print("Solver failed to converge! Returning suboptimal/last known torques.")
            return self.opti.debug.value(self.U), self.opti.debug.value(self.X)

    def get_casadi_pinn_model(self, gravity=None):
        if gravity is None:
            gravity = self.robot_config["model_gravity"]

        # Uses robot_descriptions pip package now for easy peazy stuff
        robot_urdf_model_name = self.robot_config["description"]
        pin_robot = load_robot_description(robot_urdf_model_name)
        model = pin_robot.model
        model.gravity.linear = gravity

        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        
        return model, cmodel, cdata

    def get_nominal(self):
        _, cmodel, cdata = self.get_casadi_pinn_model()

        nq, nv = cmodel.nq, cmodel.nv
        q = ca.SX.sym("q", nq)
        v = ca.SX.sym("v", nv)
        tau = ca.SX.sym("tau", nv)

        qdd = cpin.aba(cmodel, cdata, q, v, tau)
        x = ca.vertcat(q, v)
        u = tau
        xdot = ca.vertcat(v, qdd)

        x_next = x + xdot * self.dt
        return ca.Function("F", [x, u], [x_next])

    def get_knode(self):
        _, cmodel, cdata = self.get_casadi_pinn_model()
        nq, nv = cmodel.nq, cmodel.nv

        q_sx = ca.SX.sym("q", nq)
        v_sx = ca.SX.sym("v", nv)
        tau_sx = ca.SX.sym("tau", nv)

        x_sx = ca.vertcat(q_sx, v_sx)
        u_sx = tau_sx

        nominal_qdd = cpin.aba(cmodel, cdata, q_sx, v_sx, tau_sx)
        nominal_xdot = ca.vertcat(v_sx, nominal_qdd)
        aba_f = ca.Function("aba_f", [x_sx, u_sx], [nominal_xdot])

        state_dict, _ = load_state_dict_from_run(self.run_dir, epoch=self.knode_epoch)
        nn_forward, nn_jacobian = self._load_residual_nn_numpy(state_dict)
        self.nn_cb = ResidualNNCallback(nn_forward, nn_jacobian, nq=nq)

        x = ca.MX.sym("x", self.nx)
        u = ca.MX.sym("u", self.nu)
        q = x[:nq]

        nominal_xdot_mx = aba_f(x, u)
        residual_qdd_mx = self.nn_cb(q)
        xdot = nominal_xdot_mx + ca.vertcat(ca.MX.zeros(nv), residual_qdd_mx)

        x_next = x + xdot * self.dt
        return ca.Function("F", [x, u], [x_next])

    def _load_residual_nn_numpy(self, state_dict):
        layers = extract_mlp_layers(state_dict)

        def forward(q):
            q = np.asarray(q, dtype=np.float64).reshape(self.nq)
            h = q
            for W, b in layers[:-1]:
                h = np.tanh(W @ h + b)
            W, b = layers[-1]
            return W @ h + b

        def jacobian(q):
            q_t = torch.tensor(q, dtype=torch.float64, requires_grad=True).reshape(self.nq)

            def torch_forward(qq):
                h = qq.reshape(self.nq)
                for W, b in layers[:-1]:
                    Wt = torch.as_tensor(W, dtype=torch.float64)
                    bt = torch.as_tensor(b, dtype=torch.float64)
                    h = torch.tanh(Wt @ h + bt)
                W, b = layers[-1]
                Wt = torch.as_tensor(W, dtype=torch.float64)
                bt = torch.as_tensor(b, dtype=torch.float64)
                return Wt @ h + bt

            J = torch.autograd.functional.jacobian(torch_forward, q_t)
            return J.detach().numpy()

        return forward, jacobian
