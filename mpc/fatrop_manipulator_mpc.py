"""
Fatrop-based MPC for n-DOF manipulators.

Adapted from the planar main-fatrop branch: stage-wise Opti variables and the
fatrop solver plugin, with n-DOF Pinocchio dynamics and K-NODE residual support.
"""
import os
import sys

import torch
import numpy as np
import casadi as ca
import pinocchio.casadi as cpin

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.model_utils import load_state_dict_from_run
from casadi_knode_callback import ResidualNNCallback
from robot_descriptions.loaders.pinocchio import load_robot_description


def tensor_to_dm(t):
    """Array-like -> CasADi DM."""
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return ca.DM(t)


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


class FatropManipulatorControl:
    """Fatrop MPC solver for n-DOF manipulators."""

    def __init__(self, horizon, dt, enable_knode=False, run_dir=None, epoch=-1, robot_config=None):
        if robot_config is None:
            raise ValueError("robot_config is required")

        self.robot_config = robot_config
        self.nq = robot_config["nq"]
        self.nv = robot_config["nv"]
        self.nx = 2 * self.nq
        self.nu = self.nv
        self.torque_limits = robot_config["torque_limits"]
        self.mpc_horizon = horizon
        self.dt = dt
        self.enable_knode = enable_knode

        if enable_knode:
            if run_dir is None:
                raise ValueError("run_dir is required when enable_knode=True")
            self.run_dir = run_dir
            self.knode_epoch = epoch
            self.dynamics = self.get_knode()
        else:
            self.dynamics = self.get_nominal()

        self.opti = ca.Opti()
        self.X = []
        self.U = []
        for _ in range(self.mpc_horizon):
            self.X.append(self.opti.variable(self.nx, 1))
            self.U.append(self.opti.variable(self.nu, 1))

        self.X.append(self.opti.variable(self.nx, 1))

        self.x0 = self.opti.parameter(self.nx, 1)
        self.X_ref = self.opti.parameter(self.nx, self.mpc_horizon + 1)

        self._build_optimization_problem()

    def _build_optimization_problem(self):
        cost = 0

        q_weights = np.full(self.nq, 100.0)
        qdot_weights = np.full(self.nv, 100.0)
        Q = np.diag(np.concatenate([q_weights, qdot_weights]))
        R = np.diag(np.full(self.nu, 0.01))

        self.opti.subject_to(self.X[0] == self.x0)

        for k in range(self.mpc_horizon):
            tracking_error = self.X[k] - self.X_ref[:, k:k + 1]
            cost += ca.mtimes([tracking_error.T, Q, tracking_error])
            cost += ca.mtimes([self.U[k].T, R, self.U[k]])

            self.opti.subject_to(self.X[k + 1] == self.dynamics(self.X[k], self.U[k]))
            for j in range(self.nu):
                self.opti.subject_to(self.opti.bounded(-self.torque_limits[j], self.U[k][j], self.torque_limits[j]))

        terminal_error = self.X[self.mpc_horizon] - self.X_ref[:, self.mpc_horizon]
        cost += ca.mtimes([terminal_error.T, Q * 25, terminal_error])

        self.opti.minimize(cost)

        plugin_opts = {
            "print_time": False,
            "verbose": False,
            "debug": False,
            "jit": False,
            "fatrop": {
                "print_level": 0,
            },
        }
        solver_opts = {"print_level": 0}
        self.opti.solver("fatrop", plugin_opts, solver_opts)

    def solve(self, current_state, reference_horizon, u_guess=None):
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.X_ref, reference_horizon)

        if u_guess is not None:
            for k in range(self.mpc_horizon):
                self.opti.set_initial(self.U[k], u_guess[:, k:k + 1])

        try:
            sol = self.opti.solve()
            U_optimal = np.column_stack([sol.value(u).flatten() for u in self.U])
            X_predicted = np.column_stack([sol.value(x).flatten() for x in self.X])
            return U_optimal, X_predicted
        except Exception as exc:
            print(f"Fatrop solver failed: {exc}")
            raise

    def get_casadi_pinn_model(self, gravity=None):
        if gravity is None:
            gravity = self.robot_config["model_gravity"]

        pin_robot = load_robot_description(self.robot_config["description"])
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
        xdot = ca.vertcat(v, qdd)
        x_next = x + xdot * self.dt
        return ca.Function("F", [x, tau], [x_next])

    def get_knode(self):
        _, cmodel, cdata = self.get_casadi_pinn_model()
        nq, nv = cmodel.nq, cmodel.nv

        q_sx = ca.SX.sym("q", nq)
        v_sx = ca.SX.sym("v", nv)
        tau_sx = ca.SX.sym("tau", nv)
        x_sx = ca.vertcat(q_sx, v_sx)

        nominal_qdd = cpin.aba(cmodel, cdata, q_sx, v_sx, tau_sx)
        nominal_xdot = ca.vertcat(v_sx, nominal_qdd)
        aba_f = ca.Function("aba_f", [x_sx, tau_sx], [nominal_xdot])

        state_dict, _ = load_state_dict_from_run(self.run_dir, epoch=self.knode_epoch)
        nn_forward, nn_jacobian, nn_jacobian_forward = self._load_residual_nn_numpy(state_dict)
        self.nn_cb = ResidualNNCallback(
            nn_forward, nn_jacobian, nq=nq, jacobian_forward_fn=nn_jacobian_forward,
        )

        x = ca.MX.sym("x", self.nx)
        u = ca.MX.sym("u", self.nu)
        q = x[:nq]

        nominal_xdot_mx = aba_f(x, u)
        residual_qdd_mx = self.nn_cb(q)
        xdot = nominal_xdot_mx + ca.vertcat(ca.MX.zeros(nv), residual_qdd_mx)
        x_next = x + xdot * self.dt
        return ca.Function("F", [x, u], [x_next])

    def _load_residual_nn_numpy(self, state_dict):
        """Returns forward(q), jacobian(q), and jacobian_forward(q, q_dot)."""
        layers = extract_mlp_layers(state_dict)
        nq = self.nq

        def forward(q):
            q = np.asarray(q, dtype=np.float64).reshape(nq)
            h = q
            for W, b in layers[:-1]:
                h = np.tanh(W @ h + b)
            W, b = layers[-1]
            return W @ h + b

        def torch_forward(qq):
            h = qq.reshape(nq)
            for W, b in layers[:-1]:
                Wt = torch.as_tensor(W, dtype=torch.float64)
                bt = torch.as_tensor(b, dtype=torch.float64)
                h = torch.tanh(Wt @ h + bt)
            W, b = layers[-1]
            Wt = torch.as_tensor(W, dtype=torch.float64)
            bt = torch.as_tensor(b, dtype=torch.float64)
            return Wt @ h + bt

        def jacobian(q):
            q_t = torch.tensor(q, dtype=torch.float64, requires_grad=True).reshape(nq)
            J = torch.autograd.functional.jacobian(torch_forward, q_t)
            return J.detach().numpy()

        def jacobian_forward(q, q_dot):
            q = np.asarray(q, dtype=np.float64).reshape(nq)
            q_dot = np.asarray(q_dot, dtype=np.float64).reshape(nq)
            q_t = torch.tensor(q, dtype=torch.float64, requires_grad=True)
            q_dot_t = torch.tensor(q_dot, dtype=torch.float64)

            J_dot = np.zeros((nq, nq))
            for i in range(nq):
                for j in range(nq):
                    def Jij(qq):
                        return torch.autograd.functional.jacobian(torch_forward, qq)[i, j]

                    _, dJij = torch.autograd.functional.jvp(Jij, q_t, q_dot_t)
                    J_dot[i, j] = dJij.item()
            return J_dot

        return forward, jacobian, jacobian_forward
