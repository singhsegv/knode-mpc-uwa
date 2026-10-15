import os
import sys

import torch
import numpy as np
import casadi as ca

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.model_utils import load_state_dict_from_run
from casadi_knode_callback import ResidualNNCallback
from casadi_dynamics import (
    build_mujoco_continuous_dynamics,
    build_nominal_euler_dynamics,
    build_pinocchio_continuous_dynamics,
    get_pinocchio_casadi_triplet,
    mpc_dynamics_backend,
)


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

        q_weights = np.asarray(
            self.robot_config.get("mpc_Q_q", np.full(self.nq, 100.0)), dtype=float
        )
        qdot_weights = np.asarray(
            self.robot_config.get("mpc_Q_qd", np.full(self.nv, 100.0)), dtype=float
        )
        r_weights = np.asarray(
            self.robot_config.get("mpc_R", np.full(self.nu, 0.01)), dtype=float
        )
        Q = np.diag(np.concatenate([q_weights, qdot_weights]))
        R = np.diag(r_weights)
        Q_terminal = Q * 10

        for k in range(self.mpc_horizon):
            self.opti.subject_to(self.X[:, k + 1] == self.dynamics(self.X[:, k], self.U[:, k]))

            tracking_error = self.X[:, k] - self.X_ref[:, k]
            cost += ca.mtimes([tracking_error.T, Q, tracking_error])
            cost += ca.mtimes([self.U[:, k].T, R, self.U[:, k]])

        terminal_error = self.X[:, self.mpc_horizon] - self.X_ref[:, self.mpc_horizon]
        cost += ca.mtimes([terminal_error.T, Q_terminal, terminal_error])

        self.opti.minimize(cost)
        self.opti.subject_to(self.X[:, 0] == self.x0)

        for j in range(self.nu):
            self.opti.subject_to(self.opti.bounded(-self.torque_limits[j], self.U[j, :], self.torque_limits[j]))

        p_opts = {
            "expand": True, 
            "print_time": False
        }
        
        # "hessian_approximation": "limited-memory" sucks for IPOPT for OCP
        s_opts = {
            "max_iter": 100,
            "print_level": 0,
            "print_user_options": "no",
            "print_timing_statistics": "no",
            "tol": 1e-4,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 5,
            "warm_start_init_point": "yes",
            "mu_strategy": "adaptive",
        }

        self.opti.solver("ipopt", p_opts, s_opts)

    def solve(self, current_state, reference_horizon, u_guess=None, x_guess=None):
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.X_ref, reference_horizon)

        if u_guess is not None:
            self.opti.set_initial(self.U, u_guess)

        if x_guess is not None:
            self.opti.set_initial(self.X, x_guess)

        try:
            sol = self.opti.solve()
            return sol.value(self.U), sol.value(self.X)
        except Exception:
            print("Solver failed to converge! Returning suboptimal/last known torques.")
            return self.opti.debug.value(self.U), self.opti.debug.value(self.X)

    def get_casadi_pinn_model(self, gravity=None):
        return get_pinocchio_casadi_triplet(self.robot_config, gravity=gravity)

    def get_nominal(self):
        dynamics = build_nominal_euler_dynamics(self.robot_config, self.dt)
        self._dynamics_callback = getattr(dynamics, "_mujoco_callback", None)
        return dynamics

    def get_knode(self):
        nq, nv = self.nq, self.nv
        backend = mpc_dynamics_backend(self.robot_config)

        if backend == "mujoco":
            aba_f = build_mujoco_continuous_dynamics(self.robot_config)
        else:
            _, cmodel, cdata = self.get_casadi_pinn_model()
            aba_f = build_pinocchio_continuous_dynamics(cmodel, cdata)

        residual_input = "q"
        model_json = os.path.join(self.run_dir, "model.json")
        if os.path.isfile(model_json):
            import json
            with open(model_json) as f:
                residual_input = json.load(f).get("residual_input", "q")
        self.residual_input = residual_input if residual_input not in ("q,qdot", "qv") else "q_qdot"

        state_dict, _ = load_state_dict_from_run(self.run_dir, epoch=self.knode_epoch)
        nn_forward, nn_jacobian = self._load_residual_nn_numpy(state_dict)
        in_dim = 2 * nq if self.residual_input == "q_qdot" else nq
        self.nn_cb = ResidualNNCallback(
            nn_forward, nn_jacobian, n_in=in_dim, n_out=nv
        )

        x = ca.MX.sym("x", self.nx)
        u = ca.MX.sym("u", self.nu)
        q = x[:nq]
        v = x[nq:]

        nominal_xdot_mx = aba_f(x, u)
        nn_in = ca.vertcat(q, v) if self.residual_input == "q_qdot" else q
        residual_qdd_mx = self.nn_cb(nn_in)
        xdot = nominal_xdot_mx + ca.vertcat(ca.MX.zeros(nv), residual_qdd_mx)

        x_next = x + xdot * self.dt
        dynamics = ca.Function("F", [x, u], [x_next])
        dynamics._aba_f = aba_f
        dynamics._nn_cb = self.nn_cb
        return dynamics

    def _load_residual_nn_numpy(self, state_dict):
        layers = extract_mlp_layers(state_dict)
        in_dim = 2 * self.nq if getattr(self, "residual_input", "q") == "q_qdot" else self.nq

        def forward(inp):
            h = np.asarray(inp, dtype=np.float64).reshape(in_dim)
            for W, b in layers[:-1]:
                h = np.tanh(W @ h + b)
            W, b = layers[-1]
            return W @ h + b

        def jacobian(inp):
            q_t = torch.tensor(inp, dtype=torch.float64, requires_grad=True).reshape(in_dim)

            def torch_forward(qq):
                h = qq.reshape(in_dim)
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
