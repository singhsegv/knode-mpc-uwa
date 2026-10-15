"""
Acados MPC for n-DOF manipulators (nominal ABA + Euler).

Rosetta stone vs mpc/manipulator_mpc.py (CasADi Opti + IPOPT):

  Opti                                      Acados
  ---------------------------------------   ------------------------------------
  opti.variable X, U over the horizon       dims.N stages; solver owns X[k], U[k]
  for-loop dynamics + cost                  one-stage template, repeated N times
  X[:,k+1] == F(X[:,k], U[:,k])             model.disc_dyn_expr = F(x, u)
  (X-Xref)'Q(X-Xref) + U'RU                 NONLINEAR_LS: ||y - yref||_W^2
  X[:,0] == x0 parameter                    set(0, "lbx"/"ubx", x0) each solve
  set_value(X_ref)                          set(k, "yref", [xref; 0]) each solve
  opti.solver("ipopt"); opti.solve()        AcadosOcpSolver(...); solver.solve()
"""
import os
import sys

import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.model_utils import load_state_dict_from_run
from casadi_dynamics import (
    build_pinocchio_continuous_dynamics,
    get_pinocchio_casadi_triplet,
    mpc_dynamics_backend,
)


def tensor_to_dm(t):
    """torch tensor -> CasADi DM."""
    return ca.DM(t.detach().cpu().numpy())


class AcadosManipulatorControl:
    def __init__(self, horizon, dt, enable_knode=False, run_dir=None, epoch=-1, robot_config=None, json_path=None):
        """
        json_file must be a bare filename (Acados writes it under code_export_directory).
        """
        if robot_config is not None and mpc_dynamics_backend(robot_config) != "pinocchio":
            raise ValueError(
                "Acados requires mpc_dynamics_backend='pinocchio' in the robot config. "
                "Use --solver ipopt with mpc_dynamics_backend='mujoco' to match the MuJoCo plant."
            )

        if enable_knode:
            if run_dir is None:
                raise ValueError("run_dir is required when enable_knode=True")

            self.run_dir = run_dir
            self.knode_epoch = epoch

        self.robot_config = robot_config
        self.nq = robot_config["nq"]
        self.nv = robot_config["nv"]
        self.nx = 2 * self.nq
        self.nu = self.nv
        self.torque_limits = np.asarray(robot_config["torque_limits"], dtype=float)
        self.mpc_horizon = horizon
        self.dt = dt

        # Acados: build OCP template once, then compile to a C solver.
        self.json_file = "acados_ocp.json" if json_path is None else os.path.basename(json_path)

        ocp = self._build_acados_ocp(enable_knode)
        self.solver = AcadosOcpSolver(ocp, json_file=self.json_file, build=True)

    def get_nominal_dynamics(self):
        """
        ABA + forward Euler
        TODO: Check if RK4 is feasible now and is beneficial
        TODO: Check if we need residual network to be SX too somehow.
        """
        _, cmodel, cdata = self.get_casadi_pinn_model()
        nq, nv = cmodel.nq, cmodel.nv

        q = ca.SX.sym("q", nq)
        v = ca.SX.sym("v", nv)
        tau = ca.SX.sym("tau", nv)

        aba_f = build_pinocchio_continuous_dynamics(cmodel, cdata)
        x = ca.vertcat(q, v)
        xdot = aba_f(x, tau)
        u = tau

        x_next = x + xdot * self.dt
        return x, u, x_next

    def get_knode_dynamics(self):
        """
        TODO:
        1. Try with the original hardcoded maths knode
        2. Gravity mismatch might be wrong -> Just test knode mpc speed

        TODO Doubts:
        1. SX vs MX for NN?
        2. Callback style stuff?
        """
        _, cmodel, cdata = self.get_casadi_pinn_model()
        nq, nv = cmodel.nq, cmodel.nv

        q = ca.SX.sym("q", nq)
        v = ca.SX.sym("v", nv)
        tau = ca.SX.sym("tau", nv)

        # ----------- Nominal Dynamics -----------
        aba_f = build_pinocchio_continuous_dynamics(cmodel, cdata)
        x = ca.vertcat(q, v)
        nominal_xdot = aba_f(x, tau)
        nominal_qdd = nominal_xdot[nq:]

        # ----------- Residual Dynamics -----------
        state_dict, _ = load_state_dict_from_run(self.run_dir, epoch=self.knode_epoch)
        # ------------------------------------------------------------
        # For the previous full sized NN (was chosen ad-hoc though)
        # ------------------------------------------------------------
        # params = [
        #     tensor_to_dm(state_dict[f"func.residual_dynamics_network.{i}.{attr}"])
        #     for i in (0, 2, 4, 6)
        #     for attr in ("weight", "bias")
        # ]

        # # TODO: Hardcoded for 2 hidden layers -> FIXME later
        # h = ca.tanh(ca.mtimes(params[0], q) + params[1])
        # h = ca.tanh(ca.mtimes(params[2], h) + params[3])
        # h = ca.tanh(ca.mtimes(params[4], h) + params[5])

        # residual_qdd = ca.mtimes(params[6], h) + params[7]

        # # ------------------------------------------------------------
        # # Tiny single hidden layer NN with 32 neurons
        # # ------------------------------------------------------------
        # params = [
        #     tensor_to_dm(state_dict[f"func.residual_dynamics_network.{i}.{attr}"])
        #     for i in (0, 2, 4)
        #     for attr in ("weight", "bias")
        # ]
        
        # # underwater_model residual takes [q, qdot]
        # nn_in = ca.vertcat(q, v)
        # h = ca.tanh(ca.mtimes(params[0], nn_in) + params[1])
        # # h = ca.tanh(ca.mtimes(params[0], q) + params[1])
        # h = ca.tanh(ca.mtimes(params[2], h) + params[3])
        # residual_qdd = ca.mtimes(params[4], h) + params[5]

        # ------------------------------------------------------------
        # Tiny multiple hidden layer NN with 32 neurons
        # ------------------------------------------------------------
        params = [
            tensor_to_dm(state_dict[f"func.residual_dynamics_network.{i}.{attr}"])
            for i in (0, 2, 4, 6)
            for attr in ("weight", "bias")
        ]

        residual_input = "q"
        model_json = os.path.join(self.run_dir, "model.json")
        if os.path.isfile(model_json):
            import json
            with open(model_json) as f:
                residual_input = json.load(f).get("residual_input", "q")

        if residual_input in ("q_qdot", "q,qdot", "qv"):
            nn_in = ca.vertcat(q, v)
            h = ca.tanh(ca.mtimes(params[0], nn_in) + params[1])
        else:
            h = ca.tanh(ca.mtimes(params[0], q) + params[1])
        h = ca.tanh(ca.mtimes(params[2], h) + params[3])
        h = ca.tanh(ca.mtimes(params[4], h) + params[5])
        residual_qdd = ca.mtimes(params[6], h) + params[7]

        qdd = nominal_qdd + residual_qdd
        x = ca.vertcat(q, v)
        xdot = ca.vertcat(v, qdd)
        u = tau

        x_next = x + xdot * self.dt
        return x, u, x_next

    def _build_acados_ocp(self, enable_knode):
        """
        Build phase of the OCP -> C code generation
        """
        nx, nu = self.nx, self.nu

        # Acados stores one discrete map; dims.N repeats it.
        if enable_knode:
            x, u, x_next = self.get_knode_dynamics()
        else:
            x, u, x_next = self.get_nominal_dynamics()

        model = AcadosModel()
        model.name = "knode_mpc"
        model.x = x
        model.u = u
        model.disc_dyn_expr = x_next

        # Opti stage level cost vs Acados NONLINEAR_LS minimizes || y - yref ||_W^2 with
        #   y = [x; u],  yref = [x_ref; 0],  W = blkdiag(Q, R)
        # which expands to the same quadratic.
        model.cost_y_expr = ca.vertcat(x, u)   # ny = nx + nu
        model.cost_y_expr_e = x                # terminal: only state (ny_e = nx)

        # Default UR5-style weights; override via robot_config mpc_Q_q / mpc_Q_qd / mpc_R
        # for CaDeLaC-matched Panda costs (state order here is [q; qd]).
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
        Q_terminal = Q * 10  # Opti: Q_terminal = Q * 10

        ocp = AcadosOcp()
        ocp.model = model

        ocp.solver_options.N_horizon = self.mpc_horizon
        ocp.solver_options.tf = self.mpc_horizon * self.dt

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = block_diag(Q, R)        
        ocp.cost.W_e = Q_terminal

        # Placeholders; real refs set every solve() via set(k, "yref", ...)
        ocp.cost.yref = np.zeros(nx + nu)
        ocp.cost.yref_e = np.zeros(nx)

        ocp.constraints.lbu = -self.torque_limits.copy()
        ocp.constraints.ubu = self.torque_limits.copy()
        ocp.constraints.idxbu = np.arange(nu)

        # Opti: subject_to(X[:,0] == x0)  — x0 is a parameter updated each step.
        # Acados: equality via lbx=ubx at stage 0 (updated in solve()).
        ocp.constraints.x0 = np.zeros(nx)

        # Solver options: RTI = 1 SQP step per call (vs IPOPT's many NLP iters)
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.nlp_solver_max_iter = 1
        ocp.solver_options.qp_solver_iter_max = 50
        ocp.solver_options.print_level = 0

        ocp.code_export_directory = os.path.join(SCRIPT_DIR, "c_generated_code")

        return ocp

    def solve(self, current_state, reference_horizon, u_guess=None, x_guess=None):
        current_state = np.asarray(current_state, dtype=float).reshape(self.nx)
        reference_horizon = np.asarray(reference_horizon, dtype=float)
        assert reference_horizon.shape == (self.nx, self.mpc_horizon + 1)

        # Replacement for Opti set_value(self.x0, current_state)
        self.solver.set(0, "lbx", current_state)
        self.solver.set(0, "ubx", current_state)

        for k in range(self.mpc_horizon):
            # "temp" is the Stage yref = [x_ref; u_ref] with u_ref=0 → matches Opti u'R u term."""
            temp = np.asarray(reference_horizon[:, k], dtype=float).reshape(self.nx) # Shape (12, )
            temp = np.concatenate([temp, np.zeros(self.nu)]) # Shape (18, )

            self.solver.set(k, "yref", temp)

            if u_guess is not None:
                self.solver.set(k, "u", np.asarray(u_guess[:, k], dtype=float))

            if x_guess is not None:
                self.solver.set(k, "x", np.asarray(x_guess[:, k], dtype=float))

        # Terminal: only state residual (cost_y_expr_e = x)
        self.solver.set(self.mpc_horizon, "yref", reference_horizon[:, self.mpc_horizon])

        if x_guess is not None:
            self.solver.set(self.mpc_horizon, "x", np.asarray(x_guess[:, self.mpc_horizon], dtype=float))

        status = self.solver.solve()
        if status != 0:
            print(f"Acados solve returned status {status} (0 = success).")

        U = np.column_stack([self.solver.get(k, "u") for k in range(self.mpc_horizon)])
        X = np.column_stack([self.solver.get(k, "x") for k in range(self.mpc_horizon + 1)])
        return U, X

    def get_casadi_pinn_model(self, gravity=None):
        return get_pinocchio_casadi_triplet(self.robot_config, gravity=gravity)
