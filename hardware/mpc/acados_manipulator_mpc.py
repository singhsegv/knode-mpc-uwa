"""
Acados MPC for an n-DOF arm on a ros2_control stack.

The dynamics model is the Pinocchio model built from the same URDF the
controller manager is running. Joint order is that model's configuration
order, which is also Ros2ControlPlant.joint_names. The node reads
[q, qdot] from the plant, calls solve(), and writes U[:, 0] with
write_effort(). This class does not publish or subscribe.

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

import numpy as np
import casadi as ca
import pinocchio as pin
from scipy.linalg import block_diag
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

import pinocchio.casadi as cpin

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# URDF parsers store "no limit" as a huge finite number.
_UNLIMITED = 1e6


class AcadosManipulatorControl:
    def __init__(self, model, horizon, dt, torque_limits=None, gravity=None, json_path=None):
        """
        model: pinocchio.Model from the ros2_control URDF.
        torque_limits: (nv,) continuous torque in Pinocchio order.
            Defaults to the URDF effort limit.
        gravity: (3,) linear gravity in the base frame. Defaults to model.gravity.
        json_file must be a bare filename (Acados writes it under code_export_directory).
        """
        self.model = pin.Model(model)
        if gravity is not None:
            self.model.gravity.linear = np.asarray(gravity, dtype=float)

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv
        self.nu = self.nv
        self.mpc_horizon = horizon
        self.dt = dt

        if torque_limits is None:
            torque_limits = self._torque_limits_from_model()
        self.torque_limits = np.asarray(torque_limits, dtype=float).reshape(self.nu)
        if np.any(self.torque_limits <= 0.0):
            raise ValueError("torque_limits must be positive")

        self._u_guess = None
        self._x_guess = None

        self.json_file = "acados_ocp.json" if json_path is None else os.path.basename(json_path)
        ocp = self._build_acados_ocp()
        self.solver = AcadosOcpSolver(ocp, json_file=self.json_file, build=True)

    def _torque_limits_from_model(self):
        limits = np.array(self.model.effortLimit, dtype=float).reshape(self.nv)
        if np.any(~np.isfinite(limits)) or np.any(limits <= 0.0) or np.any(limits >= _UNLIMITED):
            raise ValueError(
                "URDF effort limits are missing. Pass torque_limits= (continuous torque per joint)."
            )
        return limits

    def _bounded_states(self):
        """Position and velocity indices that have a real URDF limit."""
        q_min = np.array(self.model.lowerPositionLimit, dtype=float)
        q_max = np.array(self.model.upperPositionLimit, dtype=float)
        v_max = np.array(self.model.velocityLimit, dtype=float)

        idx = []
        lb = []
        ub = []
        for i in range(self.nq):
            if (
                np.isfinite(q_min[i]) and np.isfinite(q_max[i])
                and abs(q_min[i]) < _UNLIMITED and abs(q_max[i]) < _UNLIMITED
            ):
                idx.append(i)
                lb.append(q_min[i])
                ub.append(q_max[i])
        for i in range(self.nv):
            if np.isfinite(v_max[i]) and 0.0 < v_max[i] < _UNLIMITED:
                idx.append(self.nq + i)
                lb.append(-v_max[i])
                ub.append(v_max[i])

        if not idx:
            return None
        return (
            np.asarray(idx, dtype=int),
            np.asarray(lb, dtype=float),
            np.asarray(ub, dtype=float),
        )

    def get_nominal_dynamics(self):
        """ABA + forward Euler. dt matches the ros2_control command period."""
        _, cmodel, cdata = self.get_casadi_model()
        nq, nv = cmodel.nq, cmodel.nv

        q = ca.SX.sym("q", nq)
        v = ca.SX.sym("v", nv)
        tau = ca.SX.sym("tau", nv)

        qdd = cpin.aba(cmodel, cdata, q, v, tau)
        x = ca.vertcat(q, v)
        xdot = ca.vertcat(v, qdd)
        u = tau

        x_next = x + xdot * self.dt
        return x, u, x_next

    def _build_acados_ocp(self):
        """Build phase of the OCP -> C code generation."""
        nx, nu = self.nx, self.nu
        x, u, x_next = self.get_nominal_dynamics()

        model = AcadosModel()
        model.name = "knode_mpc"
        model.x = x
        model.u = u
        model.disc_dyn_expr = x_next

        # NONLINEAR_LS minimizes || y - yref ||_W^2 with
        #   y = [x; u],  yref = [x_ref; 0],  W = blkdiag(Q, R)
        model.cost_y_expr = ca.vertcat(x, u)
        model.cost_y_expr_e = x

        q_weights = np.full(self.nq, 100.0)
        qdot_weights = np.full(self.nv, 100.0)
        Q = np.diag(np.concatenate([q_weights, qdot_weights]))
        R = np.diag(np.full(self.nu, 0.01))
        Q_terminal = Q * 10

        ocp = AcadosOcp()
        ocp.model = model

        ocp.solver_options.N_horizon = self.mpc_horizon
        ocp.solver_options.tf = self.mpc_horizon * self.dt

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = block_diag(Q, R)
        ocp.cost.W_e = Q_terminal

        ocp.cost.yref = np.zeros(nx + nu)
        ocp.cost.yref_e = np.zeros(nx)

        ocp.constraints.lbu = -self.torque_limits.copy()
        ocp.constraints.ubu = self.torque_limits.copy()
        ocp.constraints.idxbu = np.arange(nu)

        # Stage 0 is the measured plant state. Updated in solve().
        ocp.constraints.x0 = np.zeros(nx)

        bounds = self._bounded_states()
        if bounds is not None:
            idx, lb, ub = bounds
            ocp.constraints.idxbx = idx
            ocp.constraints.lbx = lb
            ocp.constraints.ubx = ub
            ocp.constraints.idxbx_e = idx
            ocp.constraints.lbx_e = lb
            ocp.constraints.ubx_e = ub

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.nlp_solver_max_iter = 1
        ocp.solver_options.qp_solver_iter_max = 50
        ocp.solver_options.print_level = 0

        ocp.code_export_directory = os.path.join(SCRIPT_DIR, "c_generated_code")
        return ocp

    def _reference_horizon(self, reference):
        """
        reference is either a joint goal (nq,) or (nx,), held for the whole
        horizon, or a full trajectory (nx, N+1). qdot defaults to zero when
        only q is given, which is the regulation case.
        """
        reference = np.asarray(reference, dtype=float)
        if reference.shape == (self.nq,):
            reference = np.concatenate([reference, np.zeros(self.nv)])
        if reference.shape == (self.nx,):
            return np.repeat(reference.reshape(self.nx, 1), self.mpc_horizon + 1, axis=1)
        if reference.shape == (self.nx, self.mpc_horizon + 1):
            return reference
        raise ValueError(
            f"reference shape {reference.shape} is not {(self.nq,)}, {(self.nx,)}, "
            f"or {(self.nx, self.mpc_horizon + 1)}"
        )

    def solve(self, current_state, reference, u_guess=None, x_guess=None):
        """
        current_state: [q, qdot] from Ros2ControlPlant.read_state(), Pinocchio order.
        reference: joint goal or state horizon. See _reference_horizon.
        Returns U (nu, N) and X (nx, N+1). The command to write is U[:, 0].
        """
        current_state = np.asarray(current_state, dtype=float).reshape(self.nx)
        reference_horizon = self._reference_horizon(reference)

        if u_guess is None:
            u_guess = self._u_guess
        if x_guess is None:
            x_guess = self._x_guess

        self.solver.set(0, "lbx", current_state)
        self.solver.set(0, "ubx", current_state)

        for k in range(self.mpc_horizon):
            yref = np.concatenate([
                np.asarray(reference_horizon[:, k], dtype=float).reshape(self.nx),
                np.zeros(self.nu),
            ])
            self.solver.set(k, "yref", yref)

            if u_guess is not None:
                self.solver.set(k, "u", np.asarray(u_guess[:, k], dtype=float))
            if x_guess is not None:
                self.solver.set(k, "x", np.asarray(x_guess[:, k], dtype=float))

        self.solver.set(self.mpc_horizon, "yref", reference_horizon[:, self.mpc_horizon])
        if x_guess is not None:
            self.solver.set(self.mpc_horizon, "x", np.asarray(x_guess[:, self.mpc_horizon], dtype=float))

        status = self.solver.solve()
        if status != 0:
            print(f"Acados solve returned status {status} (0 = success).")

        U = np.column_stack([self.solver.get(k, "u") for k in range(self.mpc_horizon)])
        X = np.column_stack([self.solver.get(k, "x") for k in range(self.mpc_horizon + 1)])
        self._store_guess(current_state, U, X)
        return U, X

    def _store_guess(self, current_state, U, X):
        u_prev = np.zeros_like(U)
        u_prev[:, :-1] = U[:, 1:]
        u_prev[:, -1] = U[:, -1]

        x_prev = np.zeros_like(X)
        x_prev[:, 0] = current_state
        x_prev[:, 1:-1] = X[:, 2:]
        x_prev[:, -1] = X[:, -1]

        self._u_guess = u_prev
        self._x_guess = x_prev

    def get_casadi_model(self):
        cmodel = cpin.Model(self.model)
        cdata = cmodel.createData()
        return self.model, cmodel, cdata
