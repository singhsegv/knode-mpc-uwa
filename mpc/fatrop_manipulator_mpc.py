import os
import sys

import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.model_utils import load_state_dict_from_run


def tensor_to_dm(t):
    """torch tensor -> CasADi DM."""
    return ca.DM(t.detach().cpu().numpy())


class ManipulatorControl:
    """
    MPC solver.
    """
    def __init__(self, horizon, dt, enable_knode=False, run_dir=None, epoch=-1):
        self.urdf_path = os.path.join(PROJECT_ROOT, "robots", "planar-2R.urdf")

        self.mpc_horizon = horizon
        self.dt = dt
        self.nx = 4
        self.nu = 2

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
        for k in range(self.mpc_horizon):
            self.X.append(self.opti.variable(self.nx, 1))
            self.U.append(self.opti.variable(self.nu, 1))

        self.X.append(self.opti.variable(self.nx, 1))

        self.x0 = self.opti.parameter(self.nx, 1)
        self.X_ref = self.opti.parameter(self.nx, self.mpc_horizon + 1)

        self._build_optimization_problem()        

    def _build_optimization_problem(self):
        """
        Sets up the symbolic graph. 
        "opti.parameter" does the magic.
        """
        cost = 0

        # TODO: Tune us
        Q = np.diag([100.0, 100.0, 100.0, 100.0]) # state penalty
        R = np.diag([0.01, 0.01]) # control errors

        self.opti.subject_to(self.X[0] == self.x0)

        for k in range(self.mpc_horizon):
            tracking_error = self.X[k] - self.X_ref[:, k:k+1]
            cost += ca.mtimes([tracking_error.T, Q, tracking_error])
            cost += ca.mtimes([self.U[k].T, R, self.U[k]])

            self.opti.subject_to(self.X[k+1] == self.dynamics(self.X[k], self.U[k]))
            self.opti.subject_to(self.opti.bounded(-100, self.U[k], 100))

        # Terminal cost
        terminal_error = self.X[self.mpc_horizon] - self.X_ref[self.mpc_horizon]
        cost += ca.mtimes([terminal_error.T, Q * 10, terminal_error])

        self.opti.minimize(cost)

        # opts = {
        #     "jit": True,
        #     "compiler": "shell",
        #     "fatrop": {"print_level": 0}
        # }
        opts = {
            "expand": True,
            "fatrop": {"print_level": 0}
        }

        self.opti.solver("fatrop", opts)

    def solve(self, current_state, reference_horizon, u_guess=None):
        """ Called every control loop to get the next optimal torque. """
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.X_ref, reference_horizon)

        # Optional: Set warm starts if you have a guess
        if u_guess is not None:
            for k in range(self.mpc_horizon):
                self.opti.set_initial(self.U[k], u_guess[:, k])

        try:
            sol = self.opti.solve()
            
            # FIX: Use column_stack to build a 2D matrix (dim x horizon)
            U_optimal = np.column_stack([sol.value(u).flatten() for u in self.U])
            X_predicted = np.column_stack([sol.value(x).flatten() for x in self.X])
            
            return U_optimal, X_predicted

        except Exception as e:
            print(f"Solver failed: {e}")
            
            # Fallback debug extraction
            U_optimal = np.column_stack([self.opti.debug.value(u).flatten() for u in self.U])
            X_predicted = np.column_stack([self.opti.debug.value(x).flatten() for x in self.X])
            
            return U_optimal, X_predicted

    def get_casadi_pinn_model(self, gravity=np.array([0.0, -9.81, 0.0])):
        model = pin.buildModelFromUrdf(self.urdf_path)
        model.gravity.linear = gravity
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        return model, cmodel, cdata

    def get_nominal(self):
        """ Perfect/Imperfect nominal models based on Pinocchio.CasADi. """
        _, cmodel, cdata = self.get_casadi_pinn_model(gravity=np.array([0.0, -11.0, 0.0]))

        # TODO: MX or SX ??
        nq, nv = cmodel.nq, cmodel.nv
        q = ca.SX.sym("q", nq)
        v = ca.SX.sym("v", nv) # qdot
        tau = ca.SX.sym("tau", nv)

        qdd = cpin.aba(cmodel, cdata, q, v, tau)

        # Moving from q to x naming convention for clear state derivative shizz
        x = ca.vertcat(q, v) # state for MPC: [q, qdot], size 4
        u = tau
        xdot = ca.vertcat(v, qdd)

        f = ca.Function("f", [x, u], [xdot])
        intg = ca.integrator(
            "intg", "rk",
            { "x": x, "p": u, "ode": f(x, u)},
            dict(tf=self.dt, simplify=True, number_of_finite_elements=4),
        )

        return ca.Function("F", [x, u], [intg(x0=x, p=u)["xf"]])

    
    def get_knode(self):
        _, cmodel, cdata = self.get_casadi_pinn_model(gravity=np.array([0.0, -11.0, 0.0]))

        nq, nv = cmodel.nq, cmodel.nv
        q = ca.SX.sym("q", nq)
        v = ca.SX.sym("v", nv) # qdot
        tau = ca.SX.sym("tau", nv)

        # ----------- Nominal Dynamics -----------
        nominal_qdd = cpin.aba(cmodel, cdata, q, v, tau)

        # ----------- Residual Dynamics -----------
        # TODO: Check us, were are sketchy and ai
        state_dict, _ = load_state_dict_from_run(self.run_dir, epoch=self.knode_epoch)
        params = [
            tensor_to_dm(state_dict[f"func.residual_dynamics_network.{i}.{attr}"])
            for i in (0, 2, 4, 6)
            for attr in ("weight", "bias")
        ]

        # TODO: Hardcoded for 2 hidden layers -> FIXME later
        h = ca.tanh(ca.mtimes(params[0], q) + params[1])
        h = ca.tanh(ca.mtimes(params[2], h) + params[3])
        h = ca.tanh(ca.mtimes(params[4], h) + params[5])
        residual_qdd = ca.mtimes(params[6], h) + params[7]

        qdd = nominal_qdd + residual_qdd

        xdot = ca.vertcat(v, qdd)
        x = ca.vertcat(q, v)
        u = tau

        # f = ca.Function("f", [x, u], [xdot])
        # intg = ca.integrator(
        #     "intg", "rk",
        #     {"x": x, "p": u, "ode": f(x, u)},
        #     dict(tf=self.dt, simplify=True, number_of_finite_elements=4),
        # )

        # return ca.Function("F", [x, u], [intg(x0=x, p=u)["xf"]])

        M = 4
        dt_step = self.dt / M
        
        X_k = x
        for _ in range(M):
            k1 = f(X_k, u)
            k2 = f(X_k + (dt_step / 2) * k1, u)
            k3 = f(X_k + (dt_step / 2) * k2, u)
            k4 = f(X_k + dt_step * k3, u)
            X_k = X_k + (dt_step / 6) * (k1 + 2*k2 + 2*k3 + k4)
            
        return ca.Function("F", [x, u], [X_k])
