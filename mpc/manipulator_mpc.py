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
        self.X = self.opti.variable(4, self.mpc_horizon + 1)
        self.U = self.opti.variable(2, self.mpc_horizon)

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

        for k in range(self.mpc_horizon):
            self.opti.subject_to(self.X[:, k+1] == self.dynamics(self.X[:, k], self.U[:, k]))

            # Tracking cost
            tracking_error = self.X[:, k] - self.X_ref[:, k]
            cost += ca.mtimes([tracking_error.T, Q, tracking_error])
            cost += ca.mtimes([self.U[:, k].T, R, self.U[:, k]])
            # cost += ca.sumsqr(X[:, k] - X_ref[:, k]) + 1e-3 * ca.sumsqr(U[:, k])
            
        # Terminal cost
        terminal_error = self.X[:, self.mpc_horizon] - self.X_ref[:, self.mpc_horizon]
        cost += ca.mtimes([terminal_error.T, Q * 25, terminal_error])
        # cost += 10 * ca.sumsqr(X[:, self.mpc_horizon] - X_ref[:, self.mpc_horizon])

        self.opti.minimize(cost)
        self.opti.subject_to(self.X[:, 0] == self.x0)
        self.opti.subject_to(self.opti.bounded(-100, self.U, 100))

        # expand=True inlines the full symbolic graph and can use 10+ GB when
        # Pinocchio ABA + MLP is chained over the MPC horizon.

        p_opts = {"expand": False, "print_time": False}
        s_opts = {
            "max_iter": 300,
            "print_level": 0,
            "print_user_options": "no",
            "print_timing_statistics": "no",
            "tol": 1e-4,
            "hessian_approximation": "limited-memory",
        }
        self.opti.solver("ipopt", p_opts, s_opts)

    def solve(self, current_state, reference_horizon, u_guess=None):
        """ Called every control loop to get the next optimal torque. """
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.X_ref, reference_horizon)

        if u_guess is not None:
            self.opti.set_initial(self.U, u_guess)

        try:
            sol = self.opti.solve()
            return sol.value(self.U), sol.value(self.X)
        except:
            print("Solver failed to converge! Returning suboptimal/last known torques.")
            return self.opti.debug.value(self.U), self.opti.debug.value(self.X)

    def get_casadi_pinn_model(self, gravity=np.array([0.0, -9.81, 0.0])):
        model = pin.buildModelFromUrdf(self.urdf_path)
        model.gravity.linear = gravity
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        return model, cmodel, cdata

    def get_nominal(self):
        """ Perfect/Imperfect nominal models based on Pinocchio.CasADi. """
        _, cmodel, cdata = self.get_casadi_pinn_model(gravity=np.array([-2.0, -12.0, 0.0]))

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

        # f = ca.Function("f", [x, u], [xdot])
        # intg = ca.integrator(
        #     "intg", "rk",
        #     { "x": x, "p": u, "ode": f(x, u)},
        #     dict(tf=self.dt, simplify=True, number_of_finite_elements=4),
        # )

        # return ca.Function("F", [x, u], [intg(x0=x, p=u)["xf"]])
        
        x_next = x + xdot * self.dt
        return ca.Function("F", [x, u], [x_next])

    
    # def get_knode(self):
    #     """
    #     Perfect/Imperfect nominal + Residual network

    #     ----------------------------------------------------------------------
    #     NOTE: Designed for residual qdd and not residual tau
    #     ----------------------------------------------------------------------

    #     Residual KNODE model (epoch=-1 -> latest checkpoint)

    #     TODO: Doesn't support input normalization yet like the previous knode model used
    #     TODO: Try L4CasADi later

    #     NOTE: Previous imperfect thingy, not I use gravity
    #     g = cpin.computeGeneralizedGravity(cmodel, cdata, q)
    #     qdd = cpin.aba(cmodel, cdata, q, v, tau + g)

    #     Only plant should do this
    #     _, cmodel, cdata = self.get_casadi_pinn_model(gravity=np.array([0.0, -11.0, 0.0]))

    #     # TODO: SKETCHY AF -> Should this be stored/loaded instead of being calculated on-the-fly??
    #     # mean = tensor_to_dm(state_dict["func.residual_input_mean"])
    #     # std  = tensor_to_dm(state_dict["func.residual_input_std"])
    #     # z_n = (z - mean) / std
    #     """
    #     _, cmodel, cdata = self.get_casadi_pinn_model(gravity=np.array([0.0, -11.0, 0.0]))

    #     nq, nv = cmodel.nq, cmodel.nv
    #     q = ca.SX.sym("q", nq)
    #     v = ca.SX.sym("v", nv) # qdot
    #     tau = ca.SX.sym("tau", nv)

    #     # ----------- Nominal Dynamics -----------
    #     nominal_qdd = cpin.aba(cmodel, cdata, q, v, tau)

    #     # ----------- Residual Dynamics -----------
    #     # TODO: Check us, were are sketchy and ai
    #     state_dict, _ = load_state_dict_from_run(self.run_dir, epoch=self.knode_epoch)
    #     params = [
    #         tensor_to_dm(state_dict[f"func.residual_dynamics_network.{i}.{attr}"])
    #         for i in (0, 2, 4, 6)
    #         for attr in ("weight", "bias")
    #     ]

    #     # TODO: Hardcoded for 2 hidden layers -> FIXME later
    #     h = ca.tanh(ca.mtimes(params[0], q) + params[1])
    #     h = ca.tanh(ca.mtimes(params[2], h) + params[3])
    #     h = ca.tanh(ca.mtimes(params[4], h) + params[5])
    #     residual_qdd = ca.mtimes(params[6], h) + params[7]

    #     qdd = nominal_qdd + residual_qdd

    #     xdot = ca.vertcat(v, qdd)
    #     x = ca.vertcat(q, v)
    #     u = tau

    #     # f = ca.Function("f", [x, u], [xdot])
    #     # intg = ca.integrator(
    #     #     "intg", "rk",
    #     #     {"x": x, "p": u, "ode": f(x, u)},
    #     #     dict(tf=self.dt, simplify=True, number_of_finite_elements=1),
    #     # )

    #     # return ca.Function("F", [x, u], [intg(x0=x, p=u)["xf"]])

    #     x_next = x + xdot * self.dt
    #     return ca.Function("F", [x, u], [x_next])

    def get_knode(self):
        """
        Perfect/Imperfect nominal + Residual network.
        CasADi KNODE fixes.
        """
        _, cmodel, cdata = self.get_casadi_pinn_model(gravity=np.array([-2.0, -12.0, 0.0]))
        nq, nv = cmodel.nq, cmodel.nv

        q_sx = ca.SX.sym("q", nq)
        v_sx = ca.SX.sym("v", nv) # qdot
        tau_sx = ca.SX.sym("tau", nv)
        
        x_sx = ca.vertcat(q_sx, v_sx)
        u_sx = tau_sx

        nominal_qdd = cpin.aba(cmodel, cdata, q_sx, v_sx, tau_sx)
        nominal_xdot = ca.vertcat(v_sx, nominal_qdd)
        aba_f = ca.Function("aba_f", [x_sx, u_sx], [nominal_xdot])

        # ----------- Residual Dynamics -----------
        state_dict, _ = load_state_dict_from_run(self.run_dir, epoch=self.knode_epoch)
        nn_forward, nn_jacobian = self._load_residual_nn_numpy(state_dict)
        self.nn_cb = ResidualNNCallback(nn_forward, nn_jacobian)

        x = ca.MX.sym("x", self.nx)
        u = ca.MX.sym("u", self.nu)
        q = x[:2]

        nominal_xdot_mx = aba_f(x, u) # Function call node — fast AD
        residual_qdd_mx = self.nn_cb(q) # Callback node — numeric AD
        
        xdot = nominal_xdot_mx + ca.vertcat(0, 0, residual_qdd_mx)

        x_next = x + xdot * self.dt
        return ca.Function("F", [x, u], [x_next])

    def _load_residual_nn_numpy(self, state_dict):
        """Returns forward(q) and jacobian(q) callables."""
        params = [
            state_dict[f"func.residual_dynamics_network.{i}.{attr}"].detach().cpu().numpy()
            for i in (0, 2, 4, 6)
            for attr in ("weight", "bias")
        ]

        W0, b0, W2, b2, W4, b4, W6, b6 = params

        def forward(q):
            q = np.asarray(q, dtype=np.float64).reshape(2)
            h = np.tanh(W0 @ q + b0)
            h = np.tanh(W2 @ h + b2)
            h = np.tanh(W4 @ h + b4)
            return W6 @ h + b6 # shape (2,)

        def _torch_forward(qq, layers):
            h = qq.reshape(2)
            for W, b in layers[:-1]:
                Wt = torch.as_tensor(W, dtype=torch.float64)
                bt = torch.as_tensor(b, dtype=torch.float64)
                h = torch.tanh(Wt @ h + bt)
            W, b = layers[-1]
            Wt = torch.as_tensor(W, dtype=torch.float64)
            bt = torch.as_tensor(b, dtype=torch.float64)
            return Wt @ h + bt

        def jacobian(q):
            q_t = torch.tensor(q, dtype=torch.float64, requires_grad=True).reshape(2)
            layers = [(W0, b0), (W2, b2), (W4, b4), (W6, b6)]
            h = q_t
            for i, (W, b) in enumerate(layers[:-1]):
                h = torch.tanh(torch.tensor(W, dtype=torch.float64) @ h + torch.tensor(b, dtype=torch.float64))
            
            W, b = layers[-1]
            out = torch.tensor(W, dtype=torch.float64) @ h + torch.tensor(b, dtype=torch.float64)
            J = torch.autograd.functional.jacobian(lambda qq: _torch_forward(qq, layers), q_t) # TODO: Check me
            return J.detach().numpy() # shape (2, 2)

        return forward, jacobian