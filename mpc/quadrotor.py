"""
Training:     PyTorch  + Pinocchio (frozen nominal) + NN (trainable)
MPC:          CasADi   + pinocchio.casadi (nominal) + exported NN weights
Evaluation:   Pinocchio true plant (with mismatch)
"""
import numpy as np
import torch
from casadi import *

from Simulation.quadrotor_params import MPC_DT, MPC_HORIZON, quad_params


class Control(object):
    """CasADi MPC controller supporting nominal and KNODE dynamics."""

    def __init__(self):
        self.mass = quad_params['mass']
        self.length = quad_params['arm_length']
        self.rotor_min = quad_params['rotor_min']
        self.rotor_max = quad_params['rotor_max']
        self.k_t = quad_params['k_t']
        self.k_d = quad_params['k_d']
        self.inertia = np.diag(np.array([quad_params['Ixx'], quad_params['Iyy'], quad_params['Izz']]))
        self.inv_inertia = np.linalg.inv(self.inertia)

        k = self.k_d / self.k_t
        self.cf_map = np.array([
            [1, 1, 1, 1],
            [0, self.length, 0, -self.length],
            [-self.length, 0, self.length, 0],
            [k, -k, k, -k],
        ])
        self.fc_map = np.linalg.inv(self.cf_map)
        self.motor_spd = 1790.0
        force = self.k_t * np.square(self.motor_spd)
        self.forces_old = np.array([force, force, force, force])

        self.init_mpc = 0
        self.x = np.zeros((1,))
        self.g = np.zeros((1,))
        self._build_nominal_dynamics()

    def _nominal_casadi(self, x, u):
        pqr_vec = vertcat(x[10], x[11], x[12])
        g_transpose = horzcat(
            vertcat(x[9], x[8], -x[7], -x[6]),
            vertcat(-x[8], x[9], x[6], -x[7]),
            vertcat(x[7], -x[6], x[9], -x[8]),
        )
        quat_dot_val = 0.5 * mtimes(g_transpose, pqr_vec)
        ode_without_u = vertcat(
            vertcat(x[3], x[4], x[5]),
            vertcat(0, 0, -9.81),
            quat_dot_val,
            mtimes(self.inv_inertia, (-cross(pqr_vec, mtimes(self.inertia, pqr_vec)))),
        )
        xdotdot_u = vertcat(
            2 * (x[6] * x[8] + x[7] * x[9]),
            2 * (x[7] * x[8] - x[6] * x[9]),
            (1 - 2 * (x[6] ** 2 + x[7] ** 2)),
        ) / self.mass * u[0]
        u_component = vertcat(
            vertcat(0, 0, 0),
            xdotdot_u,
            vertcat(0, 0, 0, 0),
            mtimes(self.inv_inertia, vertcat(u[1], u[2], u[3])),
        )
        return ode_without_u + u_component

    def _build_nominal_dynamics(self):
        x = MX.sym('x', 13, 1)
        u = MX.sym('u', 4, 1)
        nominal = self._nominal_casadi(x, u)
        f = Function('f', [x, u], [nominal])
        intg = integrator(
            'intg', 'rk', {'x': x, 'p': u, 'ode': f(x, u)},
            dict(tf=MPC_DT, simplify=True, number_of_finite_elements=4),
        )
        self.Dynamics = Function('F', [x, u], [intg(x0=x, p=u)['xf']])
        self._state_sym = x
        self._input_sym = u

    def load_knode_model(self, node_path):
        ode_torch = torch.load(node_path, map_location=torch.device('cpu'), weights_only=False)['ode_train']
        param_ls = [layer.detach().cpu().numpy() for _, layer in ode_torch.func.state_dict().items()]

        x = self._state_sym
        u = self._input_sym
        nominal = self._nominal_casadi(x, u)
        ode_nn = vertcat(x, u)
        param_cnt = 0

        for layer in ode_torch.func.nn_model:
            if str(layer) == 'Tanh()':
                ode_nn = tanh(ode_nn)
            else:
                ode_nn = mtimes(param_ls[param_cnt], ode_nn) + param_ls[param_cnt + 1]
                param_cnt += 2

        hybrid = nominal + ode_nn
        f = Function('f', [x, u], [hybrid])
        intg = integrator(
            'intg', 'rk', {'x': x, 'p': u, 'ode': f(x, u)},
            dict(tf=MPC_DT, simplify=True, number_of_finite_elements=4),
        )
        self.Dynamics = Function('F', [x, u], [intg(x0=x, p=u)['xf']])

    def update(self, state, flat):
        opti = Opti()
        n = MPC_HORIZON
        x = opti.variable(13, n + 1)
        u = opti.variable(4, n)

        opti.minimize(
            sumsqr(x[0:3, :] - repmat(DM(flat['x']), 1, n + 1))
            + sumsqr(x[3:6, :] - repmat(DM(flat['x_dot']), 1, n + 1))
            + sumsqr(x[6:9, :])
            + sumsqr(x[9, :] - 1.0)
            + sumsqr(x[10:13, :])
            + sumsqr(u)
        )

        for k in range(n):
            opti.subject_to(x[:, k + 1] == self.Dynamics(x[:, k], u[:, k]))

        opti.subject_to(opti.bounded(0.0, u[0, :], 0.575))
        opti.subject_to(x[:, 0] == vertcat(state['x'], state['v'], state['q'], state['w']))
        opti.solver('ipopt', dict(print_time=False), dict(print_level=0, warm_start_init_point='yes'))

        if self.init_mpc >= 1:
            opti.set_initial(opti.x, self.x)
            opti.set_initial(opti.lam_g, self.g)

        sol = opti.solve()

        forces = np.squeeze(self.fc_map @ np.squeeze(np.array([sol.value(u[:, 0])])))
        forces[forces < 0] = np.square(self.forces_old[forces < 0]) * self.k_t
        cmd_motor_speeds = np.clip(np.sqrt(forces / self.k_t), self.rotor_min, self.rotor_max)
        output = self.cf_map @ (self.k_t * np.square(cmd_motor_speeds))

        self.init_mpc += 1
        self.forces_old = forces
        self.x = sol.value(opti.x)
        self.g = sol.value(opti.lam_g)

        return {'cmd_thrust': output[0], 'cmd_moment': output[1:]}


def instantiate_controller(model_type='nominal', model_path=None, gp_predict_fn=None):
    if model_type == 'knode':
        controller = Control()
        controller.load_knode_model(model_path)
        return controller
    if model_type == 'gp':
        return GPControl(gp_predict_fn)
    return Control()
