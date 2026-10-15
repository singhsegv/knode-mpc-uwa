"""
CasADi dynamics builders for MPC.

Closed-loop plant rollouts use MuJoCo (`robots.simulator.Manipulator`). MPC solvers
need a differentiable discrete map inside the optimizer. Two backends are supported:

- pinocchio (default): symbolic ABA via pinocchio.casadi. Required for acados/fatrop
  because they codegen smooth SX graphs at build time.
- mujoco: CasADi Callback over Manipulator.forward_dynamics + forward Euler. Use with
  the IPOPT backend when you want the MPC internal model to match the MuJoCo plant.
"""
import os

import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

from robots.simulator import Manipulator


def mpc_dynamics_backend(robot_config):
    return robot_config.get("mpc_dynamics_backend", "pinocchio")


def load_pinocchio_casadi(robot_config, gravity=None):
    if gravity is None:
        gravity = robot_config["model_gravity"]

    mjcf_path = robot_config.get("pinocchio_mjcf")
    if mjcf_path:
        if not os.path.isabs(mjcf_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mjcf_path = os.path.join(project_root, mjcf_path)
        model = pin.buildModelFromMJCF(mjcf_path)
    else:
        from robot_descriptions.loaders.pinocchio import load_robot_description

        pin_robot = load_robot_description(robot_config["description"])
        model = pin_robot.model

    model.gravity.linear = np.asarray(gravity, dtype=float)

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    return model, cmodel, cdata


def build_pinocchio_euler_dynamics(cmodel, cdata, dt):
    """Discrete map x+ = x + dt * [v; ABA(q, v, tau)]."""
    nq, nv = cmodel.nq, cmodel.nv
    q = ca.SX.sym("q", nq)
    v = ca.SX.sym("v", nv)
    tau = ca.SX.sym("tau", nv)

    qdd = cpin.aba(cmodel, cdata, q, v, tau)
    x = ca.vertcat(q, v)
    xdot = ca.vertcat(v, qdd)
    x_next = x + xdot * dt
    return ca.Function("F_pin", [x, tau], [x_next])


def build_pinocchio_continuous_dynamics(cmodel, cdata):
    """Continuous map [q; v; tau] -> [v; ABA(q, v, tau)]."""
    nq, nv = cmodel.nq, cmodel.nv
    q = ca.SX.sym("q", nq)
    v = ca.SX.sym("v", nv)
    tau = ca.SX.sym("tau", nv)

    qdd = cpin.aba(cmodel, cdata, q, v, tau)
    x = ca.vertcat(q, v)
    xdot = ca.vertcat(v, qdd)
    return ca.Function("aba_f", [x, tau], [xdot])


class MujocoEulerDynamics(ca.Callback):
    """Forward-Euler discretization using MuJoCo forward dynamics."""

    def __init__(self, robot_config, dt, gravity=None, name="F_mujoco"):
        ca.Callback.__init__(self)
        self.robot = Manipulator(robot_config=robot_config, gravity=gravity)
        self.nq = self.robot.nq
        self.nv = self.robot.nv
        self.nx = 2 * self.nq
        self.nu = self.nv
        self.dt = float(dt)
        self.construct(name, {})

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return ca.Sparsity.dense(self.nx)
        return ca.Sparsity.dense(self.nu)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.nx)

    def eval(self, arg):
        x = np.array(arg[0].full(), dtype=float).reshape(self.nx)
        tau = np.array(arg[1].full(), dtype=float).reshape(self.nu)
        q = x[: self.nq]
        v = x[self.nq :]
        qdd = self.robot.forward_dynamics(q, v, tau)
        x_next = x + self.dt * np.concatenate([v, qdd])
        return [x_next]


class MujocoContinuousDynamics(ca.Callback):
    """Continuous dynamics [q; v] -> [v; qdd] using MuJoCo forward dynamics."""

    def __init__(self, robot_config, gravity=None, name="mujoco_aba"):
        ca.Callback.__init__(self)
        self.robot = Manipulator(robot_config=robot_config, gravity=gravity)
        self.nq = self.robot.nq
        self.nv = self.robot.nv
        self.nx = 2 * self.nq
        self.nu = self.nv
        self.construct(name, {})

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return ca.Sparsity.dense(self.nx)
        return ca.Sparsity.dense(self.nu)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.nx)

    def eval(self, arg):
        x = np.array(arg[0].full(), dtype=float).reshape(self.nx)
        tau = np.array(arg[1].full(), dtype=float).reshape(self.nu)
        q = x[: self.nq]
        v = x[self.nq :]
        qdd = self.robot.forward_dynamics(q, v, tau)
        return [np.concatenate([v, qdd])]


def _attach_callback(fn, callback):
    fn._mujoco_callback = callback
    return fn


def build_mujoco_continuous_dynamics(robot_config, gravity=None):
    callback = MujocoContinuousDynamics(robot_config=robot_config, gravity=gravity)
    x = ca.MX.sym("x", 2 * robot_config["nq"])
    u = ca.MX.sym("u", robot_config["nv"])
    return _attach_callback(ca.Function("aba_mujoco", [x, u], [callback(x, u)]), callback)


def build_mujoco_euler_dynamics(robot_config, dt, gravity=None):
    callback = MujocoEulerDynamics(robot_config=robot_config, dt=dt, gravity=gravity)
    x = ca.MX.sym("x", 2 * robot_config["nq"])
    u = ca.MX.sym("u", robot_config["nv"])
    return _attach_callback(ca.Function("F_mujoco", [x, u], [callback(x, u)]), callback)


def build_nominal_euler_dynamics(robot_config, dt, gravity=None, backend=None):
    backend = backend or mpc_dynamics_backend(robot_config)
    if backend == "mujoco":
        return build_mujoco_euler_dynamics(robot_config, dt, gravity=gravity)

    if backend != "pinocchio":
        raise ValueError(
            f"Unknown mpc_dynamics_backend '{backend}'. Expected 'pinocchio' or 'mujoco'."
        )

    _, cmodel, cdata = load_pinocchio_casadi(robot_config, gravity=gravity)
    return build_pinocchio_euler_dynamics(cmodel, cdata, dt)


def get_pinocchio_casadi_triplet(robot_config, gravity=None):
    """Return (pin_model, cmodel, cdata) for K-NODE + pinocchio nominal paths."""
    return load_pinocchio_casadi(robot_config, gravity=gravity)
