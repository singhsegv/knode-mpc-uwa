import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ROBOT_CONFIG_DIR = os.path.join(PROJECT_ROOT, "config", "robot")

import numpy as np
import pinocchio as pin

import torch
from torchdiffeq import odeint

from pinocchio.visualize import MeshcatVisualizer
from robot_descriptions.loaders.pinocchio import load_robot_description


DEFAULT_ROBOT_NAME = "ur5"


def load_robot_config(name=None):
    """Load robot config JSON and resolve model assets."""
    if name is None:
        name = DEFAULT_ROBOT_NAME

    path = os.path.join(ROBOT_CONFIG_DIR, f"{name}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Robot config not found: {path}")

    with open(path) as f:
        cfg = json.load(f)

    cfg = dict(cfg)

    cfg["plant_gravity"] = np.asarray(cfg["plant_gravity"], dtype=float)
    cfg["model_gravity"] = np.asarray(cfg["model_gravity"], dtype=float)

    cfg["default_q"] = np.asarray(cfg["default_q"], dtype=float)
    cfg["torque_limits"] = np.asarray(cfg["torque_limits"], dtype=float)

    cfg["kp"] = np.asarray(cfg["kp"], dtype=float)
    cfg["kd"] = np.asarray(cfg["kd"], dtype=float)
    if cfg["kp"].shape[0] != cfg["nq"] or cfg["kd"].shape[0] != cfg["nq"]:
        raise ValueError(f"Robot config {name}: kp/kd length must match nq={cfg['nq']}")

    return cfg, path


def make_robot(robot_name, enable_viz=False):
    """Build a Manipulator from a robot config name."""
    robot_config, robot_config_path = load_robot_config(robot_name)
    robot = Manipulator(robot_config=robot_config, enable_viz=enable_viz)
    return robot, robot_config, robot_config_path


class Manipulator:
    """
    Pinocchio-based dynamics shared by data generation, K-NODE nominal model, and MPC plant.
    TODO: Move to a full "position + orientation" space later.
    """
    def __init__(self, robot_config, enable_viz=False, gravity=None):
        self.robot_config = robot_config
        
        robot_urdf_model = robot_config["description"] # Uses robot_descriptions package now for easy peazy stuff
        pin_robot = load_robot_description(robot_urdf_model)

        self.model = pin_robot.model
        self.collision_model = pin_robot.collision_model
        self.visual_model = pin_robot.visual_model
        
        self.data = pin_robot.data
        self.nq = self.model.nq
        self.nv = self.model.nv

        ee_frame = robot_config["ee_frame"]
        self.ee_frame_id = self.model.getFrameId(ee_frame)
        self.task_space_dim = 3

        if gravity is None:
            gravity = robot_config.get("plant_gravity", np.array([0.0, 0.0, -9.81]))

        self.model.gravity.linear = np.asarray(gravity, dtype=float)

        self.viz = None
        if enable_viz:
            self._init_meshcat()

    def _init_meshcat(self):
        self.viz = MeshcatVisualizer(
            self.model, self.collision_model, self.visual_model,
        )
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()

    def mass_matrix(self, q):
        pin.crba(self.model, self.data, q)
        return np.array(self.data.M)

    def bias_forces(self, q, qdot):
        return pin.nle(self.model, self.data, q, qdot)

    def gravity_forces(self, q):
        """Generalized gravity torque g(q)."""
        return np.array(pin.computeGeneralizedGravity(self.model, self.data, q))

    def inverse_dynamics(self, q, qdot, qddot):
        """tau = M(q) qddot + C(q, qdot) qdot + g(q)."""
        return pin.rnea(self.model, self.data, q, qdot, qddot)

    def forward_dynamics(self, q, qdot, tau):
        return pin.aba(self.model, self.data, q, qdot, tau)

    def ee_position(self, q):
        """End-effector position (x, y, z) from Pinocchio FK."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return np.array(self.data.oMf[self.ee_frame_id].translation)

    def ee_positions(self, q_traj):
        """Map (N, nq) joint trajectory to (N, 3) EE positions."""
        xyz = np.empty((len(q_traj), self.task_space_dim))
        for i, q in enumerate(q_traj):
            xyz[i] = self.ee_position(q)
        return xyz

    def simulate_rollout(self, q0, qdot0, taus, dt, use_torchdiffeq=True):
        """
        Pinocchio RK4 rollout using a sequence of torques.
        Returns (N, nq) q and (N, nv) qdot logs (pre-step convention).
        """
        n = len(taus)
        q_log = np.empty((n, self.nq))
        qdot_log = np.empty((n, self.nv))

        q = np.asarray(q0, dtype=float).copy()
        qdot = np.asarray(qdot0, dtype=float).copy()
        for i in range(n):
            q_log[i] = q
            qdot_log[i] = qdot

            if use_torchdiffeq:
                q, qdot = self.step_torchdiffeq(q, qdot, taus[i], dt)
            else:
                q, qdot = self.step_rk4(q, qdot, taus[i], dt)

        return q_log, qdot_log

    def move_to(self, q):
        """Update frame placements for visualization / FK."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def ee_jacobian(self, q):
        """3 x nv position Jacobian for the end-effector frame."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jac = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.ee_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )
        return jac[: self.task_space_dim]

    def step_rk4(self, q, qdot, tau, dt):
        def rhs(q_state, qdot_state):
            qdd = self.forward_dynamics(q_state, qdot_state, tau)
            return qdot_state, qdd

        k1_q, k1_v = rhs(q, qdot)
        k2_q, k2_v = rhs(q + 0.5 * dt * k1_q, qdot + 0.5 * dt * k1_v)
        k3_q, k3_v = rhs(q + 0.5 * dt * k2_q, qdot + 0.5 * dt * k2_v)
        k4_q, k4_v = rhs(q + dt * k3_q, qdot + dt * k3_v)

        q_next = q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        qdot_next = qdot + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        return q_next, qdot_next

    def step_torchdiffeq(self, q, qdot, tau, dt):
        """Steps dynamics using torchdiffeq RK4 with zero-order hold on tau."""
        q = torch.tensor(q, dtype=torch.float64)
        qdot = torch.tensor(qdot, dtype=torch.float64)

        def ode_func(t, state):
            q_state, qdot_state = state
            q_state = q_state.detach().cpu().numpy()
            qdot_state = qdot_state.detach().cpu().numpy()
            qdd = self.forward_dynamics(q_state, qdot_state, tau)
            return (
                torch.tensor(qdot_state, dtype=torch.float64),
                torch.tensor(qdd, dtype=torch.float64),
            )

        t_span = torch.tensor([0.0, dt], dtype=q.dtype, device=q.device)
        initial_state = (q, qdot)
        trajectory = odeint(
            ode_func,
            initial_state,
            t_span,
            method="rk4",
            options={"step_size": dt},
        )

        q_next = trajectory[0][-1]
        qdot_next = trajectory[1][-1]
        return q_next.detach().cpu().numpy(), qdot_next.detach().cpu().numpy()

    def get_nominal_dynamics(self, q, qdot, tau):
        qdd = self.forward_dynamics(q, qdot, tau)
        return qdot, qdd

    def get_imperfect_dynamics(self, q, qdot, tau):
        tau_eff = tau + self.gravity_forces(q)
        qdd = self.forward_dynamics(q, qdot, tau_eff)
        return qdot, qdd

    def display(self, q):
        if self.viz is not None:
            self.viz.display(q)

    def enable_visualization(self):
        self._init_meshcat()

    def get_random_config(self):
        return pin.randomConfiguration(self.model)

    def get_default_config(self):
        if self.robot_config is not None:
            return self.robot_config["default_q"].copy()
        return np.zeros(self.nq)
