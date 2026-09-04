import os
import json
import shutil
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ROBOT_CONFIG_DIR = os.path.join(PROJECT_ROOT, "config", "robot")
SCENES_DIR = os.path.join(SCRIPT_DIR, "scenes")

import numpy as np
import mujoco

import torch
from torchdiffeq import odeint

from robot_descriptions import ur5e_mj_description


DEFAULT_ROBOT_NAME = "ur5"


def _menagerie_ur5e_dir():
    """Cached mujoco_menagerie universal_robots_ur5e directory (via robot_descriptions)."""
    return os.path.dirname(ur5e_mj_description.MJCF_PATH)


def load_mjcf_scene(scene_name):
    """
    Load a project scene XML that <include>s Menagerie ur5e.xml.

    Stages the scene next to symlinks of Menagerie ur5e.xml + assets/ in a
    temp directory so meshdir and include paths resolve without writing into
    the robot_descriptions cache.
    """
    scene_file = scene_name if scene_name.endswith(".xml") else f"{scene_name}.xml"
    src = os.path.join(SCENES_DIR, scene_file)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Scene MJCF not found: {src}")

    menagerie_dir = _menagerie_ur5e_dir()
    with tempfile.TemporaryDirectory(prefix="ur5e_scene_") as tmp:
        for name in ("ur5e.xml", "assets"):
            target = os.path.join(menagerie_dir, name)
            link = os.path.join(tmp, name)
            if not os.path.exists(target):
                raise FileNotFoundError(f"Menagerie asset missing: {target}")
            os.symlink(target, link)
        dest = os.path.join(tmp, scene_file)
        shutil.copy2(src, dest)
        return mujoco.MjModel.from_xml_path(dest)


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


def _resolve_ee_target(model, ee_frame):
    """Resolve end-effector body/site id from a config frame name."""
    for obj_type in (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_SITE):
        target_id = mujoco.mj_name2id(model, obj_type, ee_frame)
        if target_id >= 0:
            return obj_type, target_id

    raise ValueError(
        f"End-effector frame '{ee_frame}' not found in MuJoCo model as body or site."
    )


def make_robot(robot_name, enable_viz=False):
    """Build a Manipulator from a robot config name."""
    robot_config, robot_config_path = load_robot_config(robot_name)
    robot = Manipulator(robot_config=robot_config, enable_viz=enable_viz)
    return robot, robot_config, robot_config_path


class Manipulator:
    """
    MuJoCo-based dynamics shared by data generation, K-NODE nominal model, and MPC plant.
    TODO: Move to a full "position + orientation" space later.
    """

    def __init__(self, robot_config, enable_viz=False, gravity=None):
        self.robot_config = robot_config

        scene = robot_config.get("scene")
        if not scene:
            raise ValueError("Robot config must set 'scene' (e.g. 'ur5e_default' or 'ur5e_underwater').")
        self.mj_model = load_mjcf_scene(scene)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Backward-compatible aliases -> TODO: Should be usable with pinocchio/mujoco both
        self.model = self.mj_model
        self.data = self.mj_data

        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv

        ee_frame = robot_config["ee_frame"]
        self.ee_obj_type, self.ee_target_id = _resolve_ee_target(self.mj_model, ee_frame)
        self.task_space_dim = robot_config.get("task_space_dim", 3)

        if gravity is None:
            gravity = robot_config.get("plant_gravity", np.array([0.0, 0.0, -9.81]))

        self.mj_model.opt.gravity[:] = np.asarray(gravity, dtype=float)

        # Menagerie UR5e ships position-PD actuators (ctrl = q_des).
        # This stack injects joint torques via qfrc_applied / inverse_dynamics,
        # so disable built-in actuation or PD fights CTC and breaks ID↔FD.
        self.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION

        self._viewer = None
        if enable_viz:
            self._init_viewer()

    def _init_viewer(self):
        self._viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

    def _set_state(self, q, qdot):
        self.mj_data.qpos[:] = np.asarray(q, dtype=float)
        self.mj_data.qvel[:] = np.asarray(qdot, dtype=float)
        
    def _forward(self):
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _mass_matrix(self):
        mass_matrix = np.zeros((self.nv, self.nv), dtype=float)
        mujoco.mj_fullM(self.mj_model, self.mj_data, mass_matrix)
        return mass_matrix

    def mass_matrix(self, q):
        self._set_state(q, np.zeros(self.nv))
        self._forward()
        return self._mass_matrix()

    def bias_forces(self, q, qdot):
        self._set_state(q, qdot)
        self._forward()
        return np.array(self.mj_data.qfrc_bias[: self.nv])

    # TODO: Sketchy AF
    def gravity_forces(self, q):
        """Generalized gravity torque g(q)."""
        self._set_state(q, np.zeros(self.nv))
        self._forward()
        return np.array(self.mj_data.qfrc_bias[: self.nv])

    def inverse_dynamics(self, q, qdot, qddot):
        """tau = M(q) qddot + C(q, qdot) qdot + g(q)."""
        self._set_state(q, qdot)
        self._forward()
        qddot = np.asarray(qddot, dtype=float)
        return self._mass_matrix() @ qddot + self.mj_data.qfrc_bias[: self.nv]
    
    def forward_dynamics(self, q, qdot, tau):
        """q̈ from generalized torque tau."""
        self._set_state(q, qdot)
        self.mj_data.ctrl[:] = 0.0
        self.mj_data.qfrc_applied[:] = 0.0
        self.mj_data.xfrc_applied[:] = 0.0
        self.mj_data.qfrc_applied[: self.nv] = np.asarray(tau, dtype=float)
        self._forward()
        return self.mj_data.qacc.copy()

    def _body_forces_from_fext(self, fext):
        if fext is None:
            return None

        if isinstance(fext, dict):
            return {
                int(body_id): np.asarray(force, dtype=float)[:3]
                for body_id, force in fext.items()
            }

        body_forces = {}
        for body_id, wrench in enumerate(fext):
            if wrench is None:
                continue
            if hasattr(wrench, "linear"):
                force = np.asarray(wrench.linear, dtype=float)
            else:
                force = np.asarray(wrench, dtype=float)[:3]
            if np.allclose(force, 0.0):
                continue
            body_forces[int(body_id)] = force
        return body_forces or None

    # TODO: Imma sketchy, check me
    def ee_position(self, q):
        """End-effector position (x, y, z) from MuJoCo FK."""
        self._set_state(q, np.zeros(self.nv))
        self._forward()

        if self.ee_obj_type == mujoco.mjtObj.mjOBJ_SITE:
            return np.array(self.mj_data.site_xpos[self.ee_target_id])

        return np.array(self.mj_data.xpos[self.ee_target_id])

    def ee_positions(self, q_traj):
        """Map (N, nq) joint trajectory to (N, 3) EE positions."""
        xyz = np.empty((len(q_traj), self.task_space_dim))
        for i, q in enumerate(q_traj):
            xyz[i] = self.ee_position(q)
        return xyz

    # TODO: I'm sketchy and untested with mujoco setup
    def simulate_rollout(self, q0, qdot0, taus, dt, use_torchdiffeq=True, use_mujoco=False):
        """
        Rollout using a sequence of torques.
        Returns (N, nq) q and (N, nv) qdot logs (pre-step convention).

        By default this keeps the legacy torchdiffeq RK4 path. Set use_mujoco=True
        to integrate with MuJoCo's built-in integrator instead.
        """
        n = len(taus)
        q_log = np.empty((n, self.nq))
        qdot_log = np.empty((n, self.nv))

        q = np.asarray(q0, dtype=float).copy()
        qdot = np.asarray(qdot0, dtype=float).copy()
        for i in range(n):
            q_log[i] = q
            qdot_log[i] = qdot

            if use_mujoco:
                q, qdot = self.step_mujoco(q, qdot, taus[i], dt)
            elif use_torchdiffeq:
                q, qdot = self.step_torchdiffeq(q, qdot, taus[i], dt)
            else:
                q, qdot = self.step_rk4(q, qdot, taus[i], dt)

        return q_log, qdot_log

    def move_to(self, q):
        """Update frame placements for visualization / FK."""
        self._set_state(q, np.zeros(self.nv))
        self._forward()

    def ee_jacobian(self, q):
        """3 x nv position Jacobian for the end-effector frame."""
        self._set_state(q, np.zeros(self.nv))
        self._forward()

        jac_pos = np.zeros((3, self.nv))
        jac_rot = np.zeros((3, self.nv))
        if self.ee_obj_type == mujoco.mjtObj.mjOBJ_SITE:
            mujoco.mj_jacSite(self.mj_model, self.mj_data, jac_pos, jac_rot, self.ee_target_id)
        else:
            mujoco.mj_jacBody(self.mj_model, self.mj_data, jac_pos, jac_rot, self.ee_target_id)
            
        return jac_pos[: self.task_space_dim]

    def step_mujoco(self, q, qdot, tau, dt):
        """Single MuJoCo integration step with direct generalized torque injection."""
        self._set_state(q, qdot)
        self.mj_data.qfrc_applied[:] = 0.0
        self.mj_data.xfrc_applied[:] = 0.0
        self.mj_data.qfrc_applied[: self.nv] = np.asarray(tau, dtype=float)
        self.mj_model.opt.timestep = float(dt)
        mujoco.mj_step(self.mj_model, self.mj_data)
        return self.mj_data.qpos.copy(), self.mj_data.qvel.copy()

    def plant_step(self, q, qdot, tau, dt):
        """Integrate the plant one step using MuJoCo."""
        # if underwater is None:
        #     underwater = self.robot_config.get("underwater", {}).get("enable", False)
        # if underwater:
        #     return self.step_underwater_mujoco(q, qdot, tau, dt)
        return self.step_mujoco(q, qdot, tau, dt)

    def underwater_enabled(self):
        return self.robot_config.get("underwater", {}).get("enable", False)

    def step_rk4(self, q, qdot, tau, dt, fext=None):
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
        self.move_to(q)
        if self._viewer is not None:
            self._viewer.sync()

    def enable_visualization(self):
        if self._viewer is None:
            self._init_viewer()

    def get_random_config(self):
        q = np.empty(self.nq, dtype=float)
        for joint_id in range(self.nq):
            if self.mj_model.jnt_limited[joint_id]:
                low = self.mj_model.jnt_range[joint_id, 0]
                high = self.mj_model.jnt_range[joint_id, 1]
                q[joint_id] = np.random.uniform(low, high)
            else:
                q[joint_id] = np.random.uniform(-np.pi, np.pi)
        return q

    def get_default_config(self):
        if self.robot_config is not None:
            return self.robot_config["default_q"].copy()

        return np.zeros(self.nq)
