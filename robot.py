import os

import numpy as np
import pinocchio as pin

import imageio.v2 as imageio
from pinocchio.visualize import MeshcatVisualizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF = os.path.join(SCRIPT_DIR, "planar-2R.urdf")

GRAVITY = np.array([0.0, -9.81, 0.0])


class PlanarArm:
    """
    Pinocchio based dynamics shared by data generation and the K-NODE nominal model.
    """
    def __init__(self, urdf_path=DEFAULT_URDF, enable_viz=False):
        """
        TODO: Update the code to visualize
        TODO: Double check the mujoco xml to urdf conversion
        """
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path)
        self.model.gravity.linear = GRAVITY
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("ee")
        
        self.viz = None
        if enable_viz:
            self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
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
        """End-effector (x, y) from Pinocchio FK."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return np.array(self.data.oMf[self.ee_frame_id].translation[:2])

    def ee_positions(self, q_traj):
        """Map (N, nq) joint trajectory to (N, 2) EE positions."""
        xy = np.empty((len(q_traj), 2))
        for i, q in enumerate(q_traj):
            xy[i] = self.ee_position(q)
        return xy

    def link_positions(self, q):
        """Base, elbow (joint 2), and EE positions in the plane."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        base = np.zeros(2)
        elbow = np.array(self.data.oMi[2].translation[:2])
        ee = np.array(self.data.oMf[self.ee_frame_id].translation[:2])
        return base, elbow, ee

    def simulate_rollout(self, q0, qdot0, taus, dt):
        """
        Pinocchio RK4 rollout using a sequence of torques.
        Returns (N, 2) q and (N, 2) qdot logs (pre-step convention, same as data_generator).
        """
        n = len(taus)
        q_log = np.empty((n, 2))
        qdot_log = np.empty((n, 2))

        q, qdot = np.asarray(q0, dtype=float).copy(), np.asarray(qdot0, dtype=float).copy()
        for i in range(n):
            q_log[i] = q
            qdot_log[i] = qdot
            q, qdot = self.step_rk4(q, qdot, taus[i], dt)

        return q_log, qdot_log

    def move_to(self, q):
        """Update frame placements for visualization / FK."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def ee_jacobian(self, q):
        """2 x 2 Jacobian for EE (x, y) in the arm plane."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jac = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.ee_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )

        # Linear jacobian for x, y and Angular jacobian across z axis
        return jac[0:2]

    def step_rk4(self, q, qdot, tau, dt):
        """
        NOTE: This AND `imperfect_dynamics_step` actually step the dynamics. Doesn't return the dx/dt
        TODO: Might be sketchy -> Check
        """
        def rhs(q_state, qdot_state):
            # print("------------------------")
            # print(self.forward_dynamics(q_state, qdot_state, tau + self.gravity_forces(q_state)))
            # print(self.forward_dynamics(q_state, qdot_state, tau))
            # print("------------------------")
            qdd = self.forward_dynamics(q_state, qdot_state, tau)
            return qdot_state, qdd

        # *_q is qdot, v is qddot
        k1_q, k1_v = rhs(q, qdot)
        k2_q, k2_v = rhs(q + 0.5 * dt * k1_q, qdot + 0.5 * dt * k1_v)
        k3_q, k3_v = rhs(q + 0.5 * dt * k2_q, qdot + 0.5 * dt * k2_v)
        k4_q, k4_v = rhs(q + dt * k3_q, qdot + dt * k3_v)

        q_next = q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        qdot_next = qdot + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        return q_next, qdot_next

    def imperfect_dynamics_step(self, q, qdot, tau, dt):
        """
        NOTE: This AND `step_rk4` actually step the dynamics. Doesn't return the dx/dt
        Removes the gravity from forward dynamics.
        """
        def rhs(q_state, qdot_state):
            # ABA solves M qdd = tau - C(q, qdot) qdot - g(q); add g(q) to drop gravity.
            qdd = self.forward_dynamics(q_state, qdot_state, tau + self.gravity_forces(q_state))
            return qdot_state, qdd

        # *_q is qdot, v is qddot
        k1_q, k1_v = rhs(q, qdot)
        k2_q, k2_v = rhs(q + 0.5 * dt * k1_q, qdot + 0.5 * dt * k1_v)
        k3_q, k3_v = rhs(q + 0.5 * dt * k2_q, qdot + 0.5 * dt * k2_v)
        k4_q, k4_v = rhs(q + dt * k3_q, qdot + dt * k3_v)

        q_next = q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        qdot_next = qdot + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        return q_next, qdot_next

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

