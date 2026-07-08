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

    def move_to(self, q):
        """Just updating the frame placements stuff for easy peazy beginning."""
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
        TODO: Might be sketchy -> Check
        """
        def rhs(q_state, qdot_state):
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

    def display(self, q):
        if self.viz is not None:
            self.viz.display(q)

