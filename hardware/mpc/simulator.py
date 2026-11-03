"""
Hardware plant for a running ros2_control stack.

Replaces the simulated step (RK4 / torchdiffeq). Each tick the MPC node
reads [q, qdot] and writes the first optimal torque. The controller
manager holds that torque until the next command.

Expects:
  - joint_state_broadcaster on `joint_states` (sensor_msgs/JointState)
  - forward_command_controller on the effort interface, command topic
    `forward_effort_controller/commands` (std_msgs/Float64MultiArray)

`joint_names` is the Pinocchio / MPC order. Torque vectors are in that
order. The effort command is published in `command_joint_names` order,
which must match the controller yaml.
"""
import threading

import numpy as np

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


_COMMAND_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class Ros2ControlPlant:
    def __init__(
        self,
        node,
        joint_names,
        command_joint_names=None,
        joint_states_topic="joint_states",
        effort_command_topic="forward_effort_controller/commands",
        velocity_filter_tau=0.05,
    ):
        """
        velocity_filter_tau: seconds. Differentiates position and low-passes
        it into qdot. Set to 0 to use JointState.velocity as published.
        """
        self._node = node
        self.joint_names = list(joint_names)
        self.nq = len(self.joint_names)
        if self.nq == 0:
            raise ValueError("joint_names is empty")

        if command_joint_names is None:
            command_joint_names = self.joint_names
        self.command_joint_names = list(command_joint_names)
        if len(self.command_joint_names) != self.nq:
            raise ValueError("command_joint_names must have one entry per joint")
        unknown = [name for name in self.command_joint_names if name not in self.joint_names]
        if unknown:
            raise ValueError(f"command joints not in joint_names: {unknown}")
        self._command_index = np.array(
            [self.joint_names.index(name) for name in self.command_joint_names],
            dtype=int,
        )

        self.velocity_filter_tau = float(velocity_filter_tau)

        self._lock = threading.Lock()
        self._q = None
        self._qdot = None
        self._effort = None
        self._q_prev = None
        self._t_prev = None
        self._qdot_filt = None

        self._state_sub = node.create_subscription(
            JointState,
            joint_states_topic,
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self._cmd_pub = node.create_publisher(
            Float64MultiArray,
            effort_command_topic,
            _COMMAND_QOS,
        )
        node.get_logger().info(
            f"plant: state '{joint_states_topic}', effort '{effort_command_topic}', "
            f"joints {self.joint_names}"
        )

    @property
    def ready(self):
        with self._lock:
            return self._q is not None

    def read_state(self):
        """Latest (q, qdot) in joint_names order."""
        with self._lock:
            if self._q is None:
                raise RuntimeError("no joint state received yet")
            return self._q.copy(), self._qdot.copy()

    def write_effort(self, tau):
        """Publish generalized torque. `tau` is in joint_names order."""
        tau = np.asarray(tau, dtype=float).reshape(-1)
        if tau.shape != (self.nq,):
            raise ValueError(f"tau length {tau.shape[0]} != nq {self.nq}")

        msg = Float64MultiArray()
        msg.data = tau[self._command_index].tolist()
        self._cmd_pub.publish(msg)

    def _on_joint_state(self, msg):
        index = {name: i for i, name in enumerate(msg.name)}
        if any(name not in index for name in self.joint_names):
            return

        idx = [index[name] for name in self.joint_names]
        q = np.asarray(msg.position, dtype=float)[idx]

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t <= 0.0:
            t = self._node.get_clock().now().nanoseconds * 1e-9

        if self.velocity_filter_tau > 0.0 and self._q_prev is not None:
            dt = t - self._t_prev
            if dt > 1e-6:
                dq = (q - self._q_prev + np.pi) % (2.0 * np.pi) - np.pi
                qdot_meas = dq / dt
                alpha = dt / (self.velocity_filter_tau + dt)
                qdot = (1.0 - alpha) * self._qdot_filt + alpha * qdot_meas
            else:
                qdot = self._qdot_filt
        elif len(msg.velocity) >= max(idx) + 1:
            qdot = np.asarray(msg.velocity, dtype=float)[idx]
        else:
            qdot = np.zeros(self.nq)

        effort = None
        if len(msg.effort) >= max(idx) + 1:
            effort = np.asarray(msg.effort, dtype=float)[idx]

        with self._lock:
            first = self._q is None
            self._q = q
            self._qdot = qdot
            self._effort = effort
            self._q_prev = q
            self._t_prev = t
            self._qdot_filt = qdot

        if first:
            self._node.get_logger().info("plant: first joint state received")
