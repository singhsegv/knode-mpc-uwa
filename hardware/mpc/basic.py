"""
What is this:
=> Start a ros2 node without any concept of knode. Just plain MPC.
=> No data storing for stats
=> No trajectory tracking
=> Takes a cartesian xyz -> Run an acados based MPC controller to generate controls

Only has acados based MPC implementation.
"""
import os
import sys
import time
import json
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

from simulator import Manipulator
from acados_manipulator_mpc import AcadosManipulatorControl

import pink
from pink.tasks import FrameTask
from pink.limits import ConfigurationLimit, VelocityLimit


class MPCNode(Node):
    def __init__(self):
        super().__init__("mpc_node")
        
        self.current_target_pose = None
        self.create_subscription(Point, "/mpc/cartesian/target", self.target_pose_callback, 10)
        self.create_subscription(String, "/robot_description", self.urdf_callback, self.qos_profile)

        self.nq = 3
        self.dt = 0.01
        self.mpc_horizon = 10

        self.current_q = pinocchio.neutral(self.model)
        self.pink_config = pink.Configuration(self.model, self.model_data, self.current_q)

        self.plant = Manipulator(robot_config=robot_config, enable_viz=False)
        self.mpc = AcadosManipulatorControl(
            horizon=mpc_horizon, dt=dt, enable_knode=enable_knode,
            run_dir=knode_path, robot_config=robot_config,
        )

    def urdf_callback(self, msg):
        """
        TODO: Setup pinocchio and mpc model etc here
        """
        if self.model is not None:
            return
        
        try:
            self.get_logger().info("Received robot_description. Building Pinocchio model...")
            self.model = pinocchio.buildModelFromXML(msg.data)
            self.model_data = self.model.createData()

            self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)
            self.base_frame_id = self.model.getFrameId(self.base_frame_name)
            self.joint_names = [name for name in self.model.names if name != "universe"]
            self.current_q = pinocchio.neutral(self.model)

            self.get_logger().info(f"Model built successfully. Controllable joints: {len(self.joint_names)}")
        except Exception as e:
            self.get_logger().error(f"Failed to build Pinocchio model: {e}")

    def inverse_kinematics(self, robot, q_prev, pos_des):
        """
        pos_des: (3,) desired EE position
        vel_des: (3,) desired EE linear velocity
        """
        self.pink_config.update(self.current_q)

        pos_des = np.asarray(pos_des, dtype=float)
        task_dim = getattr(robot, "task_space_dim", 3)
        q_prev = np.asarray(q_prev, dtype=float)

        target_position = np.array([pos_des[0], pos_des[1], pos_des[2]])
        task = FrameTask("wrist_link", position_cost=1.0, orientation_cost=0.0)
        task.set_target(pinocchio.SE3(np.eye(3), target_position))

        velocity = pink.solve_ik(
            self.pink_config,
            [task],
            0.1,
            solver="daqp",
            limits=[ConfigurationLimit(self.model), VelocityLimit(self.model)],
        )
        
        q_des = self.pink_config.integrate(velocity, dt)
        qdot_des = np.zeros(3)

        return q_des, qdot_des

    def target_pose_callback(self, msg):
        self.current_target_pose = np.array([msg.x, msg.y, msg.z], dtype=float)
        self.get_logger().info(f"xyz = ({self.current_target_pose[0]:.4f}, {self.current_target_pose[1]:.4f}, {self.current_target_pose[2]:.4f})")

        self.track_pose(self.current_target_pose)

    def track_pose(self, target_pose_cartesian):
        # TODO: Read from a ros2 based plant
        x_current_state = np.array([0])
        X_ref = self.inverse_kinematics(q_prev, target_pose_cartesian)

        u_prev, x_prev = None, None
        U_optimal, X_optimal = mpc.solve(x_current_state, X_ref, u_guess=u_prev, x_guess=x_prev)

        u_actual = U_optimal[:, 0]
        # q_next, qdot_next = plant.step_rk4(x_current_state[:nq], x_current_state[nq:], u_actual, dt)
        execute(u_actual)

        x_current_state = np.concatenate([q_next.flatten(), qdot_next.flatten()])
        current_time += dt

        u_prev = np.zeros_like(U_optimal)
        u_prev[:, :-1] = U_optimal[:, 1:]
        u_prev[:, -1] = U_optimal[:, -1]

        x_prev = np.zeros_like(X_optimal)
        x_prev[:, 0] = x_current_state 
        x_prev[:, 1:-1] = X_optimal[:, 2:]
        x_prev[:, -1] = X_optimal[:, -1]
    

if __name__ == "__main__":
    # parser.add_argument("--config", type=str, default="ur5_basic", help="JSON filename under config/mpc/ (without .json)")
    # {
    #     "type": "circle",
    #     "duration": 10.0,
    #     "omega": 1.5,
    #     "cx": 0.47,
    #     "cy": 0.11,
    #     "cz": 0.42,
    #     "r": 0.06
    # }

    
    # parser.add_argument("--robot", type=str, default="ur5-mpc", help="Robot config name under config/robot/")
    # {
    #     "name": "ur5",
    #     "description": "ur5e_description",
    #     "ee_frame": "tool0",
    #     "nq": 6,
    #     "nv": 6,
    #     "default_q": [0.0, -1.5707963267948966, 1.5707963267948966, 0.0, 1.5707963267948966, 0.0],
    #     "plant_gravity": [0.0, 0.0, -9.81],
    #     "model_gravity": [0.0, 0.0, -9.81],
    #     "torque_limits": [150.0, 150.0, 150.0, 28.0, 28.0, 28.0],
    #     "kp": [200.0, 200.0, 150.0, 100.0, 80.0, 50.0],
    #     "kd": [20.0, 20.0, 15.0, 10.0, 8.0, 5.0],
    #     "task_space_dim": 3,
    #     "notes": "Perfect gravity model for mpc"
    # }

    # cfg["plant_gravity"] = np.asarray(cfg["plant_gravity"], dtype=float)
    # cfg["model_gravity"] = np.asarray(cfg["model_gravity"], dtype=float)

    # cfg["default_q"] = np.asarray(cfg["default_q"], dtype=float)
    # cfg["torque_limits"] = np.asarray(cfg["torque_limits"], dtype=float)

    # cfg["kp"] = np.asarray(cfg["kp"], dtype=float)
    # cfg["kd"] = np.asarray(cfg["kd"], dtype=float)
    # if cfg["kp"].shape[0] != cfg["nq"] or cfg["kd"].shape[0] != cfg["nq"]:
    #     raise ValueError(f"Robot config {name}: kp/kd length must match nq={cfg['nq']}")

    rclpy.init(args=args)
    node = MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
