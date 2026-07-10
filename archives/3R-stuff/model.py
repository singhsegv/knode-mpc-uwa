"""
TODO: Check everything here. All the shapes.
"""
import torch
import numpy as np
from torch import nn
from torchdiffeq import odeint_adjoint

from robot import PlanarArm


class PlanarArmODEFunc(nn.Module):
    """
    Vector field for the manipulator model.
    """
    def __init__(self, robot, step_size, input_shape, output_shape):
        super().__init__()
        # TODO: Sketchy input and output shapes
        # TODO: Fix the hardcoded dimensions later

        self.robot = robot
        self.step_size = step_size

        self.residual_dynamics_network = nn.Sequential(
            nn.Linear(9, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 9),
        )

        residual_last_layer = self.residual_dynamics_network[-1]
        nn.init.zeros_(residual_last_layer.weight)
        nn.init.zeros_(residual_last_layer.bias)

    def compute_nominal(self, x, xdot, u):
        """
        => With "step_rk4", the model should learn nothing.
        => With "imperfect_dynamics_step", the model should learn gravity torque thingy similar to pinocchio's
        """
        rows = x.shape[0]
        output = np.zeros((rows, 9))

        for idx in range(rows):
            x_np = x[idx].detach().cpu().numpy()
            xdot_np = xdot[idx].detach().cpu().numpy()
            u_np = u[idx].detach().cpu().numpy()

            # q_next, qdot_next = self.robot.step_rk4(x_np, xdot_np, u_np, self.step_size)
            q_next, qdot_next = self.robot.imperfect_dynamics_step(x_np, xdot_np, u_np, self.step_size)
            
            # Controls are held constant -> output[idx, 6:] = 0
            output[idx, 0:3] = q_next
            output[idx, 3:6] = qdot_next

        output = torch.from_numpy(output).to(x.device, dtype=x.dtype)
        return output

    def forward(self, t, z):
        """
        NOTE: t is there due to torchdiffeq shizz. Not used.
        State Z (9x1) => [X, u] => [q1, q2, q3, qdot1, qdot2, qdot3, tau1, tau2, tau2]

        TODO: Do I need t here?
        TODO: Gradients flow is correct?
        """
        x, xdot, u = z[:, 0:3], z[:, 3:6], z[:, 6:]

        with torch.no_grad():
            nominal_dynamics = self.compute_nominal(x, xdot, u)

        # dynamics = nominal_dynamics
        residual_dynamics = self.residual_dynamics_network(z)
        dynamics = nominal_dynamics + residual_dynamics

        return dynamics


class PlanarArmKnodeModel(nn.Module):
    """
    Wrapper that integrates the planar 3R arm dynamics over time.
    TODO: Check if my approach is good enough? Or use paper's approach for stability and shizz.

    TODO: RK4 step size etc should match.

    TODO: Batching and all is a shit show right now
    """
    def __init__(self, step_size=0.002):
        super().__init__()
        self.robot = PlanarArm()
        self.func = PlanarArmODEFunc(self.robot, step_size, input_shape=9, output_shape=6)
        self.step_size = step_size

    def forward(self, y0, timesteps):
        """TODO: 1st call to ODEFunc's forward has t = 0. Intended behavior??"""
        z = odeint_adjoint(self.func, y0, timesteps, method="rk4", options={"step_size": self.step_size})
        return z
