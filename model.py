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
    def __init__(self, robot, step_size):
        super().__init__()
        # TODO: Sketchy input and output shapes
        # TODO: Fix the hardcoded dimensions later

        self.robot = robot
        self.step_size = step_size

        self.residual_dynamics_network = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 6),
        )

        residual_last_layer = self.residual_dynamics_network[-1]
        nn.init.zeros_(residual_last_layer.weight)
        nn.init.zeros_(residual_last_layer.bias)

        self.nominal_only = False

    def compute_nominal(self, x, xdot, u):
        """
        FAAAAAK: Should return dz/dt, NOT THE NEXT STATE

        => With "step_rk4", the model should learn nothing.
        => With "imperfect_dynamics_step", the model should learn gravity torque thingy similar to pinocchio's
        """
        rows = x.shape[0]

        # DYNAMICS: dz/dt
        output = np.zeros((rows, 6))

        for idx in range(rows):
            x_np = x[idx].detach().cpu().numpy()
            xdot_np = xdot[idx].detach().cpu().numpy()
            u_np = u[idx].detach().cpu().numpy()

            qdot, qddot = self.robot.get_nominal_dynamics(x_np, xdot_np, u_np)
            # qdot, qddot = self.robot.get_imperfect_dynamics(x_np, xdot_np, u_np)
            
            # Controls are held constant -> output[idx, 4:] = 0
            output[idx, 0:2] = qdot
            output[idx, 2:4] = qddot

        output = torch.from_numpy(output).to(x.device, dtype=x.dtype)
        return output

    def forward(self, t, z):
        """
        FAAAAAK: Should return dz/dt, NOT THE NEXT STATE.

        NOTE: t is there due to torchdiffeq shizz. Not used.
        State Z (9x1) => [X, u] => [q1, q2, q3, qdot1, qdot2, qdot3, tau1, tau2, tau2]

        TODO: Do I need t here?
        TODO: Gradients flow is correct?
        """
        x, xdot, u = z[:, 0:2], z[:, 2:4], z[:, 4:]

        nominal_dynamics = self.compute_nominal(x, xdot, u)

        if self.nominal_only:
            return nominal_dynamics
        
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
        self.func = PlanarArmODEFunc(self.robot, step_size)
        self.step_size = step_size

    def forward(self, y0, timesteps):
        """TODO: 1st call to ODEFunc's forward has t = 0. Intended behavior??"""    
        z = odeint_adjoint(self.func, y0, timesteps, method="rk4", options={"step_size": self.step_size})
        return z
