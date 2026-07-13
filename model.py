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
    def __init__(self, robot, step_size, nominal_only=False, perfect_nominal=False):
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
        # nn.init.zeros_(residual_last_layer.weight)
        # nn.init.zeros_(residual_last_layer.bias)

        # Stats for residual net inputs [q, qdot, tau]. Nominal still uses physical z.
        self.register_buffer("residual_input_mean", torch.zeros(6, dtype=torch.float64))
        self.register_buffer("residual_input_std", torch.ones(6, dtype=torch.float64))

        self.nominal_only = nominal_only
        self.perfect_nominal = perfect_nominal

    def set_residual_input_stats(self, mean, std, eps=1e-8):
        mean = torch.as_tensor(mean, dtype=self.residual_input_mean.dtype, device=self.residual_input_mean.device)
        std = torch.as_tensor(std, dtype=self.residual_input_std.dtype, device=self.residual_input_std.device)

        if mean.shape != (6,) or std.shape != (6,):
            raise ValueError(f"Expected mean/std of shape (6,), got {tuple(mean.shape)} / {tuple(std.shape)}")
            
        self.residual_input_mean.copy_(mean)
        self.residual_input_std.copy_(std.clamp_min(eps))

    def normalize_residual_input(self, z):
        return (z - self.residual_input_mean) / self.residual_input_std

    def compute_nominal(self, x, xdot, u):
        """
        FAAAAAK: Should return dz/dt, NOT THE NEXT STATE

        => With "step_rk4", the model should learn nothing.
        => With "imperfect_dynamics_step", the model should learn gravity torque thingy similar to pinocchio's
        """
        rows = x.shape[0]
        output = np.zeros((rows, 6))

        for idx in range(rows):
            x_np = x[idx].detach().cpu().numpy()
            xdot_np = xdot[idx].detach().cpu().numpy()
            u_np = u[idx].detach().cpu().numpy()

            if self.perfect_nominal:
                qdot, qddot = self.robot.get_nominal_dynamics(x_np, xdot_np, u_np)
            else:
                qdot, qddot = self.robot.get_imperfect_dynamics(x_np, xdot_np, u_np)
            
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

        TODO: Gradients flow is correct?
        """
        x, xdot, u = z[:, 0:2], z[:, 2:4], z[:, 4:]

        nominal_dynamics = self.compute_nominal(x, xdot, u)

        if self.nominal_only:
            return nominal_dynamics

        residual_dynamics = self.residual_dynamics_network(self.normalize_residual_input(z))
        dynamics = nominal_dynamics + residual_dynamics
        return dynamics


class PlanarArmKnodeModel(nn.Module):
    """
    Wrapper that integrates the planar arm dynamics over time.

    TODO: Batching and all is a shit show right now
    """
    def __init__(self, step_size=0.002, nominal_only=False, perfect_nominal=False):
        """
        nominal_only: Nominal can be perfect/imperfect -> This flag just removes the residual
        perfect_nominal: Perfect or Imperfect nominal model
        """
        super().__init__()
        self.robot = PlanarArm()
        self.step_size = step_size

        self.func = PlanarArmODEFunc(
            robot=self.robot, step_size=step_size, 
            nominal_only=nominal_only, perfect_nominal=perfect_nominal
        )

    def set_residual_input_stats(self, mean, std, eps=1e-8):
        """Fit residual-input z-score from train columns [q, qdot, tau]."""
        self.func.set_residual_input_stats(mean, std, eps=eps)

    def forward(self, y0, timesteps):
        z = odeint_adjoint(self.func, y0, timesteps, method="rk4", options={"step_size": self.step_size})
        return z
