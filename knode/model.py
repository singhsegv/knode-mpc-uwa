import torch
import numpy as np
from torch import nn
from torchdiffeq import odeint_adjoint

from robots.simulator import Manipulator, load_robot_config


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


def create_residual_mlp(input_dim, output_dim, hidden_layers, hidden_neurons, activation):
    """Build MLP: input -> (Linear+act) * hidden_layers -> Linear(output)."""
    act_name = activation.lower()

    if act_name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{activation}'. Expected one of {sorted(ACTIVATIONS)}")

    act_cls = ACTIVATIONS[act_name]
    layers = []
    in_features = input_dim

    layers.append(nn.Linear(in_features, hidden_neurons))
    layers.append(act_cls())

    for _ in range(hidden_layers):
        layers.append(nn.Linear(hidden_neurons, hidden_neurons))
        layers.append(act_cls())

    layers.append(nn.Linear(hidden_neurons, output_dim))
    return nn.Sequential(*layers)


class ManipulatorODEFunc(nn.Module):
    """Vector field for an n-DOF manipulator K-NODE model."""

    def __init__(self, robot, model_cfg):
        super().__init__()
        self.robot = robot
        self.nq = robot.nq
        self.nv = robot.nv
        self.state_dim = 3 * self.nq

        residual_input = model_cfg.get("residual_input", "q")
        if residual_input == "q":
            input_dim = self.nq
        elif residual_input in ("q_qdot", "q,qdot", "qv"):
            input_dim = 2 * self.nq
            residual_input = "q_qdot"
        else:
            raise ValueError(
                f"Unknown residual_input '{residual_input}'. Use 'q' or 'q_qdot'."
            )
        self.residual_input = residual_input

        self.residual_dynamics_network = create_residual_mlp(
            input_dim=input_dim,
            output_dim=self.nv,
            hidden_layers=model_cfg["hidden_layers"],
            hidden_neurons=model_cfg["hidden_neurons"],
            activation=model_cfg["activation"],
        )

        residual_last_layer = self.residual_dynamics_network[-1]
        nn.init.zeros_(residual_last_layer.weight)
        nn.init.zeros_(residual_last_layer.bias)

        self.nominal_only = model_cfg.get("nominal_only", False)
        self.perfect_nominal = model_cfg.get("perfect_nominal", False)

    def compute_nominal(self, x, xdot, u):
        """
        This part can/should be parallelized
        x and xdot naming is SKETCHY AF -> I mixed up mpc and this
        """
        rows = x.shape[0]
        output = np.zeros((rows, self.state_dim))

        for idx in range(rows):
            x_np = x[idx].detach().cpu().numpy()
            xdot_np = xdot[idx].detach().cpu().numpy()
            u_np = u[idx].detach().cpu().numpy()
            qdot, qddot = self.robot.get_nominal_dynamics(x_np, xdot_np, u_np)
            output[idx, : self.nq] = qdot
            output[idx, self.nq : 2 * self.nq] = qddot

        output = torch.from_numpy(output).to(x.device, dtype=x.dtype)
        return output

    def forward(self, t, z):
        """
        State z => [q, qdot, tau] with shape (batch, 3*nq).
        Returns dz/dt with torques held constant (d tau / dt = 0).
        """
        x = z[:, : self.nq]
        xdot = z[:, self.nq : 2 * self.nq]
        u = z[:, 2 * self.nq :]

        nominal_dynamics = self.compute_nominal(x, xdot, u)

        if self.nominal_only:
            return nominal_dynamics

        if self.residual_input == "q_qdot":
            residual_dynamics = self.residual_dynamics_network(torch.cat([x, xdot], dim=1))
        else:
            residual_dynamics = self.residual_dynamics_network(x)
        zeros_q = torch.zeros(z.shape[0], self.nq, dtype=z.dtype, device=z.device)
        zeros_tau = torch.zeros(z.shape[0], self.nq, dtype=z.dtype, device=z.device)
        residual_dynamics = torch.cat([zeros_q, residual_dynamics, zeros_tau], dim=1) # [qdot, qddot, tau_dot]
        dynamics = nominal_dynamics + residual_dynamics
        return dynamics


class ManipulatorKnodeModel(nn.Module):
    """Wrapper that integrates manipulator dynamics over time."""

    def __init__(self, model_cfg, robot_config):
        super().__init__()

        self.step_size = model_cfg.get("step_size", 0.01)
        self.nominal_only = model_cfg.get("nominal_only", False)
        self.perfect_nominal = model_cfg.get("perfect_nominal", False)

        self.robot_config = robot_config
        self.nq = robot_config["nq"]
        self.nv = robot_config["nv"]

        if self.perfect_nominal:
            self.robot = Manipulator(robot_config=robot_config, gravity=np.array([0.0, 0.0, -9.81]))
        else:
            self.robot = Manipulator(robot_config=robot_config, gravity=robot_config["model_gravity"])

        self.func = ManipulatorODEFunc(robot=self.robot, model_cfg=model_cfg)

    def _integrate_zoh(self, y0, step_torques):
        """
        Piecewise RK4 with zero-order-hold torques.
        step_torques: (batch, n_steps, nq)
        Returns (n_steps + 1, batch, 3*nq).
        """
        n_steps = step_torques.shape[1]
        step_times = torch.tensor([0.0, self.step_size], dtype=y0.dtype, device=y0.device)
        states = [y0]
        current = y0

        # TODO: Check our correct index in the GT stored data: [q, qdot, tau] should be correct
        tau_idx_start = 2 * self.nq
        tau_idx_end = 3 * self.nq
        for k in range(n_steps):
            current = current.clone()
            current[:, tau_idx_start:tau_idx_end] = step_torques[:, k, :]
            segment = odeint_adjoint(
                self.func, current, step_times,
                method="rk4", options={"step_size": self.step_size},
            )
            current = segment[-1]
            states.append(current)

        return torch.stack(states, dim=0)

    def forward(self, y0, timesteps=None, step_torques=None):
        if step_torques is not None:
            return self._integrate_zoh(y0, step_torques)

        if timesteps is None:
            raise ValueError("Provide timesteps or step_torques.")

        if len(timesteps) > 2:
            raise ValueError(
                "Continuous integration with frozen tau is wrong for lookahead > 2.\nPass step_torques with one tau per step.")

        return odeint_adjoint(self.func, y0, timesteps, method="rk4", options={"step_size": self.step_size})
