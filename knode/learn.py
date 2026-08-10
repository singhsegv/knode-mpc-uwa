import os

import numpy as np
import torch
from torch import nn

from knode.model import PlanarArmKnodeModel
from utils.data_utils import load_data
from utils.model_utils import save_model_checkpoint, evaluate_model, evaluate_horizon_loss
from utils.visualize_trajectory import plot_trajectory_comparison, save_pinocchio_simulation_video, horizon_losses

OUTPUT_DIR = "outputs"

LOOKAHEAD = 2
EPOCHS = 15000
LR = 0.001
QDD_PLOT_STEPS = 10000

OG_UNTRAINED_MODEL = None


def build_lookahead_windows(trajectories, lookahead):
    """
    One sliding window per valid start index, never crossing trajectory boundaries.
    Returns training_x (batch, 6) and training_labels list of (lookahead, 6) tensors.
    """
    training_labels = []
    ic_rows = [] # Combining all the shizz

    for traj in trajectories:
        states = traj[:, 1:]  # drop time -> (q, qdot, tau)
        n = len(traj)
        for i in range(n - lookahead + 1):
            ic_rows.append(states[i])
            training_labels.append(torch.tensor(states[i:i + lookahead], dtype=torch.float64))

    training_x = torch.tensor(np.stack(ic_rows), dtype=torch.float64)

    assert training_x.shape[0] == len(training_labels)
    return training_x, training_labels


def train_model(model, train_data, test_data):
    """
    Sensibly update every hyperparam
    TODO: Dynamics lookahead that is useful enough shit for MPC

    -------
    params:

    train_data: list/tuple of numpy arrays
    -------
    """

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    training_x, training_labels = build_lookahead_windows(train_data, LOOKAHEAD)
    labels_stacked = torch.stack(training_labels, dim=0)  # (batch, lookahead, 6)
    # ZOH: step k uses tau from the state at the start of that interval.
    step_torques = labels_stacked[:, :-1, 4:6]

    loss_arr = []
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Old Shizz
        # y_hat = model(training_x, training_timesteps).view(-1, 6)[:, :4]
        # labels = torch.stack(training_labels, dim=1).reshape(-1, 6)[:, :4]

        # odeint returns (T, batch, 6). Drop t=0 (IC copy); loss only on predicted steps.
        # stack(dim=1) on windows gives (T, batch, 6) — same layout as y_hat.

        # NOTE: No torques in loss; step_torques teacher-forces GT tau each step.
        y_hat = model(training_x, step_torques=step_torques)[1:].reshape(-1, 6)[:, :4]
        labels = labels_stacked[:, 1:, :4].reshape(-1, 4)
        
        loss = loss_fn(y_hat, labels)
        loss_arr.append(loss.item())
        
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        if epoch % 150 == 0:
            save_model_checkpoint(model, epoch)

            y_hat_open_loop, nominal_qdd_log_open_loop, residual_qdd_log_open_loop, total_qdd_log_open_loop = evaluate_model(model, test_data, open_loop=True)
            y_hat_stepwise, nominal_qdd_log_stepwise, residual_qdd_log_stepwise, total_qdd_log_stepwise = evaluate_model(model, test_data, open_loop=False)

            # For plotting purposes: Same shape (t, q, qdot, u) and numpy type
            t_col = torch.as_tensor(test_data[:, :1], dtype=y_hat_open_loop.dtype)

            y_hat_open_loop = torch.cat([t_col, y_hat_open_loop], dim=1).detach().numpy()
            y_hat_stepwise = torch.cat([t_col, y_hat_stepwise], dim=1).detach().numpy()

            test_data_steps = len(test_data)
            plot_trajectory_comparison(
                y_hat_open_loop, test_data, dt=model.step_size, label=f"{epoch}_openloop",
                max_steps=test_data_steps, show=False,
                nominal_qdd_log=nominal_qdd_log_open_loop,
                residual_qdd_log=residual_qdd_log_open_loop,
                total_qdd_log=total_qdd_log_open_loop,
                max_qdd_steps=QDD_PLOT_STEPS,
            )
            plot_trajectory_comparison(
                y_hat_stepwise, test_data, dt=model.step_size, label=f"{epoch}_stepwise",
                max_steps=test_data_steps, show=False,
                nominal_qdd_log=nominal_qdd_log_stepwise,
                residual_qdd_log=residual_qdd_log_stepwise,
                total_qdd_log=total_qdd_log_stepwise,
                max_qdd_steps=QDD_PLOT_STEPS,
            )

            # TODO: FIX US -> Not feasible using this
            # video_path = save_pinocchio_simulation_video(y_hat_open_loop, dt=model.step_size, fps=500, max_steps=test_data_steps, label=f"{epoch}_openloop")
            # video_path = save_pinocchio_simulation_video(y_hat_stepwise, dt=model.step_size, fps=30, max_steps=test_data_steps, label=f"{epoch}_stepwise")

            losses = evaluate_horizon_loss(model, test_data, [1, 5, 10, 25, 50, 100, 250, 500])
            horizon_losses(losses, label=f"{epoch}")


