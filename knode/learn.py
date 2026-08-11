import os

import numpy as np
import torch
from torch import nn

from knode.model import ManipulatorKnodeModel
from utils.data_utils import load_data
from utils.model_utils import save_model_checkpoint, evaluate_model, evaluate_horizon_loss
from utils.visualize_trajectory import plot_trajectory_comparison, horizon_losses, set_robot

OUTPUT_DIR = "outputs"

LOOKAHEAD = 2
EPOCHS = 15000
LR = 0.001
QDD_PLOT_STEPS = 10000


def build_lookahead_windows(trajectories, lookahead):
    """
    One sliding window per valid start index, never crossing trajectory boundaries.
    Returns training_x (batch, ...) and training_labels list of (lookahead, ...) tensors.
    """
    training_labels = []
    ic_rows = []

    for traj in trajectories:
        states = traj[:, 1:]
        n = len(traj)
        for i in range(n - lookahead + 1):
            ic_rows.append(states[i])
            training_labels.append(torch.tensor(states[i:i + lookahead], dtype=torch.float64))

    training_x = torch.tensor(np.stack(ic_rows), dtype=torch.float64)

    assert training_x.shape[0] == len(training_labels)
    return training_x, training_labels


def train_model(model, train_data, test_data, robot):
    """
    Train K-NODE with multi-step lookahead and zero-order-hold torques.
    Loss is computed on [q, qdot] only (first 2*nq dimensions).
    """
    set_robot(robot)

    # TODO: Change the way I'm calculated
    # 1. Wrapped q [-pi, pi]
    # 2. More weight on velocity
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    nq = model.nq
    knode_state_dim = 2 * nq # Since we only compare [q, qdot]

    training_x, training_labels = build_lookahead_windows(train_data, LOOKAHEAD)
    labels_stacked = torch.stack(training_labels, dim=0)
    
    # TODO: Check me for multi-step lookahead stuff: labels_stacked[:, 0, 2*nq:3*nq]
    step_torques = labels_stacked[:, :-1, 2 * nq : 3 * nq]

    loss_arr = []
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # NOTE: Model outputs the predicted [q, qdot, tau] and we extract [q, qdot] for the loss
        y_hat = model(training_x, step_torques=step_torques)[1:].reshape(-1, 3 * nq)[:, :knode_state_dim]
        labels = labels_stacked[:, 1:, :knode_state_dim].reshape(-1, knode_state_dim)

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

            t_col = torch.as_tensor(test_data[:, :1], dtype=y_hat_open_loop.dtype)

            y_hat_open_loop = torch.cat([t_col, y_hat_open_loop], dim=1).detach().numpy()
            y_hat_stepwise = torch.cat([t_col, y_hat_stepwise], dim=1).detach().numpy()

            test_data_steps = len(test_data)
            plot_trajectory_comparison(
                y_hat_open_loop, test_data, dt=model.step_size, label=f"{epoch}_openloop",
                max_steps=test_data_steps, show=False, robot=robot,
                nominal_qdd_log=nominal_qdd_log_open_loop,
                residual_qdd_log=residual_qdd_log_open_loop,
                total_qdd_log=total_qdd_log_open_loop,
                max_qdd_steps=QDD_PLOT_STEPS,
            )

            plot_trajectory_comparison(
                y_hat_stepwise, test_data, dt=model.step_size, label=f"{epoch}_stepwise",
                max_steps=test_data_steps, show=False, robot=robot,
                nominal_qdd_log=nominal_qdd_log_stepwise,
                residual_qdd_log=residual_qdd_log_stepwise,
                total_qdd_log=total_qdd_log_stepwise,
                max_qdd_steps=QDD_PLOT_STEPS,
            )

            losses = evaluate_horizon_loss(model, test_data, [1, 5, 10, 25, 50, 100, 250, 500])
            horizon_losses(losses, label=f"{epoch}")
