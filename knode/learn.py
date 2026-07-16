import os

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


def train_model(model, data, test_data):
    """
    Sensibly update every hyperparam
    TODO: Dynamics lookahead that is useful enough shit for MPC
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    training_labels = []
    training_timesteps = None

    # Dataset generation shizz
    n_segments = len(data) # Trajectory Length
    for i in range(n_segments - LOOKAHEAD + 1):
        segment = torch.tensor(data[i:i+LOOKAHEAD, 1:], dtype=torch.float64) # Removing the time from here, just the x, xdot, u
        training_labels.append(segment)
    
    # Initial conditions: training_labels should match these and 1 predicted step ahead label
    training_x = torch.tensor(data[:n_segments-LOOKAHEAD+1, 1:], dtype=torch.float64)
    training_timesteps = torch.arange(LOOKAHEAD, dtype=torch.float64) * 0.002

    # Residual net sees z-scored [q, qdot, tau]; Pinocchio still gets physical units.
    residual_inputs = torch.tensor(data[:, 1:], dtype=torch.float64)
    model.set_residual_input_stats(residual_inputs.mean(dim=0), residual_inputs.std(dim=0))

    loss_arr = []
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Old Shizz
        # y_hat = model(training_x, training_timesteps).view(-1, 6)[:, :4]
        # labels = torch.stack(training_labels, dim=1).reshape(-1, 6)[:, :4]

        # odeint returns (T, batch, 6). Drop t=0 (IC copy); loss only on predicted steps.
        # stack(dim=1) on windows gives (T, batch, 6) — same layout as y_hat.

        # NOTE: No torques in loss
        y_hat = model(training_x, training_timesteps)[1:].reshape(-1, 6)[:, :4]
        labels = torch.stack(training_labels, dim=1)[1:].reshape(-1, 6)[:, :4]
        
        loss = loss_fn(y_hat, labels)
        loss_arr.append(loss.item())
        
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        if epoch % 150 == 0:
            save_model_checkpoint(model, epoch)

            y_hat_open_loop = evaluate_model(model, test_data, open_loop=True)
            y_hat_stepwise = evaluate_model(model, test_data, open_loop=False)

            # For plotting purposes: Same shape (t, q, qdot, u) and numpy type
            t_col = torch.as_tensor(test_data[:, :1], dtype=y_hat_open_loop.dtype)

            y_hat_open_loop = torch.cat([t_col, y_hat_open_loop], dim=1).detach().numpy()
            y_hat_stepwise = torch.cat([t_col, y_hat_stepwise], dim=1).detach().numpy()

            test_data_steps = len(test_data)
            plot_trajectory_comparison(y_hat_open_loop, test_data, dt=model.step_size, label=f"{epoch}_openloop", max_steps=test_data_steps, show=False)
            plot_trajectory_comparison(y_hat_stepwise, test_data, dt=model.step_size, label=f"{epoch}_stepwise", max_steps=test_data_steps, show=False)

            # TODO: FIX US -> Not feasible using this
            # video_path = save_pinocchio_simulation_video(y_hat_open_loop, dt=model.step_size, fps=500, max_steps=test_data_steps, label=f"{epoch}_openloop")
            # video_path = save_pinocchio_simulation_video(y_hat_stepwise, dt=model.step_size, fps=30, max_steps=test_data_steps, label=f"{epoch}_stepwise")

            losses = evaluate_horizon_loss(model, test_data, [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000])
            horizon_losses(losses, label=f"{epoch}")

