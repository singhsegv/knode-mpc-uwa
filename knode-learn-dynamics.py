import os

import torch
from torch import nn
from model import PlanarArmKnodeModel

from utils.data_utils import load_data
from utils.model_utils import save_model_checkpoint, evaluate_model

import numpy as np
from visualize_trajectory import plot_trajectory_comparison

OUTPUT_DIR = "outputs"

LOOKAHEAD = 2
EPOCHS = 5000
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

        # odeint returns (T, batch, 3); view(-1, 3) is time-major: all batch@t0, then all batch@t1.
        # stack(dim=1) on (2, 3) windows gives (T, batch, 3) — same layout as y_hat.
        # cat(dim=0) would be segment-major and pair the wrong rows in MSE.

        # NOTE: No torques in loss 
        y_hat = model(training_x, training_timesteps).view(-1, 6)[:, :4]
        labels = torch.stack(training_labels, dim=1).reshape(-1, 6)[:, :4]

        loss = loss_fn(y_hat, labels)
        loss_arr.append(loss.item())
        
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        if epoch % 250 == 0:
            save_model_checkpoint(model, epoch)

        if epoch % 250 == 0:
            y_hat, loss = evaluate_model(model, test_data)
            plot_trajectory_comparison(
                y_hat,
                test_data,
                dt=0.002,
                epoch=epoch,
                max_steps=n_segments,
                show=False,
            )

    # y_hat, loss = evaluate_model(model, test_data)
    # plot_trajectory_comparison(
    #     y_hat,
    #     test_data,
    #     dt=0.002,
    #     epoch=2,
    #     max_steps=10000,
    #     show=False,
    # )


if __name__ == "__main__":
    data_dir = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/data/trajectories_combo_1/"
    data = load_data(data_dir)

    model = PlanarArmKnodeModel(
        step_size=0.002, nominal_only=False, perfect_nominal=True
    ).to(torch.float64)

    # TODO: Better dataset
    train_data = data[0]
    test_data = data[0]

    train_model(model, train_data, test_data)

