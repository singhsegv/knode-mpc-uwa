import os

import torch
from torch import nn
from model import PlanarArmKnodeModel

import numpy as np
import matplotlib.pyplot as plt

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"


def checkpoint_path(epoch):
    return os.path.join(CHECKPOINT_DIR, f"neural_ode_epoch_{epoch}.pt")


def latest_checkpoint_epoch():
    if not os.path.isdir(CHECKPOINT_DIR):
        return None

    epochs = []
    for name in os.listdir(CHECKPOINT_DIR):
        if name.startswith("neural_ode_epoch_") and name.endswith(".pt"):
            epochs.append(int(name[len("neural_ode_epoch_"):-3]))
            
    return max(epochs) if epochs else None


def train_model(model, data):
    """
    Sensibly update every hyperparam
    """
    LOOKAHEAD = 2
    EPOCHS = 20000
    LR = 0.005

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    training_labels = []
    training_timesteps = None

    # Dataset generation shizz
    n_segments = len(data)
    for i in range(n_segments - LOOKAHEAD + 1):
        segment = torch.tensor(data[i:i+LOOKAHEAD, 1:], dtype=torch.float32)
        training_labels.append(segment)
    
    # TODO: Check if removing time from here is OKAY??
    # Initial conditions: training_labels should match these and 1 predicted step ahead label
    training_x = torch.tensor(data[:n_segments-LOOKAHEAD+1, 1:], dtype=torch.float32)
    # FAAAAAK: torchdiffeq uses physical time: one data index = step_size seconds (dt=0.01).
    # K-NODE's arange(lookahead) -> [0, 1] means "1 RK step", not "integrate to t=1".
    training_timesteps = torch.arange(LOOKAHEAD, dtype=torch.float32) * 0.002

    loss_arr = []
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # odeint returns (T, batch, 3); view(-1, 3) is time-major: all batch@t0, then all batch@t1.
        # stack(dim=1) on (2, 3) windows gives (T, batch, 3) — same layout as y_hat.
        # cat(dim=0) would be segment-major and pair the wrong rows in MSE.

        y_hat = model(training_x, training_timesteps).view(-1, 9)
        labels = torch.stack(training_labels, dim=1).reshape(-1, 9)

        loss = loss_fn(y_hat, labels)
        loss_arr.append(loss.item())
        
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        # if epoch % 250 == 0:
        #     with torch.no_grad():
        #         y0 = training_x[0, :]
        #         pred_timesteps = torch.arange(2000, dtype=torch.float) * model.step_size
        #         # pred_timesteps = torch.arange(8, dtype=torch.float) # Fucks up bad
        #         predictions = model(y0, pred_timesteps)
        #         plot_results(predictions, epoch)

        #     save_model_checkpoint(model, epoch)


def save_model_checkpoint(model, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path(epoch))


def load_model_checkpoint(model, epoch):
    model.load_state_dict(torch.load(checkpoint_path(epoch)))


def infer():
    latest = latest_checkpoint_epoch()
    if latest is not None:
        load_model_checkpoint(model, latest)
        start_epoch = latest + 1
        print(f"Resuming from epoch {start_epoch}")


def load_data(data_path):
    """
    Expected: ['t', 'q', 'qdot', 'tau', 'dt', 'trajectory', 'r', 'a', 'cx', 'cy', 'omega']
    """
    data = None
    with np.load(data_path) as data:
        t = data["t"][:, np.newaxis]
        q = data["q"]
        qdot = data["qdot"]
        tau = data["tau"]

        meta = data["meta"].item()

        data = np.hstack([t, q, qdot, tau])

    return data


if __name__ == "__main__":
    data_path = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/data/cartesian_circle.npz"
    data = load_data(data_path)

    model = PlanarArmKnodeModel()
    train_model(model, data)

    t_eval = torch.arange(len(y_train), dtype=torch.float) * model.step_size
    with torch.no_grad():
        y0 = y_train[0, :]
        y_hat = model(y0, t_eval)

    visualize_outputs(t_eval, y_hat)

