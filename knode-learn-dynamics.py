import os

import torch
from torch import nn
from model import PlanarArmKnodeModel

import numpy as np
from evaluate import evaluate
from visualize_trajectory import plot_trajectory_comparison

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


def train_model(model, data, test_data):
    """
    Sensibly update every hyperparam
    TODO: Dynamics lookahead that is useful enough shit for MPC
    """
    LOOKAHEAD = 2
    EPOCHS = 500
    LR = 0.005

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    training_labels = []
    training_timesteps = None

    # Dataset generation shizz
    n_segments = len(data)
    for i in range(n_segments - LOOKAHEAD + 1):
        segment = torch.tensor(data[i:i+LOOKAHEAD, 1:], dtype=torch.float64) # Removing the time from here, just the x, xdot, u
        training_labels.append(segment)
    
    # TODO: Check if removing time from here is OKAY??
    # Initial conditions: training_labels should match these and 1 predicted step ahead label
    training_x = torch.tensor(data[:n_segments-LOOKAHEAD+1, 1:], dtype=torch.float64)
    # FAAAAAK: torchdiffeq uses physical time: one data index = step_size seconds (dt=0.01).
    # K-NODE's arange(lookahead) -> [0, 1] means "1 RK step", not "integrate to t=1".
    training_timesteps = torch.arange(LOOKAHEAD, dtype=torch.float64) * 0.002

    loss_arr = []
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # odeint returns (T, batch, 3); view(-1, 3) is time-major: all batch@t0, then all batch@t1.
        # stack(dim=1) on (2, 3) windows gives (T, batch, 3) — same layout as y_hat.
        # cat(dim=0) would be segment-major and pair the wrong rows in MSE.

        y_hat = model(training_x, training_timesteps).view(-1, 6)
        labels = torch.stack(training_labels, dim=1).reshape(-1, 6)

        loss = loss_fn(y_hat, labels)
        loss_arr.append(loss.item())
        
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        # if epoch % 250 == 0:
        #     save_model_checkpoint(model, epoch)

        if epoch % 50 == 0:
            y_hat, loss = evaluate(model, test_data)
            plot_trajectory_comparison(
                y_hat,
                test_data,
                dt=0.002,
                epoch=epoch,
                max_steps=6000,
                show=False,
            )

    # y_hat, loss = evaluate(model, test_data)
    # plot_trajectory_comparison(
    #     y_hat,
    #     test_data,
    #     dt=0.002,
    #     epoch=1,
    #     max_steps=5500,
    #     show=False,
    # )


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
    Each file is expected to contain:
      - arrays: t, q, qdot, tau, dt
      - meta: dict with trajectory parameters
        
    data = np.hstack([t, q, qdot, tau])

    Returns "list" of concatenated [t, q, qdot, tau] rows across all trajectories -> 1 for each 20s rollout
    """
    def _load_npz_trajectory(path):
        with np.load(path, allow_pickle=True) as npz:
            t = npz["t"][:, np.newaxis]
            q = npz["q"]
            qdot = npz["qdot"]
            tau = npz["tau"]
            # meta = npz["meta"].item()
            trajectory = np.hstack([t, q, qdot, tau])

        # return trajectory, meta
        return trajectory

    def _list_npz_files(data_path):
        if os.path.isdir(data_path):
            npz_paths = sorted(
                os.path.join(data_path, name)
                for name in os.listdir(data_path)
                if name.endswith(".npz")
            )
            if not npz_paths:
                raise FileNotFoundError(f"No .npz files found in {data_path}")
            return npz_paths
        return [data_path]

    trajectories = []
    for path in _list_npz_files(data_path):
        trajectory = _load_npz_trajectory(path)
        trajectories.append(trajectory)

    # TODO: This doesn't make sense -> Use each rollout separately
    # combined_trajectories = np.vstack(trajectories)
    # return combined_trajectories
    return trajectories


if __name__ == "__main__":
    data_dir = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/data/trajectories_combo_1/"
    data = load_data(data_dir)

    model = PlanarArmKnodeModel().to(torch.float64)
    train_model(model, data[0], data[0])
    # train_model(model, data[0], data[2])
    # train_model(model, data[0], data[3])
    # train_model(model, data[0], data[4])
    # train_model(model, data[0], data[5])

