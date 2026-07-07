import os

import numpy as np
# import torch
# from torch import Tensor
# from torch.nn import functional as F

# from KNODE.models import QuadKNODEHybrid
# from KNODE.NODE import NeuralODE
# from KNODE.Utils.Solvers import RK
# from Simulation.quadrotor_params import quad_params, SIM_DT


DATA_DIR = '/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/data/'
MODEL_DIR = 'KNODE/SavedModels'

def load_data():
    """
    Expected: ['t', 'q', 'qdot', 'tau', 'dt', 'trajectory', 'r', 'a', 'cx', 'cy', 'omega']
    """
    data = None
    with np.load(DATA_DIR + "cartesian_circle.npz") as data:
        t = data["t"][:, np.newaxis]
        q = data["q"]
        qdot = data["qdot"]
        tau = data["tau"]

        data = np.hstack([t, q, qdot, tau])
        
    import pdb; pdb.set_trace()
    return data


def _build_ode(device='cpu'):
    func = QuadKNODEHybrid(quad_params, device).to(device)
    return NeuralODE(func, RK, STEP_SIZE=0.002)


def train_knode(
    train_paths,
    save_path,
    epochs=400,
    lr=1e-2,
    l2_lambda=5e-8,
    training_noise=1e-8,
    device='cpu',
    verbose=True,
):
    """Offline KNODE training with one-step-ahead rollout (Section III-D)."""
    ensure_dir(os.path.dirname(save_path))
    ode_train = _build_ode(device)
    optimizer = torch.optim.Adam(ode_train.parameters(), lr=lr)

    traj_list = []
    for path in train_paths:
        arr = np.load(path)
        traj = Tensor(arr).unsqueeze(1)
        traj = traj + torch.randn_like(traj) * np.sqrt(training_noise)
        traj_list.append(traj)

    lookahead = 2
    for epoch in range(epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        loss_tensor = None

        for true_traj in traj_list:
            n_segments = true_traj.size(0)
            segments = []
            for j in range(n_segments - lookahead + 1):
                segments.append(true_traj[j:j + lookahead])
            obs = torch.cat(segments, dim=1).to(device)

            pred_traj = []
            ode_input = obs[0, :, :].unsqueeze(1)
            pred_traj.append(ode_input)

            for k in range(lookahead - 1):
                z1 = ode_train(ode_input, Tensor([0.0, SIM_DT]).to(device)).squeeze(1)
                ode_input = torch.cat([z1[:, :13].unsqueeze(1), obs[k + 1, :, 13:].unsqueeze(1)], dim=2)
                pred_traj.append(ode_input)

            pred_traj = torch.transpose(torch.cat(pred_traj, dim=1), 0, 1)
            l2_norm = sum(p.pow(2.0).sum() for p in ode_train.parameters())
            loss = F.mse_loss(pred_traj[:, :, :13], obs[:, :, :13]) + l2_lambda * l2_norm
            epoch_loss += loss.item()
            loss_tensor = loss if loss_tensor is None else loss_tensor + loss

        loss_tensor.backward()
        optimizer.step()

        if epoch in (200, 350):
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.1

        if verbose and epoch % 50 == 0:
            print(f'Epoch {epoch:4d} | loss={epoch_loss:.6f}')

    torch.save({'ode_train': ode_train}, save_path)
    if verbose:
        print(f'Saved KNODE model to {save_path}')
    return ode_train


def main():
    load_data()

if __name__ == "__main__":
    main()
