import os
import torch

from torch import nn
from robots.simulator import PlanarArm


CHECKPOINT_DIR = None # NOTE: Modified by run_experiment.py
DTYPE = torch.float64


def checkpoint_path(epoch):
    return os.path.join(CHECKPOINT_DIR, f"neural_ode_epoch_{epoch}.pt")


def use_run_checkpoints(run_dir):
    """Point CHECKPOINT_DIR at `<run_dir>/checkpoints`."""
    global CHECKPOINT_DIR
    CHECKPOINT_DIR = os.path.join(run_dir, "checkpoints")
    return CHECKPOINT_DIR


def resolve_checkpoint_epoch(epoch=-1):
    """
    Resolve a checkpoint epoch.
    epoch=-1 -> latest checkpoint in CHECKPOINT_DIR; else require neural_ode_epoch_{epoch}.pt.
    """
    if CHECKPOINT_DIR is None:
        raise ValueError("CHECKPOINT_DIR is not set. Call use_run_checkpoints(run_dir) first.")

    if epoch == -1:
        resolved = latest_checkpoint_epoch()
        if resolved is None:
            raise FileNotFoundError(f"No checkpoints found under {CHECKPOINT_DIR}")
        return resolved

    path = checkpoint_path(epoch)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return epoch


def load_checkpoint_state_dict(epoch=-1):
    """Load a checkpoint state_dict. epoch=-1 -> latest."""
    resolved = resolve_checkpoint_epoch(epoch)
    state_dict = torch.load(checkpoint_path(resolved), map_location="cpu")
    return state_dict, resolved


def load_state_dict_from_run(run_dir, epoch=-1):
    """Set CHECKPOINT_DIR from run_dir and load state_dict."""
    use_run_checkpoints(run_dir)
    return load_checkpoint_state_dict(epoch)


def load_model_checkpoint(model, epoch):
    state_dict, _ = load_checkpoint_state_dict(epoch)
    model.load_state_dict(state_dict)


def save_model_checkpoint(model, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path(epoch))


def infer():
    latest = latest_checkpoint_epoch()
    if latest is not None:
        load_model_checkpoint(model, latest)
        start_epoch = latest + 1
        print(f"Resuming from epoch {start_epoch}")


def latest_checkpoint_epoch():
    if not os.path.isdir(CHECKPOINT_DIR):
        return None

    epochs = []
    for name in os.listdir(CHECKPOINT_DIR):
        if name.startswith("neural_ode_epoch_") and name.endswith(".pt"):
            epochs.append(int(name[len("neural_ode_epoch_"):-3]))
            
    return max(epochs) if epochs else None


def evaluate_model(model, test_data, open_loop=True):
    """
    open_loop=True -> Full open loop rollout comparison.
    open_loop=False -> Single step rollout comparison.
    """
    with torch.no_grad():
        model.eval()

        test_data = torch.tensor(test_data, dtype=DTYPE)
        eval_labels = test_data[:, 1:] # Converting from (t, q, qdot, u) to (q, qdot, u)

        dt = model.step_size
        step_times = torch.tensor([0.0, dt], dtype=DTYPE)
        N = eval_labels.shape[0]
        
        y_hat = torch.zeros(N, 6, dtype=DTYPE)
        y_hat[0] = eval_labels[0]

        current_state = eval_labels[0].unsqueeze(0) # (1, 6)
        for t in range(1, N):
            # TODO: Needs to be changed for multi step lookaheads comparisons
            pred = model(current_state, step_times) # (2, 1, 6) ODE thingy for timesteps
            predicted_state = pred[-1, 0].clone() # (6,)
            predicted_state[4:6] = eval_labels[t, 4:6] # Model gives q, qdot; torque comes from dataset at time t
            y_hat[t] = predicted_state

            # NOTE: This is the only difference
            if open_loop:
                current_state = predicted_state.unsqueeze(0) 
            else:
                current_state = eval_labels[t].unsqueeze(0) 

        if open_loop:
            print(f"[TEST DATASET] Whole Trajectory Open-Loop")
        else:
            print(f"[TEST DATASET] Whole Trajectory Single-Steps")
        
        pred = y_hat[1:, 0:4]
        gt = test_data[1:, 1:5]
        err = pred - gt
        
        names = ["q1 [rad]", "q2 [rad]", "qdot1 [rad/s]", "qdot2 [rad/s]"]
        mse_dim = (err ** 2).mean(dim=0)
        mae_dim = err.abs().mean(dim=0)

        # Relative to GT magnitude (avoid /0)
        rmse_dim = mse_dim.sqrt()
        gt_rms = (gt ** 2).mean(dim=0).sqrt().clamp_min(1e-12)
        nrmse_dim = rmse_dim / gt_rms
        nrmse_pct = 100.0 * (rmse_dim / gt_rms) # RMSE Error % in terms of the signal scale

        print(f"Combined MSE (q+qdot): {mse_dim.mean().item():.6e}")
        print(f"{'dim':<16} {'MAE':>12} {'RMSE':>12} {'MSE':>12} {'NRMSE (%age of signal)':>12}")

        for i, name in enumerate(names):
            print(
                f"{name:<16} {mae_dim[i].item():12.6e} {rmse_dim[i].item():12.6e} "
                f"{mse_dim[i].item():12.6e} {nrmse_pct[i].item():12.6e}"
            )

        # Grouped: angles vs velocities
        # mse_q = mse_dim[:2].mean()
        # mse_qd = mse_dim[2:].mean()
        # print(f"MSE(q):    {mse_q.item():.6e}  [rad^2]")
        # print(f"MSE(qdot): {mse_qd.item():.6e}  [(rad/s)^2]")
    
    return y_hat


def evaluate_horizon_loss(model, test_data, horizon):
    """
    Open loop losses: Horizon vs Loss
    horizon (steps): list of step indices to report MSE over [1..h]

    Returns dict {h: {"total", "q1", "q2", "qdot1", "qdot2"}} of MSE values.
    """
    if horizon is None:
        horizon = [1, 50, 100]

    with torch.no_grad():
        loss_fn = nn.MSELoss()
        model.eval()

        test_data = torch.tensor(test_data, dtype=DTYPE)

        # Converting from (t, q, qdot, u) to (q, qdot, u)
        eval_labels = test_data[:, 1:]

        dt = model.step_size
        step_times = torch.tensor([0.0, dt], dtype=DTYPE)
        N = eval_labels.shape[0]
        
        y_hat = torch.zeros(N, 6, dtype=DTYPE)
        y_hat[0] = eval_labels[0]

        current_state = eval_labels[0].unsqueeze(0) # (1, 6)
        for t in range(1, N):
            pred = model(current_state, step_times) # (2, 1, 6) ODE thingy for timesteps
            predicted_state = pred[-1, 0].clone() # (6,)
            predicted_state[4:6] = eval_labels[t, 4:6] # Model gives q, qdot; torque comes from dataset at time t
            y_hat[t] = predicted_state
            current_state = predicted_state.unsqueeze(0) # For the next iteration

        # Open-loop MSE for each requested horizon (steps 1..h inclusive)
        dim_names = ["q1", "q2", "qdot1", "qdot2"]
        losses = {}
        for h in horizon:
            h = int(h)
            if h < 1 or h >= N:
                raise ValueError(f"horizon step {h} out of range for trajectory length N={N}")

            pred = y_hat[1:h + 1, 0:4]
            gt = test_data[1:h + 1, 1:5]
            mse_dim = ((pred - gt) ** 2).mean(dim=0)

            loss_h = {
                "total": loss_fn(pred, gt),
                "q1": mse_dim[0],
                "q2": mse_dim[1],
                "qdot1": mse_dim[2],
                "qdot2": mse_dim[3],
            }

            losses[h] = loss_h
    
    return losses


def OG_evaluate_model(model, test_data):
    """
    Kept for history. All the changes I did.
    """
    loss_fn = nn.MSELoss()
    robot = PlanarArm()
    # model.func.nominal_only = True

    with torch.no_grad():
        model.eval()

        test_data = torch.tensor(test_data, dtype=DTYPE)

        # Converting from (t, q, qdot, u) to (q, qdot, u)
        eval_labels = test_data[:, 1:]

        dt = 0.002
        step_times = torch.tensor([0.0, dt], dtype=DTYPE)
        N = eval_labels.shape[0]
        
        y_hat = torch.zeros(N, 6, dtype=DTYPE)
        y_hat[0] = eval_labels[0]

        current_state = eval_labels[0].unsqueeze(0) # (1, 6)
        pinn_current_state = eval_labels[0].unsqueeze(0) # (1, 6)
        for t in range(1, N):
            # gt_compare = eval_labels[t, :].squeeze(0)

            # -----------------------------------------
            # DEBUGGING: Propagation through robot's rk4
            # -----------------------------------------
            # pinn_state = pinn_current_state.squeeze(0).detach().cpu().numpy()
            # pinn_robo_pred = robot.step_rk4(pinn_state[:2], pinn_state[2:4], pinn_state[4:], dt)

            # pinn_comparable = torch.hstack([torch.tensor(pinn_robo_pred[0][0:4]), torch.tensor(pinn_robo_pred[1][0:4])])
            # pinn_current_state = torch.hstack([pinn_comparable, eval_labels[t, 4:6]]).unsqueeze(0) # For the next iteration

            # print(f"pinocchio err: {pinn_comparable - gt_compare}")

            # -----------------------------------------
            # Propagation through torchdiffeq's odeint
            # -----------------------------------------
            pred = model(current_state, step_times) # (2, 1, 6) ODE thingy for timesteps
            predicted_state = pred[-1, 0].clone() # (6,)
            predicted_state[4:6] = eval_labels[t, 4:6] # Model gives q, qdot; torque comes from dataset at time t
            y_hat[t] = predicted_state
            current_state = predicted_state.unsqueeze(0) # For the next iteration

            # print(f"err: {current_state.squeeze(0) - gt_compare}")

        loss = loss_fn(y_hat[1:, 0:4], test_data[1:, 1:5])
        print(f"[TEST DATASET] Whole Trajectory Open-loop MSE: {loss.item()}")
    
    return y_hat, loss

