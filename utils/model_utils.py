import os
import torch

from torch import nn


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
    # model.load_state_dict(torch.load(checkpoint_path(epoch)))
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
    nq = model.nq
    knode_state_dim = 3 * nq

    nominal_qdd_log = []
    residual_qdd_log = []
    total_qdd_log = []

    with torch.no_grad():
        model.eval()

        test_data = torch.tensor(test_data, dtype=DTYPE)
        eval_labels = test_data[:, 1:]

        dt = model.step_size
        step_times = torch.tensor([0.0, dt], dtype=DTYPE)
        N = eval_labels.shape[0]

        y_hat = torch.zeros(N, knode_state_dim, dtype=DTYPE)
        y_hat[0] = eval_labels[0]

        current_state = eval_labels[0].unsqueeze(0)
        for t in range(1, N):
            pred = model(current_state, step_times)
            predicted_state = pred[-1, 0].clone()
            predicted_state[2 * nq : 3 * nq] = eval_labels[t, 2 * nq : 3 * nq] # changing the torque to the correct next one
            y_hat[t] = predicted_state
            
            x = current_state[:, :nq]
            xdot = current_state[:, nq : 2 * nq]
            u = current_state[:, 2 * nq :]

            nominal_qddot = model.func.compute_nominal(x, xdot, u)

            if getattr(model.func, "residual_input", "q") == "q_qdot":
                residual_temp = model.func.residual_dynamics_network(torch.cat([x, xdot], dim=1))
            else:
                residual_temp = model.func.residual_dynamics_network(x)

            zeros_q = torch.zeros(current_state.shape[0], nq, dtype=current_state.dtype, device=current_state.device)
            zeros_tau = torch.zeros(current_state.shape[0], nq, dtype=current_state.dtype, device=current_state.device)
            residual_qddot = torch.cat([zeros_q, residual_temp, zeros_tau], dim=1)
            total_qddot = nominal_qddot + residual_qddot

            nominal_qdd_log.append(nominal_qddot.cpu().numpy())
            residual_qdd_log.append(residual_qddot.cpu().numpy())
            total_qdd_log.append(total_qddot.cpu().numpy())

            if open_loop:
                current_state = predicted_state.unsqueeze(0)
            else:
                current_state = eval_labels[t].unsqueeze(0)

        if open_loop:
            print("[TEST DATASET] Whole Trajectory Open-Loop")
        else:
            print("[TEST DATASET] Whole Trajectory Single-Steps")

        comp_idx = 2 * nq # dim(q) + dim(qdot)
        pred = y_hat[1:, 0:comp_idx]
        gt = test_data[1:, 1:1+comp_idx]
        err = pred - gt

        names = [f"q{i+1} [rad]" for i in range(nq)] + [f"qdot{i+1} [rad/s]" for i in range(nq)]
        mse_dim = (err ** 2).mean(dim=0)
        mae_dim = err.abs().mean(dim=0)

        rmse_dim = mse_dim.sqrt()
        gt_rms = (gt ** 2).mean(dim=0).sqrt().clamp_min(1e-12)
        nrmse_pct = 100.0 * (rmse_dim / gt_rms)

        print(f"Combined MSE (q+qdot): {mse_dim.mean().item():.6e}")
        print(f"{'dim':<16} {'MAE':>12} {'RMSE':>12} {'MSE':>12} {'NRMSE (%age of signal)':>12}")

        for i, name in enumerate(names):
            print(
                f"{name:<16} {mae_dim[i].item():12.6e} {rmse_dim[i].item():12.6e} "
                f"{mse_dim[i].item():12.6e} {nrmse_pct[i].item():12.6e}"
            )

    return y_hat, nominal_qdd_log, residual_qdd_log, total_qdd_log


def evaluate_horizon_loss(model, test_data, horizon):
    """
    Open loop losses: Horizon vs Loss
    horizon (steps): list of step indices to report MSE over [1..h]

    Returns dict {h: {"total", "q1", ...}} of MSE values.
    """
    nq = model.nq
    knode_state_dim = 3 * nq
    comp_idx = 2 * nq # dim(q) + dim(qdot)

    dt = model.step_size

    with torch.no_grad():
        model.eval()

        loss_fn = nn.MSELoss()

        test_data = torch.tensor(test_data, dtype=DTYPE)
        eval_labels = test_data[:, 1:]

        step_times = torch.tensor([0.0, dt], dtype=DTYPE)
        N = eval_labels.shape[0]

        y_hat = torch.zeros(N, knode_state_dim, dtype=DTYPE)
        y_hat[0] = eval_labels[0]

        current_state = eval_labels[0].unsqueeze(0)
        for t in range(1, N):
            pred = model(current_state, step_times)
            predicted_state = pred[-1, 0].clone()
            predicted_state[2 * nq : 3 * nq] = eval_labels[t, 2 * nq : 3 * nq]
            y_hat[t] = predicted_state
            current_state = predicted_state.unsqueeze(0)

        dim_names = [f"q{i+1}" for i in range(nq)] + [f"qdot{i+1}" for i in range(nq)]
        losses = {}
        for h in horizon:
            h = int(h)
            if h < 1 or h >= N:
                raise ValueError(f"horizon step {h} out of range for trajectory length N={N}")

            pred = y_hat[1:h + 1, 0:comp_idx]
            gt = test_data[1:h+1, 1:1+comp_idx]
            mse_dim = ((pred - gt) ** 2).mean(dim=0)

            loss_h = {"total": loss_fn(pred, gt)}
            for i, name in enumerate(dim_names):
                loss_h[name] = mse_dim[i]
                
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

