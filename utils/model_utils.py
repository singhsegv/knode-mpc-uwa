import os
import torch

from torch import nn
from robot import PlanarArm


CHECKPOINT_DIR = "checkpoints"
DTYPE = torch.float64


def checkpoint_path(epoch):
    return os.path.join(CHECKPOINT_DIR, f"neural_ode_epoch_{epoch}.pt")


def load_model_checkpoint(model, epoch):
    model.load_state_dict(torch.load(checkpoint_path(epoch)))


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


def evaluate_model(model, test_data):
    """
    STEPS:
    1. 
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

