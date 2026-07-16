"""
Simulate/Plot matplotlib and videos on the desired (x,y) from the generated trajectory data.

Goal:
1. Visual differences between what the OG model generated, what the imperfect model does.
2. To test if the KNODE model has learnt the residual or not.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import fsolve

from model import PlanarArmKnodeModel
from robot import PlanarArm
from utils.data_utils import load_evaluation_data
from utils.model_utils import latest_checkpoint_epoch, load_model_checkpoint


DT = 0.002

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/"


def ee_fk(q):
    """Planar 2R FK; returns EE (x, y) in the arm plane."""
    q1, q2 = q
    L1, L2 = 1.0, 1.0
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return np.array([x, y])


def step_with_knode(model, q, qdot, tau, dt):
    """One RK4 step of PlanarArmKnodeModel: (q, qdot, tau) -> (q_next, qdot_next)."""
    model.eval()
    with torch.no_grad():
        y0 = torch.tensor(np.concatenate([q, qdot, tau], axis=0)[None, :], dtype=torch.float64)
        timesteps = torch.tensor([0.0, dt], dtype=torch.float64)
        pred = model(y0, timesteps)  # (2, 1, 6)
        nxt = pred[-1, 0].detach().cpu().numpy()
        
    return nxt[:2].copy(), nxt[2:4].copy()


def plot_evaluation_trajectories(dynamics_model):
    """
    dynamics_model: {Pinocchio Model / KNODE}

    Compare:
    1. Ground Truth: Desired EE positions in the Task Space
    2. Ground Truth: Desired EE positions in the Task Space
    """
    # Desired task-space EE position

    # Real task-space EE position [Dynamics from the Model]
    pass


def plot_evaluation_simulations():
    """
    Simulation video for sanity checks
    """
    pass


def evaluate_shit(data_path, knode_model=None):
    """
    Open-Loop replay CTC tracking on saved desired EE.

    data_path: directory or .npz with des_ee_* (+ recorded controls for ICs)
    inverse_dynamics_fn: callable(q, qdot, qdd_fb) -> tau
        e.g. robot.inverse_dynamics, or a neural wrapper with the same signature
    """
    robot = PlanarArm(enable_viz=True)
    traj = load_evaluation_data(data_path)[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    des_ee = traj["des_ee"]
    n_steps = des_ee.shape[0]

    q = traj["controls"][0, 1:3].copy()
    qdot = traj["controls"][0, 3:5].copy()

    # For EE position plotting purposes
    q_log = np.empty((n_steps, 2))

    for i in range(1, n_steps):
        robot.display(q)

        # x, y, xdot, ydot, xddot, yddot = des_ee[i]
        tau = traj["controls"][i - 1, 5:] # Correct index of tau makes/breaks this shit too

        q_log[i] = q.copy()
        if knode_model is not None:
            q, qdot = step_with_knode(knode_model, q, qdot, tau, DT)
        else:
            q, qdot = robot.step_torchdiffeq(q, qdot, tau, DT)

        # Debugging with the ground truth
        # q, qdot = traj["controls"][i, 1:3], traj["controls"][i, 3:5]

    ee_des = des_ee[:, :2]
    ee_hat = robot.ee_positions(q_log)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ee_des[:, 0], ee_des[:, 1], "r--", linewidth=1.5, label="desired EE")
    ax.plot(ee_hat[:, 0], ee_hat[:, 1], "b-", linewidth=1.5, label="rolled-out EE")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"EE trajectory")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"eval_ee_trajectory.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved EE plot -> {save_path}")

    results.append({"q": q_log, "ee_des": ee_des, "ee_hat": ee_hat})

    return results


if __name__ == "__main__":
    data_dir = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/data/evaluation1/"

    model = PlanarArmKnodeModel(step_size=DT, nominal_only=False, perfect_nominal=True).to(torch.float64)

    epoch = latest_checkpoint_epoch()
    if epoch is None:
        raise FileNotFoundError(
            f"No checkpoints found under {os.path.join(SCRIPT_DIR, 'checkpoints')}"
        )

    load_model_checkpoint(model, epoch)
    
    print(f"Loaded KNODE checkpoint epoch {epoch}")
    # evaluate_shit(data_dir)
    evaluate_shit(data_dir, model)
