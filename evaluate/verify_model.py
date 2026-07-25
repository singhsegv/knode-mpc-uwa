"""
Deeper into the validity of qddot predictions.

Run order (diagnose first, then detailed checks):
  1. diagnose_first          - state drift vs qdd error correlation
  2. evaluate_single_step    - qdd / residual on GT states
  3. evaluate_qddot_rollout  - qdd on open-loop states
"""
import os
import sys
import argparse
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import fsolve

from knode.model import PlanarArmKnodeModel
from robots.simulator import PlanarArm
from utils.data_utils import load_evaluation_data
import utils.model_utils as model_utils
from utils.model_utils import latest_checkpoint_epoch, load_model_checkpoint


DT = 0.002

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/debugging/"


def ee_fk(q):
    """Planar 2R FK; returns EE (x, y) in the arm plane."""
    q1, q2 = q
    L1, L2 = 1.0, 1.0
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return np.array([x, y])


def get_learned_residual(knode_model, q, qdot, tau):
    """Residual qddot from the trained network (normalized input, same as forward pass)."""
    knode_model.eval()
    with torch.no_grad():
        z = torch.tensor(np.concatenate([q, qdot, tau])[None, :], dtype=torch.float64)
        out = knode_model.func.residual_dynamics_network(knode_model.func.normalize_residual_input(z))
        return out[0].cpu().numpy()


def step_with_knode(model, q, qdot, tau, dt):
    """One RK4 step of PlanarArmKnodeModel: (q, qdot, tau) -> (q_next, qdot_next)."""
    model.eval()
    with torch.no_grad():
        y0 = torch.tensor(np.concatenate([q, qdot, tau], axis=0)[None, :], dtype=torch.float64)
        timesteps = torch.tensor([0.0, dt], dtype=torch.float64)
        pred = model(y0, timesteps)  # (2, 1, 6)
        nxt = pred[-1, 0].detach().cpu().numpy()
        
    return nxt[:2].copy(), nxt[2:4].copy()


def get_learned_qdd(knode_model, robot, q, qdot, tau):
    """KNODE qdd = imperfect nominal + learned residual."""
    _, qdd_imperfect = robot.get_imperfect_dynamics(q, qdot, tau)
    return qdd_imperfect + get_learned_residual(knode_model, q, qdot, tau)


def diagnose_first(data_path, knode_model):
    """
    Diagnose first (before chasing wrong fixes).

    Compare open-loop qdd error against state drift. If qdd error tracks state drift
    (low at start, grows as states diverge), the vector field is OK near data but bad
    off-manifold -> fix structure / horizon training, not integration or M invertibility.

    If qdd error is large even when state drift ~ 0, the vector field itself is wrong
    -> add direct qdd supervision or check nominal/residual definitions.
    """
    robot = PlanarArm(enable_viz=False)
    traj = load_evaluation_data(data_path)[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    controls = traj["controls"]
    n_steps = controls.shape[0]
    n_eval = n_steps - 1

    state_err_log = np.empty(n_eval)
    qdd_err_ol_log = np.empty(n_eval)
    qdd_err_gt_log = np.empty(n_eval)

    q_gt = controls[0, 1:3].copy()
    qdot_gt = controls[0, 3:5].copy()
    q_ol = q_gt.copy()
    qdot_ol = qdot_gt.copy()

    for i in range(1, n_steps):
        tau = controls[i - 1, 5:]
        step_idx = i - 1

        qdd_gt = robot.forward_dynamics(q_gt, qdot_gt, tau)
        qdd_knode_gt = get_learned_qdd(knode_model, robot, q_gt, qdot_gt, tau)
        qdd_knode_ol = get_learned_qdd(knode_model, robot, q_ol, qdot_ol, tau)

        state_err_log[step_idx] = np.linalg.norm(np.concatenate([q_ol - q_gt, qdot_ol - qdot_gt]))
        qdd_err_gt_log[step_idx] = np.linalg.norm(qdd_gt - qdd_knode_gt)
        qdd_err_ol_log[step_idx] = np.linalg.norm(qdd_gt - qdd_knode_ol)

        q_gt, qdot_gt = controls[i, 1:3].copy(), controls[i, 3:5].copy()
        q_ol, qdot_ol = step_with_knode(knode_model, q_ol, qdot_ol, tau, DT)

    time_s = np.arange(n_eval) * DT
    corr = np.corrcoef(state_err_log, qdd_err_ol_log)[0, 1]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(time_s, state_err_log, "k-", linewidth=1.5, label=r"$\|(q,\dot q)_\mathrm{ol} - (q,\dot q)_\mathrm{gt}\|$")
    axes[0].set_ylabel("state drift")
    axes[0].set_title("Open-loop state drift from ground truth")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].plot(time_s, qdd_err_gt_log, "C0-", linewidth=1.5, label=r"$\|\ddot q_\mathrm{gt} - \ddot q_\mathrm{knode}\|$ @ GT states")
    axes[1].plot(time_s, qdd_err_ol_log, "C3--", linewidth=1.5, label=r"$\|\ddot q_\mathrm{gt} - \ddot q_\mathrm{knode}\|$ @ OL states")
    axes[1].set_ylabel(r"$\ddot q$ error [rad/s$^2$]")
    axes[1].set_title("KNODE qdd error: GT replay vs open-loop evaluation point")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    sc = axes[2].scatter(state_err_log, qdd_err_ol_log, s=14, alpha=0.75, c=time_s, cmap="viridis")
    fig.colorbar(sc, ax=axes[2], label="time [s]")
    axes[2].set_xlabel(r"state drift $\|(q,\dot q)_\mathrm{ol} - (q,\dot q)_\mathrm{gt}\|$")
    axes[2].set_ylabel(r"open-loop $\ddot q$ error [rad/s$^2$]")
    axes[2].set_title(f"Correlation: r = {corr:.3f}")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "verify_diagnose_first.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved diagnose-first plot -> {save_path}")

    n_window = max(1, n_eval // 10)
    early = slice(0, n_window)
    late = slice(n_eval - n_window, n_eval)
    print("\n--- Diagnose first ---")
    print(f"early-step mean qdd err @ GT:  {qdd_err_gt_log[early].mean():.4e}  (single-step quality)")
    print(f"early-step mean qdd err @ OL:  {qdd_err_ol_log[early].mean():.4e}")
    print(f"late-step  mean qdd err @ OL:   {qdd_err_ol_log[late].mean():.4e}")
    print(f"late-step  mean state drift:   {state_err_log[late].mean():.4e}")
    print(f"corr(state drift, qdd err OL): {corr:.3f}")

    if qdd_err_gt_log[early].mean() < 0.5 and qdd_err_ol_log[late].mean() > 5 * qdd_err_gt_log[early].mean():
        print("-> qdd OK on GT, blows up with drift: off-manifold / compounding issue.")
    elif qdd_err_gt_log[early].mean() > 1.0:
        print("-> qdd bad even on GT: vector-field error; check residual / add qdd loss.")
    else:
        print("-> mixed; inspect plots and compare with stepwise eval in model_utils.")

    return {
        "state_err": state_err_log,
        "qdd_err_gt": qdd_err_gt_log,
        "qdd_err_ol": qdd_err_ol_log,
        "correlation": corr,
    }


def evaluate_single_step(data_path, knode_model=None):
    """
    Not Open Loop -> Just replaying with sanity checks?
    
    TODO: 2 different functions? Open loop with each?
    ----------------
    VERIFY
    ----------------
    qddot_gt = robot.forward_dynamics(q, qdot, tau)
    qddot_imperfect = robot.get_imperfect_dynamics(q, qdot, tau)
    delta_qddot_true = qddot_gt - qddot_imperfect

    # # Analytical verification of delta_qddot_true
    # M = robot.mass_matrix(q)
    # g = robot.gravity_forces(q)
    # delta_qddot_analytical = -np.linalg.solve(M, g)

    delta_qddot_learned = model.func.residual_dynamics_network([q, qdot, tau])
    qddot_learned = qddot_imperfect + delta_qddot_learned
    """
    robot = PlanarArm(enable_viz=False)
    traj = load_evaluation_data(data_path)[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = [] 
    n_steps = traj["des_ee"].shape[0]

    q = traj["controls"][0, 1:3].copy()
    qdot = traj["controls"][0, 3:5].copy()

    q_log = np.empty((n_steps, 2))
    n_residual_steps = n_steps - 1

    # ------------------------------------------------------------
    qddot_gt_log = np.empty((n_residual_steps, 2))
    qddot_imperfect_log = np.empty((n_residual_steps, 2))
    qddot_knode_log = np.empty((n_residual_steps, 2))

    delta_qddot_true_log = np.empty((n_residual_steps, 2))
    delta_qddot_learned_log = np.empty((n_residual_steps, 2))
    
    diff_log = np.empty(n_residual_steps)
    # ------------------------------------------------------------

    for i in range(1, n_steps):
        # robot.display(q)
        tau = traj["controls"][i - 1, 5:] # Correct index of tau makes/breaks this too

        # True
        qddot_gt = robot.forward_dynamics(q, qdot, tau)
        _, qddot_imperfect = robot.get_imperfect_dynamics(q, qdot, tau)
        delta_qddot_true = qddot_gt - qddot_imperfect

        delta_qddot_learned = get_learned_residual(knode_model, q, qdot, tau)
        qddot_knode = qddot_imperfect + delta_qddot_learned

        diff = np.linalg.norm(delta_qddot_true - delta_qddot_learned)
        
        step_idx = i - 1
        qddot_gt_log[step_idx] = qddot_gt
        qddot_imperfect_log[step_idx] = qddot_imperfect
        qddot_knode_log[step_idx] = qddot_knode
        delta_qddot_true_log[step_idx] = delta_qddot_true
        delta_qddot_learned_log[step_idx] = delta_qddot_learned
        diff_log[step_idx] = diff

        q_log[i] = q.copy()
        
        # q, qdot = step_with_knode(knode_model, q, qdot, tau, DT)
        q, qdot = traj["controls"][i, 1:3].copy(), traj["controls"][i, 3:5].copy()
        
        # ------------------- Code Sanity check -------------------
        # q_pred, qdot_pred = robot.step_torchdiffeq(q, qdot, tau, DT)
        # q, qdot = traj["controls"][i, 1:3].copy(), traj["controls"][i, 3:5].copy()
        # print("----------")
        # print(np.linalg.norm(q - q_pred))
        # print(np.linalg.norm(qdot - qdot_pred))
        # print("++++++++++")
        # # ---------------------------------------------------------
       

    time_s = np.arange(n_residual_steps) * DT

    fig_qddot, (ax_gt_j1, ax_gt_j2, ax_diff) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax_gt_j1.plot(time_s, qddot_gt_log[:, 0], "C0-", linewidth=1.5, label=r"$\ddot{q}_\mathrm{gt}$")
    ax_gt_j1.plot(time_s, qddot_imperfect_log[:, 0], "C2--", linewidth=1.5, label=r"$\ddot{q}_\mathrm{imperfect}$")
    ax_gt_j1.plot(time_s, qddot_knode_log[:, 0], "C1:", linewidth=1.5, label=r"$\ddot{q}_\mathrm{gt} + \Delta\ddot{q}_\mathrm{knode}$")
    ax_gt_j1.set_ylabel(r"$\ddot{q}_1$ [rad/s$^2$]")
    ax_gt_j1.set_title("Joint 1: ground-truth vs imperfect vs KNODE acceleration (Only 1 step predictions)")
    ax_gt_j1.grid(True, alpha=0.3)
    ax_gt_j1.legend(loc="upper right")

    ax_gt_j2.plot(time_s, qddot_gt_log[:, 1], "C0-", linewidth=1.5, label=r"$\ddot{q}_\mathrm{gt}$")
    ax_gt_j2.plot(time_s, qddot_imperfect_log[:, 1], "C2--", linewidth=1.5, label=r"$\ddot{q}_\mathrm{imperfect}$")
    ax_gt_j2.plot(time_s, qddot_knode_log[:, 1], "C1:", linewidth=1.5, label=r"$\ddot{q}_\mathrm{gt} + \Delta\ddot{q}_\mathrm{knode}$")
    ax_gt_j2.set_ylabel(r"$\ddot{q}_2$ [rad/s$^2$]")
    ax_gt_j2.set_title("Joint 2: ground-truth vs imperfect vs KNODE acceleration (Only 1 step predictions)")
    ax_gt_j2.grid(True, alpha=0.3)
    ax_gt_j2.legend(loc="upper right")

    ax_diff.plot(time_s, diff_log, "k-", linewidth=1.5, label=r"$\|\Delta_\mathrm{true} - \Delta_\mathrm{learned}\|$")
    ax_diff.set_xlabel("time [s]")
    ax_diff.set_ylabel("L2 error [rad/s$^2$]")
    ax_diff.set_title("Residual prediction error")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.legend(loc="upper right")

    fig_qddot.tight_layout()

    qddot_save_path = os.path.join(OUTPUT_DIR, "verify_qddot_comparison_single_step.png")
    fig_qddot.savefig(qddot_save_path, dpi=150)
    plt.close(fig_qddot)
    print(f"Saved qddot comparison plot -> {qddot_save_path}")

    results.append({
        "qddot_gt": qddot_gt_log,
        "qddot_imperfect": qddot_imperfect_log,
        "qddot_knode": qddot_knode_log,
        "delta_qddot_true": delta_qddot_true_log,
        "delta_qddot_learned": delta_qddot_learned_log,
        "diff": diff_log,
    })

    return results


def evaluate_qddot_rollout(data_path, knode_model):
    """Compare qddot on ground-truth replay vs KNODE open-loop rollout states."""
    robot = PlanarArm(enable_viz=False)
    traj = load_evaluation_data(data_path)[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_steps = traj["des_ee"].shape[0]
    n_qddot_steps = n_steps - 1
    qddot_gt_log = np.empty((n_qddot_steps, 2))
    qddot_knode_open_loop_log = np.empty((n_qddot_steps, 2))

    q_gt = traj["controls"][0, 1:3].copy()
    qdot_gt = traj["controls"][0, 3:5].copy()

    q_ol = q_gt.copy()
    qdot_ol = qdot_gt.copy()

    for i in range(1, n_steps):
        tau = traj["controls"][i - 1, 5:]
        step_idx = i - 1

        qddot_gt_log[step_idx] = robot.forward_dynamics(q_gt, qdot_gt, tau)
        _, qddot_imperfect = robot.get_imperfect_dynamics(q_ol, qdot_ol, tau)
        delta_qddot_learned = get_learned_residual(knode_model, q_ol, qdot_ol, tau)
        qddot_knode_open_loop_log[step_idx] = qddot_imperfect + delta_qddot_learned

        q_gt, qdot_gt = traj["controls"][i, 1:3].copy(), traj["controls"][i, 3:5].copy()
        q_ol, qdot_ol = step_with_knode(knode_model, q_ol, qdot_ol, tau, DT)

    time_s = np.arange(n_qddot_steps) * DT
    qddot_diff_log = np.linalg.norm(qddot_gt_log - qddot_knode_open_loop_log, axis=1)

    fig, (ax_j1, ax_j2, ax_diff) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax_j1.plot(time_s, qddot_gt_log[:, 0], "C0-", linewidth=1.5, label=r"$\ddot{q}_\mathrm{gt\ replay}$")
    ax_j1.plot(time_s, qddot_knode_open_loop_log[:, 0], "C0--", linewidth=1.5, label=r"$\ddot{q}_\mathrm{knode\ open loop}$")
    ax_j1.set_ylabel(r"$\ddot{q}_1$ [rad/s$^2$]")
    ax_j1.set_title("Joint 1: ground-truth replay vs KNODE open-loop acceleration")
    ax_j1.grid(True, alpha=0.3)
    ax_j1.legend(loc="upper right")

    ax_j2.plot(time_s, qddot_gt_log[:, 1], "C1-", linewidth=1.5, label=r"$\ddot{q}_\mathrm{gt\ replay}$")
    ax_j2.plot(time_s, qddot_knode_open_loop_log[:, 1], "C1--", linewidth=1.5, label=r"$\ddot{q}_\mathrm{knode\ open loop}$")
    ax_j2.set_xlabel("time [s]")
    ax_j2.set_ylabel(r"$\ddot{q}_2$ [rad/s$^2$]")
    ax_j2.set_title("Joint 2: ground-truth replay vs KNODE open-loop acceleration")
    ax_j2.grid(True, alpha=0.3)
    ax_j2.legend(loc="upper right")

    ax_diff.plot(time_s, qddot_diff_log, "k-", linewidth=1.5, label=r"$\|\ddot{q}_\mathrm{gt} - \ddot{q}_\mathrm{knode}\|$")
    ax_diff.set_xlabel("time [s]")
    ax_diff.set_ylabel("L2 error [rad/s$^2$]")
    ax_diff.set_title("Combined qddot rollout error")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.legend(loc="upper right")

    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "verify_qddot_rollout_comparison_openloop.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved qddot rollout comparison plot -> {save_path}")

    return {
        "qddot_gt_replay": qddot_gt_log,
        "qddot_knode_open_loop": qddot_knode_open_loop_log,
        "qddot_diff": qddot_diff_log,
    }


if __name__ == "__main__":
    data_dir = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/data/lala"

    model_cfg = {"step_size": DT, "nominal_only": False, "perfect_nominal": False, "hidden_layers": 2,  "hidden_neurons": 128, "activation": "tanh"}
    # model_cfg = {"step_size": DT, "nominal_only": True, "perfect_nominal": True, "hidden_layers": 2,  "hidden_neurons": 128, "activation": "tanh"}
    model = PlanarArmKnodeModel(model_cfg).to(torch.float64)
    
    epoch = 2100
    model_utils.CHECKPOINT_DIR = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/runs/20260723_013245_focussed_circle_basic1_focussed-circle-la-2/checkpoints"
    load_model_checkpoint(model, epoch)
    
    print(f"Loaded KNODE checkpoint epoch {epoch}")
    diagnose_first(data_dir, model)
    evaluate_single_step(data_dir, model)
    evaluate_qddot_rollout(data_dir, model)
