"""
Step 4: Inverse dynamics consistency (RNEA test).

If qdd_pred is physically realizable, Pinocchio RNEA should reconstruct the applied torque:

    tau_recon = M(q) qdd_pred + C(q, qdot) qdot + g(q)
    err = ||tau_recon - tau||

Runs on GT state replay and open-loop rollout states. GT forward dynamics should give
~zero error (sanity baseline); learned qdd error reveals non-physical accelerations.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import matplotlib.pyplot as plt
import numpy as np
import torch

from knode.model import PlanarArmKnodeModel
from robots.simulator import PlanarArm
from utils.data_utils import load_evaluation_data
import utils.model_utils as model_utils
from utils.model_utils import load_model_checkpoint

from verify_model import DT, OUTPUT_DIR, get_learned_residual, step_with_knode


def get_learned_qdd(knode_model, robot, q, qdot, tau):
    """KNODE qdd = imperfect nominal + learned residual."""
    _, qdd_imperfect = robot.get_imperfect_dynamics(q, qdot, tau)
    delta_learned = get_learned_residual(knode_model, q, qdot, tau)
    return qdd_imperfect + delta_learned


def rnea_consistency_error(robot, q, qdot, tau, qdd):
    """
    Inverse dynamics consistency: tau_recon = rnea(q, qdot, qdd) vs applied tau.
    Returns L2 norm and per-joint signed error.
    """
    tau_recon = robot.inverse_dynamics(q, qdot, qdd)
    err_vec = tau_recon - tau
    return np.linalg.norm(err_vec), err_vec, tau_recon


def evaluate_rnea_consistency(data_path, knode_model):
    """
    RNEA test on GT replay and open-loop states.

    Baselines:
      - qdd_gt from forward dynamics  -> err ~ 0 (sanity)
      - qdd_imperfect                 -> non-zero err (missing gravity)
      - qdd_knode (learned)           -> main test
    """
    robot = PlanarArm(enable_viz=False)
    traj = load_evaluation_data(data_path)[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    controls = traj["controls"]
    n_steps = controls.shape[0]
    n_eval = n_steps - 1

    err_gt_qdd_log = np.empty(n_eval)
    err_imperfect_log = np.empty(n_eval)
    err_knode_gt_log = np.empty(n_eval)
    err_knode_ol_log = np.empty(n_eval)

    err_knode_gt_joint_log = np.empty((n_eval, 2))
    err_knode_ol_joint_log = np.empty((n_eval, 2))
    tau_log = np.empty((n_eval, 2))
    tau_recon_knode_gt_log = np.empty((n_eval, 2))
    tau_recon_knode_ol_log = np.empty((n_eval, 2))

    state_err_ol_log = np.empty(n_eval)

    q_gt = controls[0, 1:3].copy()
    qdot_gt = controls[0, 3:5].copy()
    q_ol = q_gt.copy()
    qdot_ol = qdot_gt.copy()

    for i in range(1, n_steps):
        tau = controls[i - 1, 5:]
        step_idx = i - 1
        tau_log[step_idx] = tau

        qdd_gt = robot.forward_dynamics(q_gt, qdot_gt, tau)
        _, qdd_imperfect = robot.get_imperfect_dynamics(q_gt, qdot_gt, tau)
        qdd_knode_gt = get_learned_qdd(knode_model, robot, q_gt, qdot_gt, tau)
        qdd_knode_ol = get_learned_qdd(knode_model, robot, q_ol, qdot_ol, tau)

        err_gt_qdd_log[step_idx], _, _ = rnea_consistency_error(robot, q_gt, qdot_gt, tau, qdd_gt)
        err_imperfect_log[step_idx], _, _ = rnea_consistency_error(robot, q_gt, qdot_gt, tau, qdd_imperfect)
        err_knode_gt_log[step_idx], err_knode_gt_joint_log[step_idx], tau_recon_knode_gt_log[step_idx] = (
            rnea_consistency_error(robot, q_gt, qdot_gt, tau, qdd_knode_gt)
        )
        err_knode_ol_log[step_idx], err_knode_ol_joint_log[step_idx], tau_recon_knode_ol_log[step_idx] = (
            rnea_consistency_error(robot, q_ol, qdot_ol, tau, qdd_knode_ol)
        )

        state_err_ol_log[step_idx] = np.linalg.norm(
            np.concatenate([q_ol - q_gt, qdot_ol - qdot_gt])
        )

        q_gt, qdot_gt = controls[i, 1:3].copy(), controls[i, 3:5].copy()
        q_ol, qdot_ol = step_with_knode(knode_model, q_ol, qdot_ol, tau, DT)

    time_s = np.arange(n_eval) * DT

    # --- Plot 1: time series on GT replay ---
    fig_gt, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(time_s, err_gt_qdd_log, "C0-", linewidth=1.5, label=r"$\ddot q_\mathrm{gt}$ (sanity)")
    axes[0].plot(time_s, err_imperfect_log, "C2--", linewidth=1.5, label=r"$\ddot q_\mathrm{imperfect}$")
    axes[0].plot(time_s, err_knode_gt_log, "C1:", linewidth=1.5, label=r"$\ddot q_\mathrm{knode}$")
    axes[0].set_ylabel(r"$\|\tau_\mathrm{recon} - \tau\|$ [Nm]")
    axes[0].set_title("RNEA consistency at GT states")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(time_s, err_knode_gt_joint_log[:, 0], "C0-", linewidth=1.5, label=r"joint 1")
    axes[1].plot(time_s, err_knode_gt_joint_log[:, 1], "C1-", linewidth=1.5, label=r"joint 2")
    axes[1].axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
    axes[1].set_ylabel(r"$\tau_\mathrm{recon} - \tau$ [Nm]")
    axes[1].set_title("Per-joint RNEA error for KNODE qdd at GT states")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(time_s, tau_log[:, 0], "C0-", linewidth=1.2, label=r"$\tau_1$ applied")
    axes[2].plot(time_s, tau_recon_knode_gt_log[:, 0], "C0--", linewidth=1.2, label=r"$\tau_1$ recon (knode)")
    axes[2].plot(time_s, tau_log[:, 1], "C1-", linewidth=1.2, label=r"$\tau_2$ applied")
    axes[2].plot(time_s, tau_recon_knode_gt_log[:, 1], "C1--", linewidth=1.2, label=r"$\tau_2$ recon (knode)")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("[Nm]")
    axes[2].set_title("Applied vs RNEA-reconstructed torque (KNODE qdd, GT states)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", ncol=2, fontsize=8)

    fig_gt.tight_layout()
    gt_path = os.path.join(OUTPUT_DIR, "verify_rnea_gt_replay.png")
    fig_gt.savefig(gt_path, dpi=150)
    plt.close(fig_gt)
    print(f"Saved RNEA GT replay plot -> {gt_path}")

    # --- Plot 2: open loop vs GT replay ---
    fig_ol, axes_ol = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes_ol[0].plot(time_s, err_knode_gt_log, "C0-", linewidth=1.5, label="KNODE @ GT states")
    axes_ol[0].plot(time_s, err_knode_ol_log, "C3--", linewidth=1.5, label="KNODE @ open-loop states")
    axes_ol[0].set_ylabel(r"$\|\tau_\mathrm{recon} - \tau\|$ [Nm]")
    axes_ol[0].set_title("RNEA consistency: GT replay vs open-loop rollout")
    axes_ol[0].grid(True, alpha=0.3)
    axes_ol[0].legend(loc="upper right")

    axes_ol[1].plot(time_s, state_err_ol_log, "k-", linewidth=1.5, label=r"$\|(q,\dot q)_\mathrm{ol} - (q,\dot q)_\mathrm{gt}\|$")
    axes_ol[1].set_ylabel("state error")
    axes_ol[1].set_title("Open-loop state drift (for correlation with RNEA error)")
    axes_ol[1].grid(True, alpha=0.3)
    axes_ol[1].legend(loc="upper right")

    axes_ol[2].scatter(state_err_ol_log, err_knode_ol_log, s=12, alpha=0.7, c=time_s, cmap="viridis")
    cbar = fig_ol.colorbar(axes_ol[2].collections[0], ax=axes_ol[2], label="time [s]")
    axes_ol[2].set_xlabel(r"open-loop state error $\|(q,\dot q)_\mathrm{ol}-(q,\dot q)_\mathrm{gt}\|$")
    axes_ol[2].set_ylabel(r"RNEA error $\|\tau_\mathrm{recon}-\tau\|$ [Nm]")
    axes_ol[2].set_title("RNEA error vs state drift (open loop)")
    axes_ol[2].grid(True, alpha=0.3)

    fig_ol.tight_layout()
    ol_path = os.path.join(OUTPUT_DIR, "verify_rnea_openloop.png")
    fig_ol.savefig(ol_path, dpi=150)
    plt.close(fig_ol)
    print(f"Saved RNEA open-loop plot -> {ol_path}")

    # --- Plot 3: histograms ---
    fig_hist, axes_hist = plt.subplots(1, 3, figsize=(12, 4))

    axes_hist[0].hist(err_gt_qdd_log, bins=25, color="C0", alpha=0.85, edgecolor="white")
    axes_hist[0].set_xlabel(r"$\|\tau_\mathrm{recon}-\tau\|$ [Nm]")
    axes_hist[0].set_title(r"GT $\ddot q$ (want ~0)")
    axes_hist[0].grid(True, alpha=0.3)

    axes_hist[1].hist(err_knode_gt_log, bins=25, color="C1", alpha=0.85, edgecolor="white")
    axes_hist[1].set_xlabel(r"$\|\tau_\mathrm{recon}-\tau\|$ [Nm]")
    axes_hist[1].set_title("KNODE qdd @ GT states")
    axes_hist[1].grid(True, alpha=0.3)

    axes_hist[2].hist(err_knode_ol_log, bins=25, color="C3", alpha=0.85, edgecolor="white")
    axes_hist[2].set_xlabel(r"$\|\tau_\mathrm{recon}-\tau\|$ [Nm]")
    axes_hist[2].set_title("KNODE qdd @ open-loop states")
    axes_hist[2].grid(True, alpha=0.3)

    fig_hist.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, "verify_rnea_histograms.png")
    fig_hist.savefig(hist_path, dpi=150)
    plt.close(fig_hist)
    print(f"Saved RNEA histograms -> {hist_path}")

    # --- Plot 4: tau scatter ---
    fig_scatter, (ax_gt, ax_ol) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax_gt.scatter(tau_log[:, 0], tau_recon_knode_gt_log[:, 0], s=10, alpha=0.6, label=r"$\tau_1$")
    ax_gt.scatter(tau_log[:, 1], tau_recon_knode_gt_log[:, 1], s=10, alpha=0.6, label=r"$\tau_2$")
    lim_gt = [
        min(tau_log.min(), tau_recon_knode_gt_log.min()) - 0.1,
        max(tau_log.max(), tau_recon_knode_gt_log.max()) + 0.1,
    ]
    ax_gt.plot(lim_gt, lim_gt, "k--", linewidth=1.0, alpha=0.5, label="y = x")
    ax_gt.set_xlim(lim_gt)
    ax_gt.set_ylim(lim_gt)
    ax_gt.set_xlabel(r"applied $\tau$ [Nm]")
    ax_gt.set_ylabel(r"RNEA-reconstructed $\tau$ [Nm]")
    ax_gt.set_title("KNODE qdd @ GT states")
    ax_gt.set_aspect("equal")
    ax_gt.grid(True, alpha=0.3)
    ax_gt.legend(loc="upper left")

    ax_ol.scatter(tau_log[:, 0], tau_recon_knode_ol_log[:, 0], s=10, alpha=0.6, label=r"$\tau_1$")
    ax_ol.scatter(tau_log[:, 1], tau_recon_knode_ol_log[:, 1], s=10, alpha=0.6, label=r"$\tau_2$")
    lim_ol = [
        min(tau_log.min(), tau_recon_knode_ol_log.min()) - 0.1,
        max(tau_log.max(), tau_recon_knode_ol_log.max()) + 0.1,
    ]
    ax_ol.plot(lim_ol, lim_ol, "k--", linewidth=1.0, alpha=0.5, label="y = x")
    ax_ol.set_xlim(lim_ol)
    ax_ol.set_ylim(lim_ol)
    ax_ol.set_xlabel(r"applied $\tau$ [Nm]")
    ax_ol.set_ylabel(r"RNEA-reconstructed $\tau$ [Nm]")
    ax_ol.set_title("KNODE qdd @ open-loop states")
    ax_ol.set_aspect("equal")
    ax_ol.grid(True, alpha=0.3)
    ax_ol.legend(loc="upper left")

    fig_scatter.tight_layout()
    scatter_path = os.path.join(OUTPUT_DIR, "verify_rnea_tau_scatter.png")
    fig_scatter.savefig(scatter_path, dpi=150)
    plt.close(fig_scatter)
    print(f"Saved RNEA tau scatter -> {scatter_path}")

    print("\n--- RNEA consistency summary ---")
    print(f"GT qdd (sanity)     mean err: {err_gt_qdd_log.mean():.4e}  max: {err_gt_qdd_log.max():.4e}  (want ~0)")
    print(f"Imperfect qdd       mean err: {err_imperfect_log.mean():.4e}  max: {err_imperfect_log.max():.4e}")
    print(f"KNODE qdd @ GT      mean err: {err_knode_gt_log.mean():.4e}  max: {err_knode_gt_log.max():.4e}")
    print(f"KNODE qdd @ OL      mean err: {err_knode_ol_log.mean():.4e}  max: {err_knode_ol_log.max():.4e}")

    return {
        "time_s": time_s,
        "err_gt_qdd": err_gt_qdd_log,
        "err_imperfect": err_imperfect_log,
        "err_knode_gt": err_knode_gt_log,
        "err_knode_ol": err_knode_ol_log,
        "state_err_ol": state_err_ol_log,
        "tau_applied": tau_log,
        "tau_recon_knode_gt": tau_recon_knode_gt_log,
        "tau_recon_knode_ol": tau_recon_knode_ol_log,
    }


if __name__ == "__main__":
    data_dir = (
        "/Users/singhsegv/upenn-research-scalar/implementations/"
        "learning-manipulator-dynamics/knode-mpc-planar-arm/data/lala"
    )

    model_cfg = {
        "step_size": DT,
        "nominal_only": False,
        "perfect_nominal": False,
        "hidden_layers": 2,
        "hidden_neurons": 128,
        "activation": "tanh",
    }
    model = PlanarArmKnodeModel(model_cfg).to(torch.float64)

    epoch = 2100
    model_utils.CHECKPOINT_DIR = (
        "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/"
        "knode-mpc-planar-arm/runs/20260723_013245_focussed_circle_basic1_focussed-circle-la-2/checkpoints"
    )
    load_model_checkpoint(model, epoch)

    print(f"Loaded KNODE checkpoint epoch {epoch}")
    evaluate_rnea_consistency(data_dir, model)
