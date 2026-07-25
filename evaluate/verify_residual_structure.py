"""
Check whether the learned residual has the expected physics structure.

At fixed q, the gravity correction delta_qdd = -M(q)^{-1} g(q) should not depend on
qdot or tau. This script sweeps (qdot, tau), compares to the analytic target, and plots
spread / alignment metrics.
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

from verify_model import DT, OUTPUT_DIR, get_learned_residual


def physics_gravity_residual(robot, q):
    """Analytic target: delta_qdd = -M(q)^{-1} g(q); depends on q only."""
    M = robot.mass_matrix(q)
    g = robot.gravity_forces(q)
    return -np.linalg.solve(M, g)


def _cosine_similarity(a, b, eps=1e-12):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def _residual_std_at_fixed_q(knode_model, robot, q, qdot_samples, tau_samples):
    """
    Evaluate learned / true residual over a grid of (qdot, tau) at fixed q.
    True gravity residual should have zero std; learned std > 0 => spurious dependence.
    """
    learned = []
    true = physics_gravity_residual(robot, q)
    true_stack = []

    for qdot in qdot_samples:
        for tau in tau_samples:
            learned.append(get_learned_residual(knode_model, q, qdot, tau))
            true_stack.append(true)

    learned = np.asarray(learned)
    true_stack = np.asarray(true_stack)
    learned_std = learned.std(axis=0)
    true_std = true_stack.std(axis=0)
    learned_mean = learned.mean(axis=0)
    return {
        "learned_std": learned_std,
        "learned_std_norm": np.linalg.norm(learned_std),
        "true_std_norm": np.linalg.norm(true_std),
        "learned_mean": learned_mean,
        "physics_target": true,
        "learned_spread_norm": np.linalg.norm(learned - learned_mean, axis=1).max(),
    }


def evaluate_residual_structure(data_path, knode_model, n_qdot=7, n_tau=7, q_subsample=25):
    """
    Step 2: check whether the learned residual has the right structure.

    Physics says delta_qdd = -M(q)^{-1} g(q) depends on q only. At fixed q we sweep
    qdot and tau from the dataset range and measure residual spread. Also compare
    learned residual at GT states to the analytic gravity target (direction + magnitude).
    """
    robot = PlanarArm(enable_viz=False)
    traj = load_evaluation_data(data_path)[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    controls = traj["controls"]
    q_traj = controls[:, 1:3]
    qdot_traj = controls[:, 3:5]
    tau_traj = controls[:, 5:7]

    qdot_grid = np.linspace(qdot_traj.min(axis=0), qdot_traj.max(axis=0), n_qdot)
    tau_grid = np.linspace(tau_traj.min(axis=0), tau_traj.max(axis=0), n_tau)

    n_steps = len(q_traj)
    step_indices = np.arange(0, n_steps, max(1, n_steps // q_subsample))

    learned_std_norm_log = np.empty(len(step_indices))
    true_std_norm_log = np.zeros(len(step_indices))
    learned_spread_log = np.empty(len(step_indices))
    cos_sim_log = np.empty(len(step_indices))
    mag_ratio_log = np.empty(len(step_indices))
    delta_learned_gt_log = np.empty((len(step_indices), 2))
    delta_physics_log = np.empty((len(step_indices), 2))

    for j, idx in enumerate(step_indices):
        q = q_traj[idx]
        qdot = qdot_traj[idx]
        tau = tau_traj[idx]

        sweep = _residual_std_at_fixed_q(knode_model, robot, q, qdot_grid, tau_grid)
        learned_std_norm_log[j] = sweep["learned_std_norm"]
        true_std_norm_log[j] = sweep["true_std_norm"]
        learned_spread_log[j] = sweep["learned_spread_norm"]

        delta_learned = get_learned_residual(knode_model, q, qdot, tau)
        delta_physics = physics_gravity_residual(robot, q)
        delta_learned_gt_log[j] = delta_learned
        delta_physics_log[j] = delta_physics
        cos_sim_log[j] = _cosine_similarity(delta_learned, delta_physics)
        mag_ratio_log[j] = np.linalg.norm(delta_learned) / (np.linalg.norm(delta_physics) + 1e-12)

    time_s = step_indices * DT

    fig_summary, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(time_s, learned_std_norm_log, "C1-o", markersize=3, linewidth=1.2, label=r"learned std $\|\sigma(\Delta\ddot q)\|$")
    axes[0].plot(time_s, learned_spread_log, "C3--", linewidth=1.2, label=r"learned max deviation from mean")
    axes[0].axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
    axes[0].set_ylabel("[rad/s$^2$]")
    axes[0].set_title(r"Fixed-$q$ sweep: learned residual variation over $(\dot q, \tau)$ grid")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(time_s, cos_sim_log, "C0-o", markersize=3, linewidth=1.2)
    axes[1].axhline(1.0, color="k", linewidth=0.8, alpha=0.4, linestyle="--")
    axes[1].set_ylabel("cosine sim")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_title(r"At GT $(q,\dot q,\tau)$: alignment of learned vs $-M(q)^{-1}g(q)$")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_s, np.linalg.norm(delta_learned_gt_log, axis=1), "C1-", linewidth=1.5, label=r"$\|\Delta\ddot q_\mathrm{learned}\|$")
    axes[2].plot(time_s, np.linalg.norm(delta_physics_log, axis=1), "C0--", linewidth=1.5, label=r"$\|-M^{-1}g\|$")
    axes[2].set_ylabel("[rad/s$^2$]")
    axes[2].set_title("Residual magnitude at GT states")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    axes[3].plot(time_s, mag_ratio_log, "C2-o", markersize=3, linewidth=1.2)
    axes[3].axhline(1.0, color="k", linewidth=0.8, alpha=0.4, linestyle="--")
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("ratio")
    axes[3].set_title(r"$\|\Delta\ddot q_\mathrm{learned}\| / \|-M^{-1}g\|$ at GT states")
    axes[3].grid(True, alpha=0.3)

    fig_summary.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, "verify_residual_structure_summary.png")
    fig_summary.savefig(summary_path, dpi=150)
    plt.close(fig_summary)
    print(f"Saved residual structure summary -> {summary_path}")

    fig_hist, (ax_std, ax_cos) = plt.subplots(1, 2, figsize=(10, 4))

    ax_std.hist(learned_std_norm_log, bins=20, color="C1", alpha=0.85, edgecolor="white")
    ax_std.set_xlabel(r"$\|\sigma(\Delta\ddot q_\mathrm{learned})\|$ at fixed $q$ [rad/s$^2$]")
    ax_std.set_ylabel("count")
    ax_std.set_title("Spread over $(\\dot q, \\tau)$ sweep (want near 0)")
    ax_std.grid(True, alpha=0.3)

    ax_cos.hist(cos_sim_log, bins=20, color="C0", alpha=0.85, edgecolor="white")
    ax_cos.set_xlabel(r"cosine sim$(\Delta_\mathrm{learned}, -M^{-1}g)$")
    ax_cos.set_ylabel("count")
    ax_cos.set_title("Direction alignment at GT states (want near 1)")
    ax_cos.grid(True, alpha=0.3)

    fig_hist.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, "verify_residual_structure_histograms.png")
    fig_hist.savefig(hist_path, dpi=150)
    plt.close(fig_hist)
    print(f"Saved residual structure histograms -> {hist_path}")

    fig_cmp, (ax_j1, ax_j2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_j1.plot(time_s, delta_physics_log[:, 0], "C0-", linewidth=1.5, label=r"$-M^{-1}g$ (target)")
    ax_j1.plot(time_s, delta_learned_gt_log[:, 0], "C1--", linewidth=1.5, label=r"$\Delta\ddot q_\mathrm{learned}$")
    ax_j1.set_ylabel(r"$\Delta\ddot q_1$ [rad/s$^2$]")
    ax_j1.set_title("Learned residual vs analytic gravity correction at GT states")
    ax_j1.grid(True, alpha=0.3)
    ax_j1.legend(loc="upper right")

    ax_j2.plot(time_s, delta_physics_log[:, 1], "C0-", linewidth=1.5, label=r"$-M^{-1}g$ (target)")
    ax_j2.plot(time_s, delta_learned_gt_log[:, 1], "C1--", linewidth=1.5, label=r"$\Delta\ddot q_\mathrm{learned}$")
    ax_j2.set_xlabel("time [s]")
    ax_j2.set_ylabel(r"$\Delta\ddot q_2$ [rad/s$^2$]")
    ax_j2.grid(True, alpha=0.3)
    ax_j2.legend(loc="upper right")

    fig_cmp.tight_layout()
    cmp_path = os.path.join(OUTPUT_DIR, "verify_residual_structure_components.png")
    fig_cmp.savefig(cmp_path, dpi=150)
    plt.close(fig_cmp)
    print(f"Saved residual structure components -> {cmp_path}")

    rep_indices = [step_indices[0], step_indices[len(step_indices) // 2], step_indices[-1]]
    fig_sweep, axes_sweep = plt.subplots(len(rep_indices), 2, figsize=(11, 3.5 * len(rep_indices)))
    if len(rep_indices) == 1:
        axes_sweep = np.array([axes_sweep])

    for row, idx in enumerate(rep_indices):
        q = q_traj[idx]
        qdot_ref = qdot_traj[idx]
        tau_ref = tau_traj[idx]
        physics = physics_gravity_residual(robot, q)

        qdot1_axis = np.linspace(qdot_traj[:, 0].min(), qdot_traj[:, 0].max(), 40)
        res_qdot1 = np.array([
            get_learned_residual(knode_model, q, np.array([qd1, qdot_ref[1]]), tau_ref)
            for qd1 in qdot1_axis
        ])
        res_tau1 = np.array([
            get_learned_residual(knode_model, q, qdot_ref, np.array([t1, tau_ref[1]]))
            for t1 in np.linspace(tau_traj[:, 0].min(), tau_traj[:, 0].max(), 40)
        ])

        ax_qd, ax_tau = axes_sweep[row]
        ax_qd.plot(qdot1_axis, res_qdot1[:, 0], "C0-", label=r"$\Delta\ddot q_1$")
        ax_qd.plot(qdot1_axis, res_qdot1[:, 1], "C1-", label=r"$\Delta\ddot q_2$")
        ax_qd.axhline(physics[0], color="C0", linestyle="--", alpha=0.6, label="target joint 1")
        ax_qd.axhline(physics[1], color="C1", linestyle="--", alpha=0.6, label="target joint 2")
        ax_qd.set_ylabel(r"$\Delta\ddot q$ [rad/s$^2$]")
        ax_qd.set_title(f"fixed q={q.round(3)}, sweep qdot1 (ref qdot2, tau)")
        ax_qd.grid(True, alpha=0.3)
        ax_qd.legend(loc="upper right", fontsize=8)

        tau1_axis = np.linspace(tau_traj[:, 0].min(), tau_traj[:, 0].max(), 40)
        ax_tau.plot(tau1_axis, res_tau1[:, 0], "C0-", label=r"$\Delta\ddot q_1$")
        ax_tau.plot(tau1_axis, res_tau1[:, 1], "C1-", label=r"$\Delta\ddot q_2$")
        ax_tau.axhline(physics[0], color="C0", linestyle="--", alpha=0.6)
        ax_tau.axhline(physics[1], color="C1", linestyle="--", alpha=0.6)
        ax_tau.set_xlabel(r"$\tau_1$ [Nm]")
        ax_tau.set_ylabel(r"$\Delta\ddot q$ [rad/s$^2$]")
        ax_tau.set_title(f"fixed q={q.round(3)}, sweep tau1 (ref qdot)")
        ax_tau.grid(True, alpha=0.3)
        ax_tau.legend(loc="upper right", fontsize=8)

        if row == 0:
            ax_qd.set_xlabel(r"$\dot q_1$ [rad/s]")

    fig_sweep.tight_layout()
    sweep_path = os.path.join(OUTPUT_DIR, "verify_residual_structure_sweeps.png")
    fig_sweep.savefig(sweep_path, dpi=150)
    plt.close(fig_sweep)
    print(f"Saved residual structure sweeps -> {sweep_path}")

    print("\n--- Residual structure summary ---")
    print(f"mean learned std at fixed q: {learned_std_norm_log.mean():.4e}  (want ~0)")
    print(f"max  learned std at fixed q: {learned_std_norm_log.max():.4e}")
    print(f"mean cosine similarity:      {cos_sim_log.mean():.4f}  (want ~1)")
    print(f"min  cosine similarity:      {cos_sim_log.min():.4f}")
    print(f"mean magnitude ratio:        {mag_ratio_log.mean():.4f}  (want ~1)")

    return {
        "step_indices": step_indices,
        "learned_std_norm": learned_std_norm_log,
        "learned_spread": learned_spread_log,
        "cosine_similarity": cos_sim_log,
        "magnitude_ratio": mag_ratio_log,
        "delta_learned_gt": delta_learned_gt_log,
        "delta_physics": delta_physics_log,
    }


if __name__ == "__main__":
    data_dir = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/data/lala"

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
    evaluate_residual_structure(data_dir, model)
