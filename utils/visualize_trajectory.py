import os
import time

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinocchio.visualize import MeshcatVisualizer
from robots.simulator import PlanarArm

DEFAULT_OUTPUT_DIR = None # NOTE: Modified by run_experiment.py
robot = PlanarArm(enable_viz=False) # Commong to all functions


def _to_numpy(states):
    if isinstance(states, torch.Tensor):
        return states.detach().cpu().numpy()
    return np.asarray(states, dtype=float)


def _extract_states(states):
    """
    Numpy states: (N, 7) => [t, q0, q1, qdot0, qdot1, tau0, tau1]
    """
    t = states[:, 0]
    dyn = states[:, 1:]
    return t, dyn[:, 0:2], dyn[:, 2:4], dyn[:, 4:6]


def pinocchio_simulate_trajectory(states, dt=0.002):
    """
    Re-integrate with Pinocchio using torques from states, starting at row 0.
    Returns q_log, qdot_log, xy_log.
    """
    _, q, qdot, taus = _extract_states(states)
    q_log, qdot_log = robot.simulate_rollout(q[0], qdot[0], taus, dt)
    xy_log = robot.ee_positions(q_log)
    return q_log, qdot_log, xy_log


def plot_trajectory_comparison(
        y_hat, ground_truth, dt=0.002, label=None, max_steps=None, output_dir=None,
        show=False, record_video=True, video_fps=30,
        nominal_qdd_log=None, residual_qdd_log=None, total_qdd_log=None,
        max_qdd_steps=None,
    ):
    """
    Compare predicted rollout vs ground truth in joint and task space.

    y_hat: (N, 7) predicted [t, q, qdot, tau]
    ground_truth: (N, 7) [t, q, qdot, tau]
    nominal_qdd_log, residual_qdd_log, total_qdd_log: optional lists of (1, 6)
        dynamics vectors from evaluate_model; qddot at indices 2:4.
    max_qdd_steps: if set, only plot this many qddot samples (from t=1 onward).
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    t_pred, q_pred, qdot_pred, _ = _extract_states(y_hat)
    t_true, q_true, qdot_true, _ = _extract_states(ground_truth)

    if (len(t_pred) != len(t_true)) or (len(q_pred) != len(q_true)):
        raise Exception("Mismatch in trajectory length")

    n = len(q_pred)
    if max_steps is not None:
        n = min(n, max_steps)

    q_pred = q_pred[:n]
    q_true = q_true[:n]
    qdot_pred = qdot_pred[:n]
    qdot_true = qdot_true[:n]
    time = t_true[:n]

    xy_pred = robot.ee_positions(q_pred)
    xy_true = robot.ee_positions(q_true)

    plot_qddot = (
        nominal_qdd_log is not None
        and residual_qdd_log is not None
        and total_qdd_log is not None
    )
    n_rows = 6 if plot_qddot else 3
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 2.8 * n_rows), sharex=True)
    fig.suptitle("Trajectory comparison", fontsize=13)

    joint_labels = ["q1", "q2"]
    for i in range(2):
        ax = axes[0, i]
        ax.plot(time, q_true[:, i], "g--", label="ground truth", linewidth=1.2)
        ax.plot(time, q_pred[:, i], "r--", label="model rollout", linewidth=1.2)
        ax.set_ylabel(joint_labels[i])
        ax.tick_params(labelbottom=True)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    qdot_labels = ["qdot1", "qdot2"]
    for i in range(2):
        ax = axes[1, i]
        ax.plot(time, qdot_true[:, i], "g--", label="ground truth", linewidth=1.2)
        ax.plot(time, qdot_pred[:, i], "r--", label="model rollout", linewidth=1.2)
        ax.set_ylabel(qdot_labels[i])
        ax.tick_params(labelbottom=True)
        ax.grid(True, alpha=0.3)

    task_row = 2
    if plot_qddot:
        nominal_qdd = _to_numpy(nominal_qdd_log)[:, 0, 2:4]
        residual_qdd = _to_numpy(residual_qdd_log)[:, 0, 2:4]
        total_qdd = _to_numpy(total_qdd_log)[:, 0, 2:4]
        n_qdd = min(n - 1, len(nominal_qdd))
        if max_qdd_steps is not None:
            n_qdd = min(n_qdd, max_qdd_steps)
        qdd_time = time[1:n_qdd + 1]

        qddot_rows = [
            (nominal_qdd, "nominal qddot1", "nominal qddot2", "C0"),
            (residual_qdd, "residual qddot1", "residual qddot2", "C1"),
            (total_qdd, "total qddot1", "total qddot2", "C3"),
        ]
        for row_offset, (qdd_data, label0, label1, color) in enumerate(qddot_rows):
            row = 2 + row_offset
            for i, ylabel in enumerate([label0, label1]):
                ax = axes[row, i]
                ax.plot(qdd_time, qdd_data[:n_qdd, i], color=color, linewidth=1.2)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
        task_row = 5

    task_labels = ["x", "y"]
    for i in range(2):
        ax = axes[task_row, i]
        ax.plot(time, xy_true[:, i], "g--", label="ground truth", linewidth=1.5)
        ax.plot(time, xy_pred[:, i], "r--", label="model rollout", linewidth=1.2)
        ax.set_ylabel(task_labels[i])
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch_{label}" if label is not None else ""
    save_path = os.path.join(output_dir, f"trajectory_comparison{suffix}.png")
    fig.savefig(save_path, dpi=150)

    xy_fig, xy_ax = plt.subplots(figsize=(6, 6))
    xy_ax.plot(xy_true[:, 0], xy_true[:, 1], "g--", label="ground truth", linewidth=1.5)
    xy_ax.plot(xy_pred[:, 0], xy_pred[:, 1], "r--", label="model rollout", linewidth=1.2)
    xy_ax.set_xlabel("x [m]")
    xy_ax.set_ylabel("y [m]")
    xy_ax.set_title("Task-space path")
    xy_ax.axis("equal")
    xy_ax.grid(True, alpha=0.3)
    xy_ax.legend()
    xy_path = os.path.join(output_dir, f"task_space_path{suffix}.png")
    xy_fig.tight_layout()
    xy_fig.savefig(xy_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(xy_fig)

    return fig, save_path


def horizon_losses(losses, label=None, output_dir=None, show=False):
    """
    Plot open-loop MSE vs prediction horizon.

    losses: dict {horizon_steps: {"total","q1","q2","qdot1","qdot2"}} from evaluate_horizon_loss
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    horizons = sorted(losses.keys())
    keys = ["total", "q1", "q2", "qdot1", "qdot2"]
    series = {k: [] for k in keys}
    for h in horizons:
        entry = losses[h]
        for k in keys:
            v = entry[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            series[k].append(float(v))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = {
        "total": ("k-o", 2.0),
        "q1": ("C0--s", 1.3),
        "q2": ("C1--s", 1.3),
        "qdot1": ("C2:^", 1.3),
        "qdot2": ("C3:^", 1.3),
    }
    for k in keys:
        fmt, lw = styles[k]
        ax.plot(horizons, series[k], fmt, linewidth=lw, markersize=5, label=k)

    ax.set_xlabel("horizon [steps]")
    ax.set_ylabel("open-loop MSE")
    ax.set_title("Horizon vs Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch_{label}" if label is not None else ""
    save_path = os.path.join(output_dir, f"horizon_losses{suffix}.png")
    fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, save_path


def horizon_losses_comparison(losses_left, losses_right, label=None, output_dir=None,
                              titles=("KNODE", "OG"), show=False):
    """
    Side-by-side open-loop MSE vs horizon panels (e.g. KNODE | OG).

    Each losses_* dict: {horizon_steps: {"total","q1","q2","qdot1","qdot2"}}
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    keys = ["total", "q1", "q2", "qdot1", "qdot2"]
    styles = {
        "total": ("k-o", 2.0),
        "q1": ("C0--s", 1.3),
        "q2": ("C1--s", 1.3),
        "qdot1": ("C2:^", 1.3),
        "qdot2": ("C3:^", 1.3),
    }

    def _series(losses):
        horizons = sorted(losses.keys())
        series = {k: [] for k in keys}
        for h in horizons:
            entry = losses[h]
            for k in keys:
                v = entry[k]
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().item()
                series[k].append(float(v))
        return horizons, series

    panels = [(losses_left, titles[0])]
    if losses_right is not None:
        panels.append((losses_right, titles[1]))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (losses, title) in zip(axes, panels):
        horizons, series = _series(losses)
        for k in keys:
            fmt, lw = styles[k]
            ax.plot(horizons, series[k], fmt, linewidth=lw, markersize=5, label=k)
        ax.set_xlabel("horizon [steps]")
        ax.set_ylabel("open-loop MSE")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch_{label}" if label is not None else ""
    save_path = os.path.join(output_dir, f"horizon_losses_comparison{suffix}.png")
    fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, save_path


def save_pinocchio_simulation_video(states, dt=0.002, fps=60, max_steps=None, output_dir=None, label=None):
    """
    Offline render of a Pinocchio rollout to mp4 using Panda3d.
    
    Avoids Matplotlib and Meshcat. Renders locally using Panda3D
    and captures directly to video using Pinocchio's context manager.
    """
    print("lalala1")
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    t_pred, q_log, qdot_log, taus = _extract_states(states)
    n = len(q_log)

    if max_steps is not None:
        n = min(n, max_steps)
        
    frame_skip = max(1, int(round(1.0 / (fps * dt))))
    qs_to_render = q_log[:n:frame_skip]
    
    # The actual dt between the frames we are feeding to the visualizer
    actual_dt = dt * frame_skip # TODO: Check if the integration blows up

    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    print("Waiting for Meshcat WebGL client to connect...")
    time.sleep(4.0)

    # (Optional) Adjust camera position for your specific 2R arm
    # viz.viewer.cam.setPos(0, -3, 0)
    # viz.viewer.cam.lookAt(0, 0, 0)

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch_{label}" if label is not None else ""
    output_path = os.path.join(output_dir, f"pinocchio_simulation{suffix}.mp4")

    print(f"Starting video render ({len(qs_to_render)} frames)...")
    with viz.create_video_ctx(output_path, fps=fps, macro_block_size=1):
        viz.play(qs_to_render, actual_dt)

    print(f"Saved Pinocchio simulation video -> {output_path}")
    return output_path
