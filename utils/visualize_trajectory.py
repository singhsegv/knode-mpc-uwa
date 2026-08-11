import os
import time

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinocchio.visualize import MeshcatVisualizer
from robots.simulator import Manipulator, load_robot_config

DEFAULT_OUTPUT_DIR = None
_robot = None


def set_robot(robot):
    """Set the robot used for FK / rollout visualization."""
    global _robot
    _robot = robot


def _to_numpy(states):
    if isinstance(states, torch.Tensor):
        return states.detach().cpu().numpy()
    return np.asarray(states, dtype=float)


def _infer_nq(states):
    """Infer nq from state row: [t, q(nq), qdot(nq), tau(nq)]."""
    n_cols = states.shape[1]
    return (n_cols - 1) // 3


def _extract_states(states):
    """
    Numpy states: (N, 1 + 3*nq) => [t, q, qdot, tau]
    """
    states = _to_numpy(states)
    nq = _infer_nq(states)
    t = states[:, 0]
    dyn = states[:, 1:]
    return t, dyn[:, :nq], dyn[:, nq : 2 * nq], dyn[:, 2 * nq : 3 * nq]


def pinocchio_simulate_trajectory(states, dt=0.002, robot=None):
    """Re-integrate with Pinocchio using torques from states, starting at row 0."""
    if robot is None:
        robot = _robot
    _, q, qdot, taus = _extract_states(states)
    q_log, qdot_log = robot.simulate_rollout(q[0], qdot[0], taus, dt)
    xyz_log = robot.ee_positions(q_log)
    return q_log, qdot_log, xyz_log


def _plot_joint_comparison(axes_row, time, true_data, pred_data, prefix, nq, row_idx, ncols):
    nrows = int(np.ceil(nq / ncols))
    for j in range(nq):
        r = row_idx * nrows + j // ncols
        c = j % ncols
        ax = axes_row[r, c] if axes_row.ndim == 2 else axes_row[r]
        ax.plot(time, true_data[:, j], "g--", label="ground truth", linewidth=1.2)
        ax.plot(time, pred_data[:, j], "r--", label="model rollout", linewidth=1.2)
        ax.set_ylabel(f"{prefix}{j + 1}")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="best", fontsize=7)


def _plot_task_space_3d(time, xyz_true, xyz_pred, output_dir, suffix, show, robot=None, q_true=None):
    """3D task-space comparison with optional robot skeleton snapshots."""
    from utils.robot_viz import  _set_axes_equal_3d

    fig = plt.figure(figsize=(14, 6))

    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.plot(
        xyz_true[:, 0], xyz_true[:, 1], xyz_true[:, 2],
        "g--", label="ground truth", linewidth=1.5, alpha=0.9,
    )
    ax3d.plot(
        xyz_pred[:, 0], xyz_pred[:, 1], xyz_pred[:, 2],
        "r-", label="model rollout", linewidth=1.2, alpha=0.9,
    )

    if robot is not None and q_true is not None:
        n = len(q_true)
        indices = list(range(0, n, max(1, n // 5)))
        if indices[-1] != n - 1:
            indices.append(n - 1)
        cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(indices)))
        for k, idx in enumerate(indices):
            # plot_robot_skeleton(
            #     ax3d, robot, q_true[idx],
            #     color=cmap[k], linewidth=1.5, alpha=0.55,
            #     label="robot" if k == 0 else None,
            # )
            pass

    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_title("Task-space path (3D)")
    ax3d.legend(fontsize=8)
    _set_axes_equal_3d(ax3d)

    ax_ts = fig.add_subplot(122)
    for idx, label in enumerate(["x", "y", "z"]):
        ax_ts.plot(time, xyz_true[:, idx], "--", alpha=0.8, label=f"gt {label}", linewidth=1.2)
        ax_ts.plot(time, xyz_pred[:, idx], "-", label=f"pred {label}", linewidth=1.0)
    ax_ts.set_xlabel("time [s]")
    ax_ts.set_ylabel("position [m]")
    ax_ts.set_title("Task-space vs time")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(fontsize=7)

    fig.tight_layout()
    path = os.path.join(output_dir, f"task_space_path{suffix}.png")
    fig.savefig(path, dpi=150)
    if not show:
        plt.close(fig)
    return fig, path


def _plot_task_space_2d(time, xy_true, xy_pred, output_dir, suffix, show):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(xy_true[:, 0], xy_true[:, 1], "g--", label="ground truth", linewidth=1.5)
    axes[0].plot(xy_pred[:, 0], xy_pred[:, 1], "r-", label="model rollout", linewidth=1.2)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("Task-space path")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for idx, label in enumerate(["x", "y"]):
        axes[1].plot(time, xy_true[:, idx], "--", label=f"gt {label}", linewidth=1.2)
        axes[1].plot(time, xy_pred[:, idx], "-", label=f"pred {label}", linewidth=1.0)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("position [m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, f"task_space_path{suffix}.png")
    fig.savefig(path, dpi=150)
    if not show:
        plt.close(fig)
    return fig, path


def plot_trajectory_comparison(
    y_hat, ground_truth, dt=0.002, label=None, max_steps=None, output_dir=None,
    show=False, record_video=True, video_fps=30,
    nominal_qdd_log=None, residual_qdd_log=None, total_qdd_log=None,
    max_qdd_steps=None, robot=None,
):
    """
    Compare predicted rollout vs ground truth in joint and task space.

    y_hat / ground_truth: (N, 1 + 3*nq) = [t, q, qdot, tau]
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    if robot is None:
        robot = _robot

    t_pred, q_pred, qdot_pred, _ = _extract_states(y_hat)
    t_true, q_true, qdot_true, _ = _extract_states(ground_truth)

    if (len(t_pred) != len(t_true)) or (len(q_pred) != len(q_true)):
        raise Exception("Mismatch in trajectory length")

    nq = q_pred.shape[1]
    n = len(q_pred)
    if max_steps is not None:
        n = min(n, max_steps)

    q_pred = q_pred[:n]
    q_true = q_true[:n]
    qdot_pred = qdot_pred[:n]
    qdot_true = qdot_true[:n]
    time = t_true[:n]

    xyz_pred = robot.ee_positions(q_pred)
    xyz_true = robot.ee_positions(q_true)
    task_dim = getattr(robot, "task_space_dim", 3)

    plot_qddot = (
        nominal_qdd_log is not None
        and residual_qdd_log is not None
        and total_qdd_log is not None
    )

    ncols = min(3, nq)
    nrows_per_block = int(np.ceil(nq / ncols))
    n_joint_blocks = 2 + (3 if plot_qddot else 0)
    total_rows = n_joint_blocks * nrows_per_block

    fig, axes = plt.subplots(total_rows, ncols, figsize=(4 * ncols, 2.2 * total_rows), sharex=True)
    if total_rows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif total_rows == 1:
        axes = axes.reshape(1, ncols)

    fig.suptitle("Trajectory comparison", fontsize=13)

    _plot_joint_comparison(axes, time, q_true, q_pred, "q", nq, 0, ncols)
    _plot_joint_comparison(axes, time, qdot_true, qdot_pred, "qdot", nq, 1, ncols)

    next_block = 2
    if plot_qddot:
        nominal_qdd = _to_numpy(nominal_qdd_log)[:, 0, nq : 2 * nq]
        residual_qdd = _to_numpy(residual_qdd_log)[:, 0, nq : 2 * nq]
        total_qdd = _to_numpy(total_qdd_log)[:, 0, nq : 2 * nq]
        n_qdd = min(n - 1, len(nominal_qdd))
        if max_qdd_steps is not None:
            n_qdd = min(n_qdd, max_qdd_steps)
        qdd_time = time[1:n_qdd + 1]

        qddot_rows = [
            (nominal_qdd, "nom qdd"),
            (residual_qdd, "res qdd"),
            (total_qdd, "tot qdd"),
        ]
        for offset, (qdd_data, prefix) in enumerate(qddot_rows):
            for j in range(nq):
                r = (next_block + offset) * nrows_per_block + j // ncols
                c = j % ncols
                ax = axes[r, c]
                ax.plot(qdd_time, qdd_data[:n_qdd, j], linewidth=1.2)
                ax.set_ylabel(f"{prefix}{j + 1}")
                ax.grid(True, alpha=0.3)
        next_block += 3

    for col in range(ncols):
        axes[-1, col].set_xlabel("time [s]")

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch_{label}" if label is not None else ""
    save_path = os.path.join(output_dir, f"trajectory_comparison{suffix}.png")
    fig.savefig(save_path, dpi=150)
    if not show:
        plt.close(fig)

    if task_dim == 3:
        _plot_task_space_3d(time, xyz_true, xyz_pred, output_dir, suffix, show, robot=robot, q_true=q_true)
    else:
        _plot_task_space_2d(time, xyz_true, xyz_pred, output_dir, suffix, show)

    return fig, save_path


def horizon_losses(losses, label=None, output_dir=None, show=False):
    """Plot open-loop MSE vs prediction horizon."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    horizons = sorted(losses.keys())
    keys = [k for k in losses[horizons[0]].keys()]

    series = {k: [] for k in keys}
    for h in horizons:
        entry = losses[h]
        for k in keys:
            v = entry[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            series[k].append(float(v))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in keys:
        ax.plot(horizons, series[k], "-o", markersize=4, linewidth=1.3, label=k)

    ax.set_xlabel("horizon [steps]")
    ax.set_ylabel("open-loop MSE")
    ax.set_title("Horizon vs Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)
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
    """Side-by-side open-loop MSE vs horizon panels."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    def _series(losses):
        horizons = sorted(losses.keys())
        keys = list(losses[horizons[0]].keys())
        series = {k: [] for k in keys}
        for h in horizons:
            entry = losses[h]
            for k in keys:
                v = entry[k]
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().item()
                series[k].append(float(v))
        return horizons, keys, series

    panels = [(losses_left, titles[0])]
    if losses_right is not None:
        panels.append((losses_right, titles[1]))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (losses, title) in zip(axes, panels):
        horizons, keys, series = _series(losses)
        for k in keys:
            ax.plot(horizons, series[k], "-o", markersize=4, linewidth=1.3, label=k)
        ax.set_xlabel("horizon [steps]")
        ax.set_ylabel("open-loop MSE")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

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


def save_pinocchio_simulation_video(states, dt=0.002, fps=60, max_steps=None, output_dir=None, label=None, robot=None):
    """Offline render of a Pinocchio rollout to mp4 using Meshcat."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    if robot is None:
        robot = _robot

    _, q_log, _, _ = _extract_states(states)
    n = len(q_log)
    if max_steps is not None:
        n = min(n, max_steps)

    frame_skip = max(1, int(round(1.0 / (fps * dt))))
    qs_to_render = q_log[:n:frame_skip]
    actual_dt = dt * frame_skip

    viz = MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model,
    )
    viz.initViewer(open=True)
    viz.loadViewerModel()

    print("Waiting for Meshcat WebGL client to connect...")
    time.sleep(4.0)

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch_{label}" if label is not None else ""
    output_path = os.path.join(output_dir, f"pinocchio_simulation{suffix}.mp4")

    print(f"Starting video render ({len(qs_to_render)} frames)...")
    with viz.create_video_ctx(output_path, fps=fps, macro_block_size=1):
        viz.play(qs_to_render, actual_dt)

    print(f"Saved Pinocchio simulation video -> {output_path}")
    return output_path
