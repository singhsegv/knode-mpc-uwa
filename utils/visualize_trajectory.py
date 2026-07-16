import os
import time

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinocchio.visualize import MeshcatVisualizer
from robots.simulator import PlanarArm

DEFAULT_OUTPUT_DIR = None # NOTE: Modified by run_experiment.py


def _to_numpy(states):
    if isinstance(states, torch.Tensor):
        return states.detach().cpu().numpy()
    return np.asarray(states, dtype=float)


def _extract_states(states):
    """
    states: (N, 6) => [q0, q1, qdot0, qdot1, tau0, tau1]
    or (N, 7) => [t, q0, q1, qdot0, qdot1, tau0, tau1]
    """
    arr = _to_numpy(states)
    if arr.shape[1] == 7:
        t = arr[:, 0]
        dyn = arr[:, 1:]
    else:
        t = None
        dyn = arr
    return t, dyn[:, 0:2], dyn[:, 2:4], dyn[:, 4:6]


def pinocchio_task_trajectory(states, robot=None):
    """Joint states (N, 6) -> EE (x, y) via Pinocchio FK."""
    robot = robot or PlanarArm()
    _, q, _, _ = _extract_states(states)
    return robot.ee_positions(q)


def pinocchio_simulate_trajectory(states, dt=0.002, robot=None):
    """
    Re-integrate with Pinocchio using torques from states, starting at row 0.
    Returns q_log, qdot_log, xy_log.
    """
    robot = robot or PlanarArm()
    _, q, qdot, taus = _extract_states(states)
    q_log, qdot_log = robot.simulate_rollout(q[0], qdot[0], taus, dt)
    xy_log = robot.ee_positions(q_log)
    return q_log, qdot_log, xy_log


def save_pinocchio_simulation_video(
    states,
    dt=0.002,
    output_path=None,
    fps=500,
    max_steps=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    epoch=None,
    robot=None,
):
    """
    Record a Pinocchio Meshcat rollout (tau replay via simulate_rollout) to mp4.
    """
    robot = robot or PlanarArm(enable_viz=False)
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=False)
    viz.loadViewerModel()

    _, q, qdot, taus = _extract_states(states)
    n = len(q)
    if max_steps is not None:
        n = min(n, max_steps)

    q_log, _ = robot.simulate_rollout(q[0], qdot[0], taus[:n], dt)

    frame_skip = max(1, int(round(1.0 / (fps * dt))))
    frames = []
    for i in range(0, n, frame_skip):
        viz.display(q_log[i])
        time.sleep(0.02)
        # meshcat Visualizer.get_image() in some versions ignores w/h args
        frames.append(viz.captureImage())

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch_{epoch}" if epoch is not None else ""
    if output_path is None:
        output_path = os.path.join(output_dir, f"pinocchio_simulation{suffix}.mp4")

    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved Pinocchio simulation video -> {output_path}")
    return output_path


def plot_trajectory_comparison(
        y_hat, ground_truth, dt=0.002, epoch=None, max_steps=None, output_dir=None,
        show=False, record_video=True, video_fps=30
    ):
    """
    Compare predicted rollout vs ground truth in joint and task space.

    y_hat: (N, 6) predicted [q, qdot, tau]
    ground_truth: (N, 7) [t, q, qdot, tau] or (N, 6)
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    robot = PlanarArm(enable_viz=False)

    t_pred, q_pred, _, _ = _extract_states(y_hat)
    t_true, q_true, _, _ = _extract_states(ground_truth)

    n = min(len(q_pred), len(q_true))
    if max_steps is not None:
        n = min(n, max_steps)

    q_pred = q_pred[:n]
    q_true = q_true[:n]

    if t_true is not None:
        time = t_true[:n]
    elif t_pred is not None:
        time = t_pred[:n]
    else:
        time = np.arange(n) * dt

    xy_pred = robot.ee_positions(q_pred)
    xy_true = robot.ee_positions(q_true)

    q_sim, _, xy_sim = pinocchio_simulate_trajectory(
        _to_numpy(y_hat)[:n],
        dt=dt,
        robot=robot,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"Trajectory comparison{f' (epoch {epoch})' if epoch is not None else ''}",
        fontsize=13,
    )

    joint_labels = ["q1", "q2"]
    for i in range(2):
        ax = axes[0, i]
        ax.plot(time, q_true[:, i], "k-", label="ground truth", linewidth=1.5)
        ax.plot(time, q_pred[:, i], "r--", label="model rollout", linewidth=1.2)
        ax.plot(time, q_sim[:, i], "g:", label="pinocchio sim (tau replay)", linewidth=1.2)
        ax.set_ylabel(joint_labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    task_labels = ["x", "y"]
    for i in range(2):
        ax = axes[1, i]
        ax.plot(time, xy_true[:, i], "k-", label="ground truth", linewidth=1.5)
        ax.plot(time, xy_pred[:, i], "r--", label="model rollout", linewidth=1.2)
        ax.plot(time, xy_sim[:, i], "g:", label="pinocchio sim (tau replay)", linewidth=1.2)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(task_labels[i])
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch_{epoch}" if epoch is not None else ""
    save_path = os.path.join(output_dir, f"trajectory_comparison{suffix}.png")
    fig.savefig(save_path, dpi=150)

    xy_fig, xy_ax = plt.subplots(figsize=(6, 6))
    xy_ax.plot(xy_true[:, 0], xy_true[:, 1], "k-", label="ground truth", linewidth=1.5)
    xy_ax.plot(xy_pred[:, 0], xy_pred[:, 1], "r--", label="model rollout", linewidth=1.2)
    xy_ax.plot(xy_sim[:, 0], xy_sim[:, 1], "g:", label="pinocchio sim (tau replay)", linewidth=1.2)
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

    # print(f"Saved plots -> {save_path}, {xy_path}")

    # video_path = None
    # if record_video:
    #     video_path = save_pinocchio_simulation_video(
    #         _to_numpy(y_hat)[:n],
    #         dt=dt,
    #         fps=video_fps,
    #         max_steps=n,
    #         output_dir=output_dir,
    #         epoch=epoch,
    #         robot=robot,
    #     )

    # return fig, save_path, video_path
    return fig, save_path
