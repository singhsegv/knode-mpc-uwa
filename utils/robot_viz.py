"""Matplotlib helpers for 3D robot and task-space visualization."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _set_axes_equal_3d(ax):
    """Make 3D axes have equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    if radius < 1e-6:
        radius = 0.1
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


def plot_task_space_3d_scene(
    ax,
    robot,
    q_traj,
    ee_actual,
    ee_desired=None,
    skeleton_stride=None,
    max_skeleton=5,
):
    """
    Plot EE trajectory and robot skeleton snapshots on a 3D axes.

    q_traj: (N, nq) joint trajectory for skeleton snapshots
    ee_actual: (N, 3) actual EE path
    ee_desired: optional (N, 3) desired EE path
    """
    if ee_desired is not None:
        ax.plot(
            ee_desired[:, 0], ee_desired[:, 1], ee_desired[:, 2],
            "g--", linewidth=1.5, label="desired EE", alpha=0.85,
        )

    ax.plot(
        ee_actual[:, 0], ee_actual[:, 1], ee_actual[:, 2],
        "b-", linewidth=1.5, label="actual EE", alpha=0.9,
    )

    n = len(q_traj)
    if n == 0:
        return

    if skeleton_stride is None:
        skeleton_stride = max(1, n // max_skeleton)
    indices = list(range(0, n, skeleton_stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(indices)))
    # for k, idx in enumerate(indices):
    #     plot_robot_skeleton(
    #         ax, robot, q_traj[idx],
    #         color=cmap[k], linewidth=1.5, alpha=0.55,
    #         label="robot" if k == 0 else None,
    #     )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    _set_axes_equal_3d(ax)
    ax.legend(fontsize=8, loc="upper left")


def save_task_space_3d_figure(
    robot, q_traj, ee_actual, ee_desired=None, title="", save_path=None, show=False,
):
    """Create and save a standalone 3D task-space figure with robot context."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    plot_task_space_3d_scene(ax, robot, q_traj, ee_actual, ee_desired=ee_desired)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
