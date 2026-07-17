"""
Compare applied torques and EE paths across collected rollout folders.

Each folder should contain a single .npz from training_data_generator.save_log:
  t, q, qdot, tau, des_ee_pos, des_ee_vel, des_ee_acc, dt, meta

Usage:
  python utils/compare_collected_rollouts.py data/standalone data/other_run
  python utils/compare_collected_rollouts.py standalone other_run
  python utils/compare_collected_rollouts.py data/standalone --save
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robots.simulator import PlanarArm

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
robot = PlanarArm(enable_viz=False)


def resolve_folder(name):
    """Accept absolute path, relative path, or bare name under data/."""
    candidates = [
        name,
        os.path.join(PROJECT_ROOT, name),
        os.path.join(DATA_DIR, name),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"Folder not found: {name} (tried {candidates})")


def load_folder_npz(folder):
    npz_files = sorted(
        f for f in os.listdir(folder) if f.endswith(".npz")
    )
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz in {folder}")
    if len(npz_files) > 1:
        print(f"Warning: {folder} has {len(npz_files)} .npz files; using {npz_files[0]}")

    path = os.path.join(folder, npz_files[0])
    data = np.load(path, allow_pickle=True)
    label = os.path.basename(folder.rstrip(os.sep))
    return label, path, data


def plot_comparison(rollouts, output_path=None, show=True):
    """
    rollouts: list of (label, data) where data is np.load result.
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="col")
    fig.suptitle("Collected rollout comparison", fontsize=13)

    for label, data in rollouts:
        t = np.asarray(data["t"]).reshape(-1)
        tau = np.asarray(data["tau"])
        # des_ee = np.asarray(data["des_ee_pos"])
        q = np.asarray(data["q"])
        act_ee = robot.ee_positions(q)

        axes[0, 0].plot(t, tau[:, 0], label=label, linewidth=1.2)
        axes[0, 1].plot(t, tau[:, 1], label=label, linewidth=1.2)

        # axes[1, 0].plot(t, des_ee[:, 0], "--", linewidth=1.2, label=f"{label} des")
        axes[1, 0].plot(t, act_ee[:, 0], "-", linewidth=1.2, label=f"{label} act")
        # axes[1, 1].plot(t, des_ee[:, 1], "--", linewidth=1.2, label=f"{label} des")
        axes[1, 1].plot(t, act_ee[:, 1], "-", linewidth=1.2, label=f"{label} act")

        # axes[2, 0].plot(des_ee[:, 0], des_ee[:, 1], "--", linewidth=1.2, label=f"{label} des")
        axes[2, 0].plot(act_ee[:, 0], act_ee[:, 1], "-", linewidth=1.2, label=f"{label} act")

    axes[0, 0].set_ylabel("tau1 [Nm]")
    axes[0, 1].set_ylabel("tau2 [Nm]")
    axes[0, 0].set_title("Applied torque")
    axes[0, 1].set_title("Applied torque")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)

    axes[1, 0].set_ylabel("x [m]")
    axes[1, 1].set_ylabel("y [m]")
    axes[1, 0].set_title("EE x vs time")
    axes[1, 1].set_title("EE y vs time")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="best", fontsize=7)
    axes[1, 1].legend(loc="best", fontsize=7)
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")

    # Path plot spans left cell; hide unused right cell.
    axes[2, 0].set_xlabel("x [m]")
    axes[2, 0].set_ylabel("y [m]")
    axes[2, 0].set_title("Task-space path")
    axes[2, 0].axis("equal")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend(loc="best", fontsize=7)
    axes[2, 1].axis("off")

    fig.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot -> {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Compare torques and EE paths across collected .npz folders."
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="Folder names/paths each containing one collected .npz (e.g. data/standalone)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"Save figure under {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output PNG path (implies save)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window",
    )
    args = parser.parse_args()

    rollouts = []
    for name in args.folders:
        folder = resolve_folder(name)
        label, path, data = load_folder_npz(folder)
        print(f"Loaded {label}: {path} (N={len(data['t'])})")
        rollouts.append((label, data))

    output_path = args.output
    if output_path is None and args.save:
        names = "_vs_".join(label for label, _ in rollouts)
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"compare_rollouts_{names}.png")

    plot_comparison(rollouts, output_path=output_path, show=not args.no_show)


if __name__ == "__main__":
    main()
