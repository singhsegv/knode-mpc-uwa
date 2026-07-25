"""
USECASE:
1. Have an existing set of collected data: {q, qdot, tau}
2. Only want to use q0, qdot0 but with all the applied taus
3. Change robot urdf: gravity/mass/length and see the effects of those same taus
4. Do: visualize in pinnochio, plot the ee, plot the joints etc

Usage:
  python evaluate/rollout_existing_data.py evaluation1/test
  python evaluate/rollout_existing_data.py evaluation1/test --viz
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from robots.simulator import PlanarArm
from utils.training_data_generator import plot_collected_trajectory

DATA_DIR = os.path.join(PROJECT_ROOT, "data")


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


def list_npz_files(folder):
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.endswith(".npz")
    )


def rollout_npz(robot, npz_path, use_torchdiffeq=True):
    data = np.load(npz_path, allow_pickle=True)
    t = np.asarray(data["t"]).reshape(-1)
    q = np.asarray(data["q"])
    qdot = np.asarray(data["qdot"])
    tau = np.asarray(data["tau"])
    dt = float(np.asarray(data["dt"]).reshape(-1)[0])

    q_log, qdot_log = robot.simulate_rollout(q[0], qdot[0], tau, dt, use_torchdiffeq=use_torchdiffeq, visualize=True)

    qddot_log = np.empty_like(q_log)
    for i in range(len(tau)):
        qddot_log[i] = robot.forward_dynamics(q_log[i], qdot_log[i], tau[i])

    meta = data["meta"].item() if "meta" in data else {}

    return {
        "t": t,
        "q": q_log,
        "qdot": qdot_log,
        "tau": tau,
        "qddot": qddot_log,
        "meta": meta,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Re-simulate saved torques from q0, qdot0 and plot the rollout."
    )
    parser.add_argument(
        "folder",
        help="Folder under data/ (or path) containing .npz rollout files",
    )
    parser.add_argument("--viz", action="store_true", help="Enable Meshcat visualization")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for plots (default: same folder as the .npz files)",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    # parser.add_argument(
    #     "--rk4",
    #     action="store_true",
    #     help="Use manual RK4 integration instead of torchdiffeq",
    # )
    args = parser.parse_args()

    folder = resolve_folder(args.folder)
    npz_paths = list_npz_files(folder)
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in {folder}")

    robot = PlanarArm(enable_viz=args.viz)
    # robot.GRAVITY = np.array([0.0, -5.81, 0.0])
    output_dir = args.output_dir or folder

    for npz_path in npz_paths:
        log = rollout_npz(robot, npz_path, use_torchdiffeq=True)
        name_prefix = os.path.splitext(os.path.basename(npz_path))[0]
        meta = log["meta"]

        plot_collected_trajectory(
            t_log=log["t"],
            q_log=log["q"],
            qdot_log=log["qdot"],
            tau_log=log["tau"],
            qddot_log=log["qddot"],
            trajectory=meta.get("trajectory"),
            max_steps=meta.get("n_steps"),
            name_prefix=name_prefix,
            output_dir=output_dir,
            arm=robot,
            show=args.show,
        )
        print(f"Rolled out {npz_path} -> plots in {output_dir}")


if __name__ == "__main__":
    main()
