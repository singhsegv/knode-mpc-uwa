"""
Goal: Testing generalizability of the trained KNODE across multiple trajectories.
Data:
    - Multiple trajectory shapes
    - Variety of positions
    - Variety of speeds
"""
import os
import sys
import time
import json
import argparse

import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from knode.model import PlanarArmKnodeModel
from robots.simulator import PlanarArm

from utils.data_utils import load_data as load_rollouts
import utils.model_utils as model_utils
from utils.model_utils import evaluate_horizon_loss, latest_checkpoint_epoch, load_model_checkpoint
from utils.visualize_trajectory import horizon_losses_comparison

RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
HORIZONS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500]
# HORIZONS = list(range(1, 20))
LOSS_KEYS = ["total", "q1", "q2", "qdot1", "qdot2"]
DT = 0.002


def step_with_knode(model, q, qdot, tau, dt):
    """One RK4 step of PlanarArmKnodeModel: (q, qdot, tau) -> (q_next, qdot_next)."""
    model.eval()
    with torch.no_grad():
        y0 = torch.tensor(np.concatenate([q, qdot, tau], axis=0)[None, :], dtype=torch.float64)
        timesteps = torch.tensor([0.0, dt], dtype=torch.float64)
        pred = model(y0, timesteps)  # (2, 1, 6)
        nxt = pred[-1, 0].detach().cpu().numpy()
        
    return nxt[:2].copy(), nxt[2:4].copy()


def resolve_run_dir(run_name):
    if os.path.isabs(run_name) and os.path.isdir(run_name):
        return run_name
    candidate = os.path.join(RUNS_DIR, run_name)
    if os.path.isdir(candidate):
        return candidate
    if os.path.isdir(run_name):
        return os.path.abspath(run_name)
    raise FileNotFoundError(f"Run directory not found: {run_name} (looked under {RUNS_DIR})")


def load_model(run_dir, epoch=None):
    """
    Read model.json from the run folder, build PlanarArmKnodeModel, load checkpoint.
    epoch=None -> latest checkpoint.
    """
    model_cfg_path = os.path.join(run_dir, "model.json")
    if not os.path.isfile(model_cfg_path):
        raise FileNotFoundError(f"Missing model.json in {run_dir}")

    with open(model_cfg_path) as f:
        model_cfg = json.load(f)

    model_utils.CHECKPOINT_DIR = os.path.join(run_dir, "checkpoints")
    if epoch is None:
        epoch = latest_checkpoint_epoch()
    if epoch is None:
        raise FileNotFoundError(f"No checkpoints found under {model_utils.CHECKPOINT_DIR}")

    model = PlanarArmKnodeModel(model_cfg).to(torch.float64)
    load_model_checkpoint(model, epoch)
    model.eval()
    return model, epoch, model_cfg


def load_data(data_path):
    """
    Load benchmark rollouts from a directory of .npz files (or a single .npz).

    Returns list of (name, trajectory) where trajectory is (N, 7) = [t, q, qdot, tau].
    """
    if os.path.isdir(data_path):
        names = sorted(n for n in os.listdir(data_path) if n.endswith(".npz"))
        if not names:
            raise FileNotFoundError(f"No .npz files found in {data_path}")
        rollouts = load_rollouts(data_path)
        return list(zip([os.path.splitext(n)[0] for n in names], rollouts))

    rollouts = load_rollouts(data_path)
    name = os.path.splitext(os.path.basename(data_path))[0]
    return [(name, rollouts[0])]


def _scalar(v):
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)


def losses_to_floats(losses):
    return {
        h: {k: _scalar(entry[k]) for k in LOSS_KEYS}
        for h, entry in losses.items()
    }


def print_horizon_table(title, knode_losses, og_losses=None):
    print(f"\n{'=' * 88}")
    print(title)
    print(f"{'=' * 88}")

    header = f"{'h':>6} | {'knode_total':>12} {'q1':>10} {'q2':>10} {'qdot1':>10} {'qdot2':>10}"
    if og_losses is not None:
        header += f" | {'og_total':>12}"
    print(header)
    print("-" * len(header))

    for h in sorted(knode_losses.keys()):
        e = knode_losses[h]
        row = (
            f"{h:6d} | {e['total']:12.4e} {e['q1']:10.4e} {e['q2']:10.4e} "
            f"{e['qdot1']:10.4e} {e['qdot2']:10.4e}"
        )
        if og_losses is not None:
            row += f" | {og_losses[h]['total']:12.4e}"
        print(row)


def mean_losses(all_losses):
    """Average float loss dicts across trajectories: list[dict[h -> metrics]]."""
    if not all_losses:
        return {}

    horizons = sorted(all_losses[0].keys())
    out = {}
    n = len(all_losses)
    for h in horizons:
        out[h] = {k: sum(d[h][k] for d in all_losses) / n for k in LOSS_KEYS}
    return out


def make_nominal_model(model_cfg):
    og_cfg = dict(model_cfg)
    og_cfg["nominal_only"] = True
    og_cfg["perfect_nominal"] = False
    return PlanarArmKnodeModel(og_cfg).to(torch.float64)


def visualize_robot_simulation(model, trajectory):
    """
    Uses the tau from ground truth but dynamics from learned knode.
    """
    robot = PlanarArm(enable_viz=True)
    n_steps = trajectory.shape[0]

    q = trajectory[0, 1:3].copy()
    qdot = trajectory[0, 3:5].copy()
    robot.display(q)
    time.sleep(2)

    for i in range(1, n_steps):
        tau = trajectory[i, 5:]
        q, qdot = step_with_knode(model, q, qdot, tau, DT)
        robot.display(q)


def main():
    parser = argparse.ArgumentParser(description="Benchmark a trained KNODE run across multiple trajectories (horizon vs open-loop MSE).")
    parser.add_argument("--run", type=str, required=True, help="Run directory name under runs/ (or absolute path to a run folder)")
    parser.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch (default: latest)")
    parser.add_argument("--dataset", type=str, default="benchmark_basic", help="Dataset name under data/ (default: benchmark_basic)")
    parser.add_argument("--data-dir", type=str, default=None, help="Override path to .npz directory (default: data/<dataset>)")
    parser.add_argument("--viz", action="store_true", help="Enable Meshcat visualization on the test data and stops everything else")

    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    data_path = args.data_dir or os.path.join(DATA_DIR, args.dataset)

    model, epoch, model_cfg = load_model(run_dir, epoch=args.epoch)
    trajectories = load_data(data_path)

    nominal_model = make_nominal_model(model_cfg)

    out_dir = os.path.join(run_dir, "outputs", "benchmark")
    os.makedirs(out_dir, exist_ok=True)

    max_h = max(HORIZONS)
    print(f"Run:      {run_dir}")
    print(f"Epoch:    {epoch}")
    print(f"Data:     {data_path} ({len(trajectories)} trajectories)")
    print(f"Horizons: {HORIZONS}")

    if args.viz:
        for name, traj in trajectories:
            print("------------------------------------")
            print(f"Visualizing: {name}")
            print("------------------------------------")
            visualize_robot_simulation(model, traj)
    else:

        knode_all = []
        og_all = []
        per_traj = []

        for name, traj in trajectories:
            # Only need max horizon + 1 steps for evaluate_horizon_loss
            traj_eval = traj[: max_h + 1]

            knode_raw = evaluate_horizon_loss(model, traj_eval, HORIZONS)
            knode_f = losses_to_floats(knode_raw)
            knode_all.append(knode_f)

            og_f = None
            og_raw = None
            if nominal_model is not None:
                og_raw = evaluate_horizon_loss(nominal_model, traj_eval, HORIZONS)
                og_f = losses_to_floats(og_raw)
                og_all.append(og_f)

            print_horizon_table(f"Trajectory: {name}", knode_f, og_f)
            per_traj.append({"name": name, "knode": knode_f, "og": og_f})

            _, plot_path = horizon_losses_comparison(
                knode_raw, og_raw,
                label=f"benchmark_{name}_epoch_{epoch}",
                output_dir=out_dir, show=False,
            )
            print(f"Saved plot -> {plot_path}")

        knode_mean = mean_losses(knode_all)
        og_mean = mean_losses(og_all) if og_all else None
        print_horizon_table(
            f"OVERALL mean over {len(trajectories)} trajectories (epoch {epoch})",
            knode_mean,
            og_mean,
        )

        knode_plot = {
            h: {k: torch.tensor(v) for k, v in metrics.items()}
            for h, metrics in knode_mean.items()
        }
        og_plot = None
        if og_mean is not None:
            og_plot = {
                h: {k: torch.tensor(v) for k, v in metrics.items()}
                for h, metrics in og_mean.items()
            }
        _, plot_path = horizon_losses_comparison(
            knode_plot, og_plot,
            label=f"benchmark_mean_epoch_{epoch}",
            output_dir=out_dir, show=False,
        )
        print(f"Saved plot -> {plot_path}")

        summary = {
            "run_dir": run_dir,
            "epoch": epoch,
            "data_path": data_path,
            "horizons": HORIZONS,
            "model_cfg": model_cfg,
            "per_trajectory": per_traj,
            "overall_mean": {"knode": knode_mean, "og": og_mean},
        }
        summary_path = os.path.join(out_dir, f"benchmark_summary_epoch_{epoch}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nWrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
