"""
Collect CTC training data on CaDeLaC-style Panda + random EE payload plants.

Usage:
  python -m utils.collect_panda_payload_data --dataset panda_payload_exc
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robots.panda_payload import build_payload_models
from robots.simulator import Manipulator, load_robot_config
from utils.exc_trajectory import sample_valid_exc_trajectory
from utils.training_data_generator import save_log


DATASETS_DIR = os.path.join(PROJECT_ROOT, "config", "datasets")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def collect_exc_ctc(plant, traj, dt):
    """Closed-loop CTC tracking of a joint EXC reference on `plant`."""
    kp = plant.robot_config["kp"]
    kd = plant.robot_config["kd"]
    nq = plant.nq
    n_steps = len(traj["t"])

    q = traj["q"][0].copy()
    qdot = traj["qd"][0].copy()
    plant.move_to(q)

    t_log = np.empty(n_steps)
    q_log = np.empty((n_steps, nq))
    qdot_log = np.empty((n_steps, nq))
    qdd_fb_log = np.empty((n_steps, nq))
    tau_log = np.empty((n_steps, nq))
    des_ee = np.empty((n_steps, 3))

    for i in range(n_steps):
        q_des = traj["q"][i]
        qd_des = traj["qd"][i]
        qdd_des = traj["qdd"][i]
        e = q_des - q
        ed = qd_des - qdot
        qdd_fb = qdd_des + kp * e + kd * ed
        tau = plant.inverse_dynamics(q, qdot, qdd_fb).copy()

        t_log[i] = traj["t"][i]
        q_log[i] = q
        qdot_log[i] = qdot
        qdd_fb_log[i] = qdd_fb
        tau_log[i] = tau
        des_ee[i] = plant.ee_position(q_des)

        q, qdot = plant.plant_step(q, qdot, tau, dt)

    return {
        "t": t_log,
        "q": q_log,
        "qdot": qdot_log,
        "qdd_fb": qdd_fb_log,
        "tau": tau_log,
        "des_ee_pos": des_ee,
        "des_ee_vel": np.zeros_like(des_ee),
        "des_ee_acc": np.zeros_like(des_ee),
        "dt": dt,
        "meta": {"type": "exc_fourier", "space": "joint", "n_steps": n_steps},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="panda_payload_exc")
    parser.add_argument("--robot", default="panda")
    parser.add_argument("--out", default=None, help="Output data/<name> directory")
    args = parser.parse_args()

    cfg_path = os.path.join(DATASETS_DIR, f"{args.dataset}.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    robot_cfg, _ = load_robot_config(args.robot)
    dt = float(cfg["dt"])
    duration = float(cfg["duration"])
    seed = int(cfg.get("seed", 0))
    n_train = int(cfg["n_train_envs"])
    n_eval = int(cfg["n_eval_envs"])
    runs = int(cfg["runs_per_env"])
    include_nominal = bool(cfg.get("include_nominal", True))
    n_envs = n_train + n_eval

    plant_models, _, payloads = build_payload_models(
        num_envs=n_envs,
        seed=seed,
        include_nominal=include_nominal,
    )

    out_root = args.out or os.path.join(DATA_DIR, cfg.get("name", args.dataset))
    train_dir = os.path.join(out_root, "train")
    test_dir = os.path.join(out_root, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    payload_meta = []
    for env_id, (mj_model, payload) in enumerate(zip(plant_models, payloads)):
        split = "train" if env_id < n_train else "test"
        split_dir = train_dir if split == "train" else test_dir
        plant = Manipulator(robot_config=robot_cfg, mj_model=mj_model)

        mass = None if payload is None else payload.mass
        pos = None if payload is None else payload.pos
        payload_meta.append({"env_id": env_id, "split": split, "mass": mass, "pos": pos})
        print(f"[{split}] env {env_id}: mass={mass} pos={pos}")

        for run in range(runs):
            traj = sample_valid_exc_trajectory(
                duration=duration,
                dt=dt,
                seed=seed + 1000 * env_id + run,
            )
            log = collect_exc_ctc(plant, traj, dt)
            log["meta"]["env_id"] = env_id
            log["meta"]["run"] = run
            log["meta"]["mass"] = mass
            log["meta"]["pos"] = pos
            out_path = os.path.join(split_dir, f"env{env_id:03d}_run{run:02d}_exc.npz")
            save_log(out_path, log)
            print(f"  saved {out_path} ({len(log['t'])} steps)")

    with open(os.path.join(out_root, "payloads.json"), "w") as f:
        json.dump(payload_meta, f, indent=2)
    print(f"Done. Data in {out_root}")


if __name__ == "__main__":
    main()
