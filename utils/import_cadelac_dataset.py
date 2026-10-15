"""
Convert a downloaded CaDeLaC HuggingFace pickle into knode-mpc .npz rollouts.

Dataset (after `git lfs pull` or HF clone):
  cadelac/learning/datasets/panda/panda_mj_101_rand_envs_20_runs_50Hz_lqr.pkl

Keys used: labels, t, qp, qv, tau  (diff_tau / qa ignored for KNODE)

Usage:
  python -m utils.import_cadelac_dataset \\
      --pkl /path/to/panda_mj_101_rand_envs_20_runs_50Hz_lqr.pkl \\
      --out data/panda_cadelac_hf \\
      --train-envs 100 --test-envs 30
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.training_data_generator import save_log


def _parse_env_id(label: str) -> int:
    m = re.search(r"env[_]?(\d+)", str(label))
    if m:
        return int(m.group(1))
    # fallback: sequential
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-envs", type=int, default=100)
    parser.add_argument("--test-envs", type=int, default=30)
    parser.add_argument("--max-runs", type=int, default=None, help="Cap runs for a smoke test")
    args = parser.parse_args()

    # Prefer dill (CaDeLaC), fall back to pickle
    try:
        import dill as pickle
    except ImportError:
        import pickle

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    required = ["labels", "t", "qp", "qv", "tau"]
    for k in required:
        if k not in data:
            raise KeyError(f"Dataset missing key '{k}'. Found: {list(data.keys())}")

    train_dir = os.path.join(args.out, "train")
    test_dir = os.path.join(args.out, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    n_saved = 0
    for i, label in enumerate(data["labels"]):
        if args.max_runs is not None and n_saved >= args.max_runs:
            break

        env_id = _parse_env_id(label)
        if env_id < 0:
            env_id = i  # unknown labeling

        t = np.asarray(data["t"][i]).reshape(-1)
        q = np.asarray(data["qp"][i])
        qdot = np.asarray(data["qv"][i])
        tau = np.asarray(data["tau"][i])
        n = len(t)
        zeros = np.zeros((n, 3))

        log = {
            "t": t,
            "q": q,
            "qdot": qdot,
            "qdd_fb": np.asarray(data["qa"][i]) if "qa" in data else np.zeros_like(q),
            "tau": tau,
            "des_ee_pos": zeros,
            "des_ee_vel": zeros,
            "des_ee_acc": zeros,
            "dt": float(t[1] - t[0]) if n > 1 else 0.02,
            "meta": {"label": str(label), "env_id": env_id, "source": "cadelac_hf"},
        }

        split = "train" if env_id < args.train_envs else "test"
        if env_id >= args.train_envs + args.test_envs:
            continue
        out_dir = train_dir if split == "train" else test_dir
        safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(label))
        out_path = os.path.join(out_dir, f"{safe}.npz")
        save_log(out_path, log)
        n_saved += 1
        if n_saved % 50 == 0:
            print(f"saved {n_saved} rollouts...")

    print(f"Done. Wrote {n_saved} rollouts under {args.out}")


if __name__ == "__main__":
    main()
