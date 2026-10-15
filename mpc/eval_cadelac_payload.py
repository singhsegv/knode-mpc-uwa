"""
Evaluate knode-MPC on CaDeLaC Panda joint EXC tracking across payload envs.

Matches CaDeLaC protocol (scaled down by default for local runs):
  - 50 Hz control, N=12, T=0.24 s
  - Fourier EXC references, 10 s
  - Plant: MuJoCo Panda + random EE mass; MPC model: Pinocchio panda.xml (no payload)
  - Metrics: per-joint RMS q / qd tracking error

Usage:
  # Nominal MPC baseline
  python -m mpc.eval_cadelac_payload --config panda_exc --robot panda

  # KNODE residual MPC
  python -m mpc.eval_cadelac_payload --config panda_exc --robot panda \\
      --knode-run runs/<stamp>_panda_payload_exc_panda_q_qdot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from robots.panda_payload import build_payload_models
from robots.simulator import Manipulator, load_robot_config
from utils.exc_trajectory import mpc_reference_horizon, sample_valid_exc_trajectory


def _load_mpc_class(solver_name):
    if solver_name == "ipopt":
        from manipulator_mpc import ManipulatorControl
        return ManipulatorControl
    if solver_name == "fatrop":
        from fatrop_manipulator_mpc import FatropManipulatorControl
        return FatropManipulatorControl
    if solver_name == "acados":
        from acados_manipulator_mpc import AcadosManipulatorControl
        return AcadosManipulatorControl
    raise ValueError(f"Unknown solver '{solver_name}'")


def run_closed_loop(plant, mpc, traj, dt, horizon, viz=False, realtime=False):
    q0 = traj["q"][0].copy()
    qd0 = traj["qd"][0].copy()
    plant.move_to(q0)
    if viz:
        plant.enable_visualization()
        plant.display(q0)
    x = np.concatenate([q0, qd0])

    n_steps = len(traj["t"])
    hist_q = np.empty((n_steps, plant.nq))
    hist_qd = np.empty((n_steps, plant.nq))
    hist_q_ref = np.empty((n_steps, plant.nq))
    hist_qd_ref = np.empty((n_steps, plant.nq))
    hist_u = np.empty((n_steps, plant.nv))
    solve_times = []

    u_prev = None
    x_prev = None
    for i in range(n_steps):
        t = traj["t"][i]
        X_ref = mpc_reference_horizon(traj, t, horizon, dt)

        t0 = time.perf_counter()
        U, X_opt = mpc.solve(x, X_ref, u_guess=u_prev, x_guess=x_prev)
        solve_times.append(time.perf_counter() - t0)

        u = U[:, 0]
        hist_q[i] = x[: plant.nq]
        hist_qd[i] = x[plant.nq :]
        hist_q_ref[i] = traj["q"][i]
        hist_qd_ref[i] = traj["qd"][i]
        hist_u[i] = u

        q_next, qd_next = plant.plant_step(x[: plant.nq], x[plant.nq :], u, dt)
        x = np.concatenate([q_next.flatten(), qd_next.flatten()])

        if viz:
            plant.display(x[: plant.nq])
            if realtime:
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, dt - elapsed))

        u_prev = np.zeros_like(U)
        u_prev[:, :-1] = U[:, 1:]
        u_prev[:, -1] = U[:, -1]
        x_prev = np.zeros_like(X_opt)
        x_prev[:, 0] = x
        x_prev[:, 1:-1] = X_opt[:, 2:]
        x_prev[:, -1] = X_opt[:, -1]

    rms_q = np.sqrt(np.mean((hist_q_ref - hist_q) ** 2, axis=0))
    rms_qd = np.sqrt(np.mean((hist_qd_ref - hist_qd) ** 2, axis=0))
    return {
        "rms_q": rms_q,
        "rms_qd": rms_qd,
        "solve_ms_mean": float(np.mean(solve_times) * 1000.0),
        "q": hist_q,
        "qd": hist_qd,
        "q_ref": hist_q_ref,
        "qd_ref": hist_qd_ref,
        "u": hist_u,
    }


def main():
    parser = argparse.ArgumentParser(description="CaDeLaC-style Panda payload MPC eval")
    parser.add_argument("--config", default="panda_exc", help="config/mpc/*.json name")
    parser.add_argument("--robot", default="panda")
    parser.add_argument("--solver", default="acados", choices=["acados", "ipopt", "fatrop"])
    parser.add_argument("--knode-run", default=None, help="runs/<dir> with trained residual")
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--n-eval-envs", type=int, default=None)
    parser.add_argument("--runs-per-env", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--viz", action="store_true", help="Open MuJoCo passive viewer during closed-loop eval")
    parser.add_argument("--realtime", action="store_true", help="Sleep to match control dt (only useful with --viz)")
    args = parser.parse_args()

    cfg_path = os.path.join(PROJECT_ROOT, "config", "mpc", f"{args.config}.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    robot_cfg, _ = load_robot_config(args.robot)
    # Inject CaDeLaC-matched cost weights into robot config for acados builder
    if "Q_q" in cfg:
        robot_cfg = dict(robot_cfg)
        robot_cfg["mpc_Q_q"] = cfg["Q_q"]
        robot_cfg["mpc_Q_qd"] = cfg["Q_qd"]
        robot_cfg["mpc_R"] = cfg["R"]

    dt = float(cfg["dt"])
    horizon = int(cfg["mpc_horizon"])
    duration = float(cfg["duration"])
    seed = int(cfg.get("seed", 0))
    n_train = int(cfg.get("n_train_envs", 0))
    n_eval = int(args.n_eval_envs or cfg["n_eval_envs"])
    runs = int(args.runs_per_env or cfg["runs_per_env"])
    n_envs = n_train + n_eval

    plant_models, _, payloads = build_payload_models(
        num_envs=n_envs, seed=seed, include_nominal=True
    )

    enable_knode = args.knode_run is not None
    Control = _load_mpc_class(args.solver)
    mpc = Control(
        horizon=horizon,
        dt=dt,
        enable_knode=enable_knode,
        run_dir=args.knode_run,
        epoch=args.epoch,
        robot_config=robot_cfg,
    )

    results = {
        "env_id": [],
        "mass": [],
        "rms_q": [],
        "rms_qd": [],
        "solve_ms_mean": [],
    }

    eval_ids = range(n_train, n_envs)
    print(f"Evaluating envs {list(eval_ids)} × {runs} runs | knode={enable_knode}")

    for env_id in eval_ids:
        plant = Manipulator(robot_config=robot_cfg, mj_model=plant_models[env_id])
        mass = None if payloads[env_id] is None else payloads[env_id].mass
        print(f"\n### Env {env_id} mass={mass} ###")
        for run in range(runs):
            traj = sample_valid_exc_trajectory(
                duration=duration, dt=dt, seed=seed + 10_000 + 100 * env_id + run
            )
            out = run_closed_loop(
                plant, mpc, traj, dt, horizon, viz=args.viz, realtime=args.realtime
            )
            results["env_id"].append(env_id)
            results["mass"].append(mass)
            results["rms_q"].append(out["rms_q"])
            results["rms_qd"].append(out["rms_qd"])
            results["solve_ms_mean"].append(out["solve_ms_mean"])
            print(
                f"  run {run}: RMS q={np.mean(out['rms_q']):.4f} "
                f"qd={np.mean(out['rms_qd']):.4f} "
                f"solve={out['solve_ms_mean']:.1f} ms"
            )

    rms_q = np.asarray(results["rms_q"])
    rms_qd = np.asarray(results["rms_qd"])
    print("\n#### Aggregate ####")
    print(f"mean RMS q  per joint:\n{np.mean(rms_q, axis=0)}")
    print(f"mean RMS qd per joint:\n{np.mean(rms_qd, axis=0)}")
    print(f"mean RMS q  (scalar): {np.mean(rms_q):.6f}")
    print(f"mean RMS qd (scalar): {np.mean(rms_qd):.6f}")
    print(f"mean solve: {np.mean(results['solve_ms_mean']):.2f} ms")

    out_dir = args.out or os.path.join(PROJECT_ROOT, "mpc", "cadelac_eval_outputs")
    os.makedirs(out_dir, exist_ok=True)
    tag = "knode" if enable_knode else "nominal"
    out_path = os.path.join(out_dir, f"{tag}_n{n_eval}_r{runs}.npz")
    np.savez(
        out_path,
        env_id=np.asarray(results["env_id"]),
        mass=np.asarray(results["mass"], dtype=object),
        rms_q=rms_q,
        rms_qd=rms_qd,
        solve_ms_mean=np.asarray(results["solve_ms_mean"]),
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
