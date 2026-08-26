import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "config", "mpc")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "benchmark_outputs")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robots.simulator import Manipulator, load_robot_config

from trajectory_utils import (
    compute_task_space_tracking_stats,
    get_task_space_trajectory_mpc,
    get_trajectory_start_position,
    inverse_kinematics,
    plot_task_space_tracking,
)

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
    raise ValueError(f"Unknown solver '{solver_name}'. Use ipopt, fatrop, or acados.")


def get_plant_and_controller(dt=0.001, mpc_horizon=20, enable_knode=False, knode_path=None, robot_config=None, solver="ipopt"):
    plant = Manipulator(robot_config=robot_config, enable_viz=False)
    Control = _load_mpc_class(solver)
    mpc = Control(
        horizon=mpc_horizon, dt=dt, enable_knode=enable_knode,
        run_dir=knode_path, robot_config=robot_config,
    )
    return plant, mpc


def run_single_trajectory_mpc(spec, dt, mpc_horizon, enable_knode=False, knode_path=None, output_dir=None, traj_index=0, robot_config=None, solver="ipopt"):
    sim_time = spec["duration"]
    plant, mpc = get_plant_and_controller(
        dt=dt, mpc_horizon=mpc_horizon, enable_knode=enable_knode,
        knode_path=knode_path, robot_config=robot_config, solver=solver,
    )

    nq = plant.nq
    x_start, y_start, z_start = get_trajectory_start_position(spec)
    # x_perturb = np.random.uniform(low=0.01, high=0.015)
    # y_perturb = np.random.uniform(low=0.01, high=0.015)
    # z_perturb = np.random.uniform(low=0.01, high=0.015)

    # x_start += x_perturb
    # y_start += y_perturb
    # z_start += z_perturb

    q0, _ = inverse_kinematics(
        plant, plant.get_default_config(),
        np.array([x_start, y_start, z_start]), np.zeros(3),
    )

    plant.move_to(q0)
    x_current_state = np.concatenate([q0, np.zeros(nq)])

    history_x = [x_current_state]
    history_u = []
    solve_times = []
    u_prev = None
    x_prev = None
    current_time = 0.0

    traj_label = f"{traj_index:02d}_{spec['type']}"
    print(f"\nStarting closed-loop MPC for {traj_label} ({sim_time:.1f}s)...")

    steps = int(sim_time / dt)
    for step in range(steps):
        X_ref = get_task_space_trajectory_mpc(
            current_time,
            mpc_horizon=mpc_horizon,
            dt=dt,
            robot=plant,
            q_prev=x_current_state[:nq],
            spec=spec,
        )

        t_start = time.perf_counter()
        U_optimal, X_optimal = mpc.solve(x_current_state, X_ref, u_guess=u_prev, x_guess=x_prev)
        solve_times.append(time.perf_counter() - t_start)

        u_actual = U_optimal[:, 0]
        q_next, qdot_next = plant.step_rk4(x_current_state[:nq], x_current_state[nq:], u_actual, dt)

        x_current_state = np.concatenate([q_next.flatten(), qdot_next.flatten()])
        current_time += dt

        u_prev = np.zeros_like(U_optimal)
        u_prev[:, :-1] = U_optimal[:, 1:]
        u_prev[:, -1] = U_optimal[:, -1]

        x_prev = np.zeros_like(X_optimal)
        x_prev[:, 0] = x_current_state 
        x_prev[:, 1:-1] = X_optimal[:, 2:]
        x_prev[:, -1] = X_optimal[:, -1]

        history_x.append(x_current_state)
        history_u.append(u_actual) 

        if step % 100 == 0:
            print(
                f"  Step {step}/{steps} | Time: {current_time:.2f}s | "
                f"Solve: {solve_times[-1] * 1000:.1f}ms per single step"
            )

    history_x = np.array(history_x)
    history_u = np.array(history_u)
    tracking_stats = compute_task_space_tracking_stats(history_x, plant, dt, spec=spec)

    stats = {
        "index": traj_index,
        "type": spec["type"],
        "duration": sim_time,
        "n_steps": steps,
        "dt": dt,
        "mpc_horizon": mpc_horizon,
        "spec": spec,
        "solve_time_mean_ms": float(np.mean(solve_times) * 1000.0),
        "solve_time_max_ms": float(np.max(solve_times) * 1000.0),
        **tracking_stats,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{traj_label}_tracking.png")
        plot_task_space_tracking(
            history_x,
            plant,
            dt=dt,
            spec=spec,
            show=False,
            save_path=plot_path,
        )
        stats["plot_path"] = plot_path

    return stats, history_x, history_u


def run_closed_loop_mpc(config_path, knode_path=None, enable_knode=True, mpc_horizon=25, output_dir=None, robot_config=None, solver="ipopt"):
    with open(config_path) as f:
        config = json.load(f)

    dt = config["dt"]
    specs = config["trajectories"]
    config_name = config.get("name", os.path.splitext(os.path.basename(config_path))[0])

    if output_dir is None:
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, config_name)

    os.makedirs(output_dir, exist_ok=True)

    all_stats = []
    for i, spec in enumerate(specs):
        stats, _, _ = run_single_trajectory_mpc(
            spec=spec,
            dt=dt,
            mpc_horizon=mpc_horizon,
            enable_knode=enable_knode,
            knode_path=knode_path,
            output_dir=output_dir,
            traj_index=i,
            robot_config=robot_config,
            solver=solver,
        )
        all_stats.append(stats)

    summary = {
        "config_path": config_path,
        "config_name": config_name,
        "dt": dt,
        "mpc_horizon": mpc_horizon,
        "enable_knode": enable_knode,
        "knode_path": knode_path,
        "solver": solver,
        "trajectories": all_stats,
    }

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 104)
    print(
        f"{'idx':>3}  {'type':<12}  {'RMSE [m]':>10}  {'DTW [m]':>10}  "
        f"{'DTW mean [m]':>12}  {'max [m]':>10}  {'solve avg [ms]':>14}"
    )
    print("-" * 104)
    for stats in all_stats:
        print(
            f"{stats['index']:>3}  {stats['type']:<12}  "
            f"{stats['task_space_rmse']:>10.4f}  {stats['task_space_dtw']:>10.4f}  "
            f"{stats['task_space_dtw_mean']:>12.4f}  {stats['task_space_max_error']:>10.4f}  "
            f"{stats['solve_time_mean_ms']:>14.1f}"
        )
    print("=" * 104)
    print(f"Saved stats -> {stats_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run closed-loop MPC benchmarks from a dataset config JSON.")
    parser.add_argument("--config", type=str, default="ur5_basic", help="JSON filename under config/mpc/ (without .json)")
    parser.add_argument("--knode-path", type=str, default=None, help="Path to a KNODE run directory (enables learned dynamics in MPC)")
    parser.add_argument("--no-knode", action="store_true", help="Use nominal dynamics instead of KNODE")
    parser.add_argument("--mpc-horizon", type=int, default=10, help="MPC prediction horizon")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Folder name under mpc/benchmark_outputs/ (prefixed with YYYYMMDD_HHMMSS) "
             "for per-trajectory plots and stats.json (defaults to the config name)",
    )
    parser.add_argument("--robot", type=str, default="ur5-mpc", help="Robot config name under config/robot/")
    parser.add_argument(
        "--solver",
        type=str,
        default="ipopt",
        choices=["ipopt", "fatrop", "acados"],
        help="MPC backend: ipopt (CasADi Opti), fatrop, or acados",
    )
    args = parser.parse_args()

    knode_path = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/runs/20260811_165726_ur5_generalization_basic1"

    robot_config, _ = load_robot_config(args.robot)
    config_path = os.path.join(DATASETS_DIR, f"{args.config}.json")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        os.path.join(DEFAULT_OUTPUT_DIR, f"{timestamp}_{args.output_dir}")
        if args.output_dir is not None
        else None
    )
    run_closed_loop_mpc(
        config_path=config_path,
        knode_path=knode_path,
        enable_knode=not args.no_knode,
        mpc_horizon=args.mpc_horizon,
        output_dir=output_dir,
        robot_config=robot_config,
        solver=args.solver,
    )


if __name__ == "__main__":
    main()
