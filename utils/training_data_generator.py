"""
Collect closed-loop CTC training data for n-DOF manipulators (default: UR5).
Reads trajectory specs from JSON configs under config/datasets_full_arm/.
"""
import json
import os
import sys
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import least_squares

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from knode.model import ManipulatorKnodeModel
import utils.model_utils as model_utils
from utils.model_utils import latest_checkpoint_epoch, load_model_checkpoint
from utils.trajectory_generator import sample_trajectory, get_trajectory_start

DATASETS_DIR = os.path.join(PROJECT_ROOT, "config", "datasets")
DEFAULT_CONFIG_NAME = "ur5_demo"
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
PLOT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def inverse_kinematics(robot, q_prev, pos_des, vel_des, damping=0.05, posture_weight=0.05):
    """Numerical IK + damped least-squares velocity mapping."""
    pos_des = np.asarray(pos_des, dtype=float)
    vel_des = np.asarray(vel_des, dtype=float)
    task_dim = getattr(robot, "task_space_dim", 3)
    q_prev = np.asarray(q_prev, dtype=float)

    def fk_residual(q):
        pos_err = robot.ee_position(q) - pos_des
        if robot.nq > task_dim:
            posture_err = posture_weight * (q - q_prev)
            return np.concatenate([pos_err, posture_err])

        return pos_err

    result = least_squares(fk_residual, q_prev, method="trf")
    q_des = result.x

    J = robot.ee_jacobian(q_des)
    JJt = J @ J.T
    damp = damping ** 2 * np.eye(task_dim)
    J_pinv = J.T @ np.linalg.inv(JJt + damp)
    qdot_des = J_pinv @ vel_des

    return q_des, qdot_des


def step_with_knode(model, q, qdot, tau, dt):
    """One RK4 step of ManipulatorKnodeModel: (q, qdot, tau) -> (q_next, qdot_next)."""
    nq = model.nq
    model.eval()
    with torch.no_grad():
        y0 = torch.tensor(np.concatenate([q, qdot, tau])[None, :], dtype=torch.float64)
        timesteps = torch.tensor([0.0, dt], dtype=torch.float64)
        pred = model(y0, timesteps)
        nxt = pred[-1, 0].detach().cpu().numpy()
    return nxt[:nq].copy(), nxt[nq : 2 * nq].copy()


def collect(spec, robot, enable_viz=False, knode_model=None, dt=None):
    """
    Collect one trajectory rollout with computed-torque control.
    """
    kp = robot.robot_config["kp"]
    kd = robot.robot_config["kd"]
    nq = robot.nq
    task_dim = getattr(robot, "task_space_dim", 3)
    duration = spec["duration"]
    n_steps = int(duration / dt)

    t_log = np.empty(n_steps)
    q_log = np.empty((n_steps, nq))
    qdot_log = np.empty((n_steps, nq))
    qdd_fb_log = np.empty((n_steps, nq))
    tau_log = np.empty((n_steps, nq))
    des_ee_pos_log = np.empty((n_steps, task_dim))
    des_ee_vel_log = np.empty((n_steps, task_dim))
    des_ee_acc_log = np.empty((n_steps, task_dim))

    x_start, y_start, z_start = get_trajectory_start(spec)
    pos_start = np.array([x_start, y_start, z_start])[:task_dim]

    q_initial, _ = inverse_kinematics(robot, robot.get_default_config(), pos_start, np.zeros(task_dim))
    robot.move_to(q_initial)
    q = q_initial.copy()

    if enable_viz:
        robot.display(q)
        time.sleep(2)

    qdot = np.zeros(nq)
    q_prev = q.copy()

    for i in range(n_steps):
        t = i * dt
        t_log[i] = t

        vals = sample_trajectory(t, spec)
        pos_des = np.array(vals[:3])[:task_dim]
        vel_des = np.array(vals[3:6])[:task_dim]
        acc_des = np.array(vals[6:9])[:task_dim]

        des_ee_pos_log[i] = pos_des
        des_ee_vel_log[i] = vel_des
        des_ee_acc_log[i] = acc_des

        # -------------------------------------------------------------------------------------
        # TODO: FIXME, I'm sketchy
        q_des, qdot_des = inverse_kinematics(robot, q_prev, pos_des, vel_des)
        J = robot.ee_jacobian(q_des)
        task_dim_local = J.shape[0]
        JJt = J @ J.T
        damp = 0.05 ** 2 * np.eye(task_dim_local)
        J_pinv = J.T @ np.linalg.inv(JJt + damp)
        qddot_des = J_pinv @ acc_des # TODO: Missing possibly qdot terms here like the planar thingy
        q_prev = q_des.copy()
        # -------------------------------------------------------------------------------------

        e = q_des - q
        ed = qdot_des - qdot
        qdd_fb = qddot_des + kp * e + kd * ed
        tau = robot.inverse_dynamics(q, qdot, qdd_fb).copy()

        q_log[i] = q.copy()
        qdot_log[i] = qdot.copy()
        qdd_fb_log[i] = qdd_fb.copy()
        tau_log[i] = tau.copy()

        if enable_viz:
            robot.display(q)

        if knode_model is not None:
            q, qdot = step_with_knode(knode_model, q, qdot, tau, dt)
        else:
            q, qdot = robot.step_torchdiffeq(q, qdot, tau, dt)

    meta = dict(spec)
    meta["n_steps"] = n_steps
    meta["trajectory"] = spec["type"]

    return {
        "t": t_log,
        "q": q_log,
        "qdot": qdot_log,
        "qdd_fb": qdd_fb_log,
        "tau": tau_log,
        "des_ee_pos": des_ee_pos_log,
        "des_ee_vel": des_ee_vel_log,
        "des_ee_acc": des_ee_acc_log,
        "dt": dt,
        "meta": meta,
    }


def save_log(output_path, log):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        t=log["t"],
        q=log["q"],
        qdot=log["qdot"],
        qdd_fb=log["qdd_fb"],
        tau=log["tau"],
        des_ee_pos=log["des_ee_pos"],
        des_ee_vel=log["des_ee_vel"],
        des_ee_acc=log["des_ee_acc"],
        dt=np.array(log["dt"]),
        meta=np.array(log["meta"], dtype=object),
    )


def plot_collected_trajectory(
    t_log, q_log, qdot_log, qdd_fb_log, tau_log, robot,
    trajectory=None, label=None, max_steps=None, output_dir=PLOT_OUTPUT_DIR, show=False,
    des_ee_pos=None,
):
    """Plot collected rollout in joint space and 3D task space."""
    n = len(t_log)
    if max_steps is not None:
        n = min(n, max_steps)

    time = t_log[:n]
    q = q_log[:n]
    qdot = qdot_log[:n]
    qdd_fb = qdd_fb_log[:n]
    tau = tau_log[:n]
    xyz = robot.ee_positions(q)
    nq = robot.nq
    ncols = min(3, nq)
    nrows_per_qty = int(np.ceil(nq / ncols))

    if label is not None:
        suffix = f"_{label}"
        title_suffix = f" ({label})"
    elif trajectory is not None:
        suffix = f"_{trajectory}"
        title_suffix = f" ({trajectory})"
    else:
        suffix = ""
        title_suffix = ""

    fig, axes = plt.subplots(4 * nrows_per_qty, ncols, figsize=(4 * ncols, 2.5 * 4 * nrows_per_qty), sharex=True)
    if nrows_per_qty * ncols == 1:
        axes = np.array([[axes]])
    elif nrows_per_qty == 1:
        axes = axes.reshape(4, ncols)
    else:
        axes = axes.reshape(4 * nrows_per_qty, ncols)

    fig.suptitle(f"Collected trajectory{title_suffix}", fontsize=13)

    row_specs = [
        (q, "q"),
        (qdot, "qdot"),
        (qdd_fb, "qddot"),
        (tau, "tau"),
    ]
    for block, (data, prefix) in enumerate(row_specs):
        for j in range(nq):
            row = block * nrows_per_qty + j // ncols
            col = j % ncols
            ax = axes[row, col]
            ax.plot(time, data[:, j], "b-", linewidth=1.2)
            ax.set_ylabel(f"{prefix}{j + 1}")
            ax.grid(True, alpha=0.3)

    for col in range(ncols):
        axes[-1, col].set_xlabel("time [s]")

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"collected_trajectory{suffix}.png")
    fig.savefig(save_path, dpi=150)

    task_dim = getattr(robot, "task_space_dim", 3)
    if task_dim == 3:
        from utils.robot_viz import save_task_space_3d_figure

        des = None
        if des_ee_pos is not None:
            des = des_ee_pos[:n]
        xyz_path = os.path.join(output_dir, f"collected_task_space_path{suffix}.png")
        save_task_space_3d_figure(
            robot, q, xyz, ee_desired=des,
            title=f"Task-space path (3D){title_suffix}",
            save_path=xyz_path, show=False,
        )
    else:
        xy_fig, xy_ax = plt.subplots(figsize=(6, 6))
        xy_ax.plot(xyz[:, 0], xyz[:, 1], "b-", linewidth=1.5)
        xy_ax.set_xlabel("x [m]")
        xy_ax.set_ylabel("y [m]")
        xy_ax.set_title(f"Task-space path{title_suffix}")
        xy_ax.axis("equal")
        xy_ax.grid(True, alpha=0.3)
        xyz_path = os.path.join(output_dir, f"collected_task_space_path{suffix}.png")
        xy_fig.tight_layout()
        xy_fig.savefig(xyz_path, dpi=150)
        plt.close(xy_fig)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, save_path


def main():
    parser = argparse.ArgumentParser(description="Collect trajectory training data from a JSON config.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_NAME, help="Dataset Comfiguration: JSON filename (without .json)")
    parser.add_argument("--robot-config", type=str, default="ur5", help="Robot Configuration: JSON filename (without .json)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for .npz files")
    parser.add_argument("--viz", action="store_true", help="Enable Meshcat visualization")
    parser.add_argument("--name", type=str, help="Custom output folder name override")
    parser.add_argument("--custom-dynamics", action="store_true", help="Integrate with a learned KNODE model")
    parser.add_argument("--knode-run", type=str, default=None, help="Path to KNODE run directory for custom dynamics")
    args = parser.parse_args()

    config_path = os.path.join(DATASETS_DIR, f"{args.config}.json")
    with open(config_path) as f:
        config = json.load(f)

    combo_name = args.name or config["name"]
    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, combo_name)
    dt = config["dt"]

    robot, robot_config = make_robot(args.robot_config)

    if args.viz:
        robot.enable_visualization()

    # model = None
    # if args.custom_dynamics:
    #     if args.knode_run is None:
    #         raise ValueError("--knode-run is required with --custom-dynamics")

    #     model_cfg = {
    #         "step_size": dt,
    #         "nominal_only": False,
    #         "perfect_nominal": False,
    #         "hidden_layers": 2,
    #         "hidden_neurons": 128,
    #         "activation": "tanh",
    #     }
    #     model = ManipulatorKnodeModel(model_cfg, robot_config=robot_config).to(torch.float64)
    #     model_utils.CHECKPOINT_DIR = os.path.join(args.knode_run, "checkpoints")
    #     epoch = latest_checkpoint_epoch()
    #     if epoch is None:
    #         raise FileNotFoundError(f"No checkpoints found under {model_utils.CHECKPOINT_DIR}")
    #     load_model_checkpoint(model, epoch)
    #     print(f"Loaded KNODE checkpoint epoch {epoch}")

    splits = ["train", "test"]
    for split in splits:
        key = f"{split}_trajectories"
        specs = config[key]
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for i, spec in enumerate(specs):
            output_path = os.path.join(split_dir, f"{i:02d}_{spec['type']}.npz")
            log = collect(spec, robot, enable_viz=args.viz, knode_model=None, dt=dt)
            save_log(output_path, log)
            print(f"Saved {len(log['t'])} samples at {1.0 / dt:.0f} Hz -> {output_path}")

            plot_collected_trajectory(
                t_log=log["t"], q_log=log["q"], qdot_log=log["qdot"],
                qdd_fb_log=log["qdd_fb"], tau_log=log["tau"], robot=robot,
                label=f"{i:02d}_{spec['type']}",
                trajectory=spec["type"],
                max_steps=log["meta"]["n_steps"], show=False,
                output_dir=split_dir,
                des_ee_pos=log["des_ee_pos"],
            )
            

if __name__ == "__main__":
    main()
