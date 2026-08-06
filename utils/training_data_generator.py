"""
Similar to data_generator.py but reads from json.
"""
import json
import os
import sys
import time
import torch
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robots.simulator import PlanarArm
from knode.model import PlanarArmKnodeModel
import utils.model_utils as model_utils
from utils.model_utils import latest_checkpoint_epoch, load_model_checkpoint
from utils.trajectory_generator import circle_trajectory, lemniscate_trajectory, line_trajectory

# Common thingy for all functions
robot = PlanarArm(enable_viz=False)
# robot = PlanarArm(enable_viz=False, gravity=np.array([0.0, -11.0, 0.0]))
# robot = PlanarArm(enable_viz=False, gravity=np.array([0.0, 0.0, 0.0]))

DOF = 2
Kp = np.array([50.0, 50.0])
Kd = np.array([4.0, 4.0])

DATASETS_DIR = os.path.join(PROJECT_ROOT, "config", "datasets")
DEFAULT_CONFIG_NAME = "standalone"
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
PLOT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def plot_collected_trajectory(
        t_log, q_log, qdot_log, qdd_fb_log, tau_log,
        trajectory=None, max_steps=None, output_dir=PLOT_OUTPUT_DIR, show=False,
    ):
    """
    Plot collected rollout: q, qdot, qdd_fb, tau in joint space; EE path separately.

    t_log: (N,) time [s]
    q_log, qdot_log, qdd_fb_log, tau_log: (N, 2)
    """
    n = len(t_log)
    if max_steps is not None:
        n = min(n, max_steps)

    time = t_log[:n]
    q = q_log[:n]
    qdot = qdot_log[:n]
    qdd_fb = qdd_fb_log[:n]
    tau = tau_log[:n]
    xy = robot.ee_positions(q)

    suffix = f"_{trajectory}" if trajectory is not None else ""
    title_suffix = f" ({trajectory})" if trajectory is not None else ""

    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    fig.suptitle(f"Collected trajectory{title_suffix}", fontsize=13)

    row_specs = [
        (q, ["q1", "q2"]),
        (qdot, ["qdot1", "qdot2"]),
        (qdd_fb, ["qddot1", "qddot2"]),
        (tau, ["tau1", "tau2"]),
    ]
    for row, (data, labels) in enumerate(row_specs):
        for i in range(2):
            ax = axes[row, i]
            ax.plot(time, data[:, i], "b-", linewidth=1.2)
            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"collected_trajectory{suffix}.png")
    fig.savefig(save_path, dpi=150)

    xy_fig, xy_ax = plt.subplots(figsize=(6, 6))
    xy_ax.plot(xy[:, 0], xy[:, 1], "b-", linewidth=1.5)
    xy_ax.set_xlabel("x [m]")
    xy_ax.set_ylabel("y [m]")
    xy_ax.set_title(f"Task-space path{title_suffix}")
    xy_ax.axis("equal")
    xy_ax.grid(True, alpha=0.3)
    xy_path = os.path.join(output_dir, f"collected_task_space_path{suffix}.png")
    xy_fig.tight_layout()
    xy_fig.savefig(xy_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(xy_fig)

    # print(f"Saved plots -> {save_path}, {xy_path}")
    return fig, save_path


def ee_fk(q):
    """Planar 2R FK; returns EE (x, y) in the arm plane."""
    q1, q2 = q
    L1, L2 = 1.0, 1.0
    c1, c12 = q1, q1 + q2
    x = L1 * np.cos(c1) + L2 * np.cos(c12)
    y = L1 * np.sin(c1) + L2 * np.sin(c12)
    return np.array([x, y])


def ik_residual(q_seed, x_des, y_des):
    sol = ee_fk(q_seed)
    x, y = sol[0], sol[1]
    return x - x_des, y - y_des


def step_with_knode(model, q, qdot, tau, dt):
    """One RK4 step of PlanarArmKnodeModel: (q, qdot, tau) -> (q_next, qdot_next)."""
    model.eval()
    with torch.no_grad():
        y0 = torch.tensor(np.concatenate([q, qdot, tau], axis=0)[None, :], dtype=torch.float64)
        timesteps = torch.tensor([0.0, dt], dtype=torch.float64)
        pred = model(y0, timesteps)  # (2, 1, 6)
        nxt = pred[-1, 0].detach().cpu().numpy()
        
    return nxt[:2].copy(), nxt[2:4].copy()


def collect(duration, trajectory, cx, cy, omega, a=None, r=None, lx0=None, ly0=None, lx1=None, ly1=None, enable_viz=False, knode_model=None, dt=None):
    """
    Common: omega, duration
    Circle: cx, cy, r
    Lemniscate: cx, cy, a
    Line: lx0, ly0, lx1, ly1
    """
    n_steps = int(duration / dt)
    t_log = np.empty(n_steps)
    q_log = np.empty((n_steps, DOF))
    qdot_log = np.empty((n_steps, DOF))
    qdd_fb_log = np.empty((n_steps, DOF))
    tau_log = np.empty((n_steps, DOF))
    des_ee_pos_log = np.empty((n_steps, 2))
    des_ee_vel_log = np.empty((n_steps, 2))
    des_ee_acc_log = np.empty((n_steps, 2))

    if trajectory == "line":
        x_start, y_start = 0.5 * (lx0 + lx1), 0.5 * (ly0 + ly1)
    elif trajectory == "circle":
        x_start, y_start = cx + r, cy
    else:
        x_start, y_start = cx, cy

    q_initial = fsolve(ik_residual, [0.5, 0.5], args=(x_start, y_start))
    robot.move_to(q_initial)
    q = q_initial.copy()

    if enable_viz:
        robot.display(q)
        time.sleep(2)

    qdot = np.zeros(DOF)
    q_prev = q.copy() # Just as a seed for the solver

    for i in range(n_steps):
        t = i * dt
        t_log[i] = t

        if trajectory == "circle":
            x, y, xdot, ydot, xddot, yddot = circle_trajectory(t, r=r, cx=cx, cy=cy, omega=omega)
        elif trajectory == "lemniscate":
            x, y, xdot, ydot, xddot, yddot = lemniscate_trajectory(t, a=a, cx=cx, cy=cy, omega=omega)
        elif trajectory == "line":
            x, y, xdot, ydot, xddot, yddot = line_trajectory(t, lx0, ly0, lx1, ly1, omega=omega)

        # Holding a static pose
        # x, y, xdot, ydot, xddot, yddot = cx, cy, 0.0, 0.0, 0.0, 0.0

        des_ee_pos_log[i] = (x, y)
        des_ee_vel_log[i] = (xdot, ydot)
        des_ee_acc_log[i] = (xddot, yddot)

        # ---------------------------------------------
        # Computer Torque Controller Implementation
        # Why q_des in J and not the current q? Dafuq?
        # ---------------------------------------------
        q_des = fsolve(ik_residual, q_prev, args=(x, y))
        J = robot.ee_jacobian(q_des)
        J_inv = np.linalg.inv(J)

        qdot_des = J_inv @ np.array([xdot, ydot])
        qddot_des = J_inv @ np.array([xddot, yddot]) # TODO: Missing J dot thingy
        q_prev = q_des.copy()

        e = q_des - q
        ed = qdot_des - qdot
        qdd_fb = qddot_des + Kp * e + Kd * ed
        tau = robot.inverse_dynamics(q, qdot, qdd_fb).copy()
        # tau = np.array([0.0, 0.0, 0.0])

        # TODO: We might be sketchy: Check data written at the beginning/end of the step
        q_log[i] = q.copy()
        qdot_log[i] = qdot.copy()
        qdd_fb_log[i] = qdd_fb.copy()
        tau_log[i] = tau.copy()

        if enable_viz:
            robot.display(q)

        # These get saved in the next iteration
        # q, qdot = robot.step_rk4(q, qdot, tau, DT)
        if knode_model is not None:
            q, qdot = step_with_knode(knode_model, q, qdot, tau, dt)
        else:
            q, qdot = robot.step_torchdiffeq(q, qdot, tau, dt)

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
        "meta": {
            "trajectory": trajectory,
            "duration": duration,
            "n_steps": n_steps,
            "r": r,
            "a": a,
            "cx": cx,
            "cy": cy,
            "omega": omega,
            "lx0": lx0,
            "ly0": ly0,
            "lx1": lx1,
            "ly1": ly1,
        },
    }


def collect_from_spec(spec, enable_viz=False, knode_model=None, dt=None):
    trajectory = spec["type"]
    duration = spec.get("duration", -1)
    omega = spec.get("omega", -1)

    return collect(
        duration=duration,
        trajectory=trajectory,
        cx=spec.get("cx"),
        cy=spec.get("cy"),
        omega=omega,
        a=spec.get("a"),
        r=spec.get("r"),
        lx0=spec.get("lx0"),
        ly0=spec.get("ly0"),
        lx1=spec.get("lx1"),
        ly1=spec.get("ly1"),
        enable_viz=enable_viz,
        knode_model=knode_model,
        dt=dt,
    )


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


def main():
    parser = argparse.ArgumentParser(description="Collect trajectory training data from a JSON config at 500 Hz (Pinocchio).")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_NAME, help="JSON filename under configurations/datasets/")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for .npz files (default: data/<config name>/)",)
    parser.add_argument("--viz", action="store_true", help="Enable Meshcat visualization")
    parser.add_argument("--name", type=str, help="Custom name override")
    parser.add_argument("--custom-dynamics", action="store_true", help="Enable custom NN for forward integration")
    
    args = parser.parse_args()

    config_path = os.path.join(DATASETS_DIR, f"{args.config}.json")
    with open(config_path) as f:
        config = json.load(f)

    if args.name:
        combo_name = args.name
    else:
        combo_name = config["name"]

    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, combo_name)

    if args.viz:
        robot.enable_visualization()

    model = None

    dt = config["dt"]

    # Only for Standalone script
    if args.custom_dynamics:
        model_cfg = {"step_size": DT, "nominal_only": False, "perfect_nominal": False, "hidden_layers": 2,  "hidden_neurons": 128, "activation": "tanh"}
        model = PlanarArmKnodeModel(model_cfg).to(torch.float64)
        run_dir = "/Users/singhsegv/upenn-research-scalar/implementations/learning-manipulator-dynamics/knode-mpc-planar-arm/runs/20260715_213721_evaluation1_basic1"
        model_utils.CHECKPOINT_DIR = os.path.join(run_dir, "checkpoints")
        epoch = latest_checkpoint_epoch()

        if epoch is None:
            raise FileNotFoundError(f"No checkpoints found under {os.path.join(SCRIPT_DIR, 'checkpoints')}")

        load_model_checkpoint(model, epoch)
        print(f"Loaded KNODE checkpoint epoch {epoch}")

    # Only creates the training trajectories
    specs = config[f"train_trajectories"]
    for i, spec in enumerate(specs):
        output_path = os.path.join(output_dir, f"{i:02d}_{spec['type']}.npz")
        log = collect_from_spec(spec, enable_viz=args.viz, knode_model=model, dt=dt)
        save_log(output_path, log)
        print(f"Saved {len(log['t'])} samples at {1.0 / dt:.0f} Hz -> {output_path}")


if __name__ == "__main__":
    main()
