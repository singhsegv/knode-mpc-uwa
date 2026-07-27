"""Verify multi-step integration uses per-step torques (ZOH), not frozen IC tau."""
import numpy as np
import torch
from torchdiffeq import odeint

from knode.model import PlanarArmKnodeModel
from robots.simulator import PlanarArm


def main():
    robot = PlanarArm()
    dt = 0.002
    cfg = {
        "step_size": dt,
        "nominal_only": True,
        "perfect_nominal": True,
        "hidden_layers": 2,
        "hidden_neurons": 128,
        "activation": "tanh",
    }
    model = PlanarArmKnodeModel(cfg).to(torch.float64)
    model.eval()

    q = np.array([0.3, -0.5])
    qdot = np.array([0.1, -0.2])
    tau0 = np.array([1.0, 0.5])
    tau1 = np.array([-0.3, 0.8])
    tau2 = np.array([0.2, -0.4])

    y0 = torch.tensor(np.concatenate([q, qdot, tau0])[None, :], dtype=torch.float64)

    # Ground truth: 3 RK4 steps with ZOH torques tau0, tau1, tau2
    q_sim, qdot_sim = q.copy(), qdot.copy()
    for tau in (tau0, tau1, tau2):
        q_sim, qdot_sim = robot.step_rk4(q_sim, qdot_sim, tau, dt)

    # Wrong: one continuous odeint with tau frozen at tau0 (old training bug)
    timesteps = torch.tensor([0.0, dt, 2 * dt, 3 * dt], dtype=torch.float64)
    wrong = odeint(
        model.func, y0, timesteps,
        method="rk4", options={"step_size": dt},
    )[-1, 0, :4].detach().numpy()

    # Correct: piecewise ZOH with scheduled torques
    step_torques = torch.tensor([[tau0, tau1, tau2]], dtype=torch.float64)
    correct = model(y0, step_torques=step_torques)[-1, 0, :4].detach().numpy()

    gt = np.concatenate([q_sim, qdot_sim])
    err_wrong = np.linalg.norm(wrong - gt)
    err_correct = np.linalg.norm(correct - gt)

    print("GT (Pinocchio ZOH):     ", gt)
    print("Wrong (frozen tau):     ", wrong, "  err=", err_wrong)
    print("Correct (step torques): ", correct, "  err=", err_correct)
    assert err_correct < 1e-10, "ZOH integration should match Pinocchio rollout"
    assert err_wrong > 1e-6, "Frozen-tau integration should diverge when tau varies"
    print("OK: multi-step ZOH matches simulator; frozen-tau path is wrong as expected.")


if __name__ == "__main__":
    main()
