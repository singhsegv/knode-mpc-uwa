import torch
from torch import nn
from robot import PlanarArm


def evaluate(model, test_data):
    loss_fn = nn.MSELoss()

    robot = PlanarArm()

    with torch.no_grad():
        DTYPE = torch.float64
        model.func.nominal_only = True
        model.eval()

        test_data = torch.tensor(test_data, dtype=DTYPE)

        # Breaking from (t, q, qdot, u) to (t) and (q, qdot, u)
        eval_timesteps = test_data[:, 0]
        eval_labels = test_data[:, 1:]

        dt = 0.002
        step_times = torch.tensor([0.0, dt], dtype=DTYPE)
        N = eval_labels.shape[0]
        
        y_hat = torch.zeros(N, 6, dtype=DTYPE)
        y_hat[0] = eval_labels[0].clone()

        current_state = eval_labels[0].unsqueeze(0) # (1, 6)
        pinn_current_state = eval_labels[0].unsqueeze(0) # (1, 6)
        for t in range(1, N):
            gt_compare = eval_labels[t, 0:4].squeeze(0)

            # -----------------------------------------
            # Propagation through robot's rk4
            # -----------------------------------------
            pinn_state = pinn_current_state.squeeze(0).detach().cpu().numpy()
            pinn_robo_pred = robot.step_rk4(pinn_state[:2], pinn_state[2:4], pinn_state[4:], dt)

            pinn_comparable = torch.hstack([torch.tensor(pinn_robo_pred[0][0:4]), torch.tensor(pinn_robo_pred[1][0:4])])
            pinn_current_state = torch.hstack([pinn_comparable, eval_labels[t, 4:6]]).unsqueeze(0) # For the next iteration

            print(f"pinocchio err: {pinn_comparable - gt_compare}")

            # -----------------------------------------
            # Propagation through torchdiffeq's odeint
            # -----------------------------------------
            pred = model(current_state, step_times) # (2, 1, 6) ODE thingy for timesteps
            predicted_state = pred[-1, 0].clone() # (6,)
            predicted_state[4:6] = eval_labels[t, 4:6].clone() # Model gives q, qdot; torque comes from dataset at time t
            y_hat[t] = predicted_state.clone()
            current_state = predicted_state.unsqueeze(0).clone() # For the next iteration

            print(f"err: {current_state.squeeze(0)[:4] - gt_compare}")
                       
            print("----------------------")
            
        loss = loss_fn(y_hat[:, 0:4], test_data[:, 1:5])
        print(f"Nominal-only MSE: {loss.item()}")
    
    return y_hat, loss

