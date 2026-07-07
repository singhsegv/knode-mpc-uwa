"""
https://gemini.google.com/app/9d5071a08a5bdcd5
Not needed most probably.
"""
import torch
import pinocchio as pin
import numpy as np

class PinocchioForwardDynamics(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q, v, tau, model, data):
        # 1. Convert PyTorch tensors to NumPy arrays
        q_np = q.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        tau_np = tau.detach().cpu().numpy()
        
        # 2. Compute forward dynamics (Articulated Body Algorithm)
        pin.aba(model, data, q_np, v_np, tau_np)
        ddq_np = data.ddq.copy()
        
        # 3. Save state for the backward pass
        ctx.model = model
        # Note: In multi-threaded environments, passing 'data' can cause race conditions. 
        # It's often safer to instantiate a new data object in backward() if parallelized.
        ctx.data = data 
        ctx.save_for_backward(q, v, tau)
        
        # 4. Return the result as a PyTorch tensor
        return torch.tensor(ddq_np, dtype=q.dtype, device=q.device)

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Retrieve saved variables
        q, v, tau = ctx.saved_tensors
        model = ctx.model
        data = ctx.data
        
        q_np = q.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        tau_np = tau.detach().cpu().numpy()
        
        # 2. Compute analytical derivatives in Pinocchio
        pin.computeABADerivatives(model, data, q_np, v_np, tau_np)
        
        # Pinocchio stores the Jacobians in the data object:
        # ddq_dq is [nv, nv], ddq_dv is [nv, nv], Minv (ddq_dtau) is [nv, nv]
        ddq_dq = torch.tensor(data.ddq_dq, dtype=q.dtype, device=q.device)
        ddq_dv = torch.tensor(data.ddq_dv, dtype=q.dtype, device=q.device)
        ddq_dtau = torch.tensor(data.Minv, dtype=q.dtype, device=q.device)
        
        # 3. Apply the chain rule: grad_input = grad_output @ Jacobian
        # (Assuming grad_output is of shape [nv])
        grad_q = grad_output @ ddq_dq
        grad_v = grad_output @ ddq_dv
        grad_tau = grad_output @ ddq_dtau
        
        # 4. Return gradients. Must match the number of inputs to forward()
        # model and data are not tensors, so their gradients are None
        return grad_q, grad_v, grad_tau, None, None

# --- Example Usage ---
# model = pin.buildModelFromUrdf("robot.urdf")
# data = model.createData()
# q = torch.randn(model.nq, requires_grad=True)
# v = torch.randn(model.nv, requires_grad=True)
# tau = torch.randn(model.nv, requires_grad=True)
#
# ddq = PinocchioForwardDynamics.apply(q, v, tau, model, data)
# loss = ddq.sum()
# loss.backward()