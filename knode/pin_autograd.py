"""
Differentiable pinocchio ABA() for using torque as NN predition.
"""
import torch
import pinocchio as pin
import numpy as np


class BatchedPinocchioABA(torch.autograd.Function):
    """
    Differentiable wrapper for Pinocchio's Articulated Body Algorithm (ABA).
    Forward pass: Computes qddot = aba(q, qdot, tau)
    Backward pass: Analytically computes gradients d(qddot)/dq, d(qddot)/dqdot, d(qddot)/dtau
    """
    @staticmethod
    def forward(ctx, q_batch, v_batch, tau_batch, model, data):
        """
        model: Pinocchio model
        """
        batch_size = q_batch.shape[0]
        nv = model.nv

        qddot_batch = torch.zeros_like(v_batch)
        d_dq_batch = torch.zeros((batch_size, nv, nv), device=q_batch.device, dtype=q_batch.dtype)
        d_dv_batch = torch.zeros((batch_size, nv, nv), device=q_batch.device, dtype=q_batch.dtype)
        d_dtau_batch = torch.zeros((batch_size, nv, nv), device=q_batch.device, dtype=q_batch.dtype) # M_inv basically

        # TODO: Check if the same model object is not storing state across different rows
        for idx in range(batch_size):
            q = q_batch[idx].detach().cpu().numpy()
            v = v_batch[idx].detach().cpu().numpy()
            tau = tau_batch[idx].detach().cpu().numpy()

            qddot = pin.aba(model, data, q, v, tau)
            qddot_batch[idx] = torch.from_numpy(qddot)

            pin.computeABADerivatives(model, data, q, v, tau)
            d_dq_batch[idx] = torch.from_numpy(data.ddq_dq)
            d_dv_batch[idx] = torch.from_numpy(data.ddq_dv)
            d_dtau_batch[idx] = torch.from_numpy(data.Minv)
        
        ctx.save_for_backward(d_dq_batch, d_dv_batch, d_dtau_batch)
        return qddot_batch

    
    @staticmethod
    def backward(ctx, grad_output):
        """
        TODO: Check if torchdiffeq will cause any issues here
        grad_output represents dL/d(qddot) passed down from the ODE solver.
        """
        d_dq_batch, d_dv_batch, d_dtau_batch = ctx.saved_tensors

        # grad_output shape is (Batch, NV). We unsqueeze to (Batch, 1, NV) for batched matrix multiplication
        grad_output_unsqueeze = grad_output.unsqueeze(1)

        # Chain rule: dL/dx = (dL/d(qddot)) @ (d(qddot)/dx)
        grad_q = torch.bmm(grad_output_unsqueeze, d_dq_batch).squeeze(1)
        grad_v = torch.bmm(grad_output_unsqueeze, d_dv_batch).squeeze(1)
        grad_tau = torch.bmm(grad_output_unsqueeze, d_dtau_batch).squeeze(1)
        
        # Return gradients for inputs (q, v, tau). Return None for static objects (model, data).
        return grad_q, grad_v, grad_tau, None, None
