"""
TODO: Read about this format
"""
import torch
import numpy as np
import casadi as ca

class ResidualNNCallback(ca.Callback):
    """
    Black-box residual dynamics: q (2,) -> delta_qdd (2,)
    CasADi sees this as an opaque node; derivatives come from get_jacobian().
    """
    def __init__(self, forward_fn, jacobian_fn, name="residual_nn"):
        ca.Callback.__init__(self)
        self.forward_fn = forward_fn
        self.jacobian_fn = jacobian_fn
        self.construct(name, {})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(2)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(2)

    # --- forward pass (numeric, called during IPOPT evals) ---
    def eval(self, arg):
        q = np.array(arg[0].full()).reshape(2)
        return [self.forward_fn(q)]
    
    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        parent = self

        class ResidualNNJacobian(ca.Callback):
            def __init__(self):
                ca.Callback.__init__(self)
                self.construct(name, {})

            def get_n_in(self):
                return 2

            def get_n_out(self):
                return 1

            def get_sparsity_in(self, i):
                if i == 0:
                    return ca.Sparsity.dense(2)   # q

                return ca.Sparsity.dense(2)       # f(q) — can ignore in eval

            def get_sparsity_out(self, i):
                return ca.Sparsity.dense(2, 2)

            def eval(self, arg):
                q = np.array(arg[0].full()).reshape(2)
                return [parent.jacobian_fn(q)]   # 2x2 matrix

        self.jac_callback = ResidualNNJacobian()
        return self.jac_callback