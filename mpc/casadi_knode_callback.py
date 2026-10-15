"""
CasADi Callback wrapper for the K-NODE residual MLP.

Supports residual maps R^{n_in} -> R^{n_out} (e.g. q -> Δq̈ or [q;q̇] -> Δq̈).
"""
import numpy as np
import casadi as ca


class ResidualNNCallback(ca.Callback):
    """
    Black-box residual dynamics: input (n_in,) -> delta_qdd (n_out,).
    Supplies forward + Jacobian for IPOPT/fatrop.
    """

    def __init__(
        self,
        forward_fn,
        jacobian_fn,
        nq=None,
        jacobian_forward_fn=None,
        name="residual_nn",
        n_in=None,
        n_out=None,
    ):
        ca.Callback.__init__(self)
        self.forward_fn = forward_fn
        self.jacobian_fn = jacobian_fn
        self.jacobian_forward_fn = jacobian_forward_fn
        # Backward compatible: nq means square map nq->nq
        self.n_in = int(n_in if n_in is not None else nq)
        self.n_out = int(n_out if n_out is not None else nq)
        self.nq = self.n_in  # legacy alias
        self.construct(name, {})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.n_in)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.n_out)

    def eval(self, arg):
        x = np.array(arg[0].full()).reshape(self.n_in)
        return [self.forward_fn(x)]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        parent = self
        n_in, n_out = self.n_in, self.n_out

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
                    return ca.Sparsity.dense(n_in)
                return ca.Sparsity.dense(n_out)

            def get_sparsity_out(self, i):
                return ca.Sparsity.dense(n_out, n_in)

            def eval(self, arg):
                x = np.array(arg[0].full()).reshape(n_in)
                return [parent.jacobian_fn(x)]

            def has_forward(self, nfwd):
                return parent.jacobian_forward_fn is not None and nfwd == 1

            def get_forward(self, nfwd, name, inames, onames, opts):
                class ResidualNNJacobianForward(ca.Callback):
                    def __init__(self):
                        ca.Callback.__init__(self)
                        self.construct(name, {})

                    def get_n_in(self):
                        return 5

                    def get_n_out(self):
                        return 1

                    def get_sparsity_in(self, i):
                        if i == 0:
                            return ca.Sparsity.dense(n_in)
                        if i == 1:
                            return ca.Sparsity.dense(n_out)
                        if i == 2:
                            return ca.Sparsity.dense(n_out, n_in)
                        return ca.Sparsity.dense(n_in)

                    def get_sparsity_out(self, i):
                        return ca.Sparsity.dense(n_out, n_in)

                    def eval(self, arg):
                        x = np.array(arg[0].full()).reshape(n_in)
                        x_dot = np.array(arg[3].full()).reshape(n_in)
                        return [parent.jacobian_forward_fn(x, x_dot)]

                self.fwd_callback = ResidualNNJacobianForward()
                return self.fwd_callback

        self.jac_callback = ResidualNNJacobian()
        return self.jac_callback
