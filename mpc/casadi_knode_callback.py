"""
WARNING: AI SLOP

CasADi Callback wrapper for the K-NODE residual MLP.

Fatrop builds an exact Lagrangian Hessian via AD through the dynamics.
A Callback with only get_jacobian() is not enough: CasADi also needs forward
mode through the Jacobian callback (second derivatives of the original map).
"""
import numpy as np
import casadi as ca


class ResidualNNCallback(ca.Callback):
    """
    Black-box residual dynamics: q (nq,) -> delta_qdd (nv,)
    Supplies forward, Jacobian, and optional forward-through-Jacobian for fatrop.
    """

    def __init__(self, forward_fn, jacobian_fn, nq, jacobian_forward_fn=None, name="residual_nn"):
        ca.Callback.__init__(self)
        self.forward_fn = forward_fn
        self.jacobian_fn = jacobian_fn
        self.jacobian_forward_fn = jacobian_forward_fn
        self.nq = nq
        self.construct(name, {})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.nq)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.nq)

    def eval(self, arg):
        q = np.array(arg[0].full()).reshape(self.nq)
        return [self.forward_fn(q)]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        parent = self
        nq = self.nq

        class ResidualNNJacobian(ca.Callback):
            def __init__(self):
                ca.Callback.__init__(self)
                self.construct(name, {})

            def get_n_in(self):
                return 2

            def get_n_out(self):
                return 1

            def get_sparsity_in(self, i):
                return ca.Sparsity.dense(nq)

            def get_sparsity_out(self, i):
                return ca.Sparsity.dense(nq, nq)

            def eval(self, arg):
                q = np.array(arg[0].full()).reshape(nq)
                return [parent.jacobian_fn(q)]

            def has_forward(self, nfwd):
                return parent.jacobian_forward_fn is not None and nfwd == 1

            def get_forward(self, nfwd, name, inames, onames, opts):
                class ResidualNNJacobianForward(ca.Callback):
                    def __init__(self):
                        ca.Callback.__init__(self)
                        self.construct(name, {})

                    def get_n_in(self):
                        # n_in + n_out + n_in * nfwd = 2 + 1 + 2
                        return 5

                    def get_n_out(self):
                        return 1

                    def get_sparsity_in(self, i):
                        if i == 0:
                            return ca.Sparsity.dense(nq)       # q
                        if i == 1:
                            return ca.Sparsity.dense(nq)       # f(q)
                        if i == 2:
                            return ca.Sparsity.dense(nq, nq)   # J(q)
                        return ca.Sparsity.dense(nq)           # forward seed on q

                    def get_sparsity_out(self, i):
                        return ca.Sparsity.dense(nq, nq)

                    def eval(self, arg):
                        q = np.array(arg[0].full()).reshape(nq)
                        q_dot = np.array(arg[3].full()).reshape(nq)
                        return [parent.jacobian_forward_fn(q, q_dot)]

                self.fwd_callback = ResidualNNJacobianForward()
                return self.fwd_callback

        self.jac_callback = ResidualNNJacobian()
        return self.jac_callback
