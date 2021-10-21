from __future__ import print_function, absolute_import, division

from collections import OrderedDict

import tensorflow as tf

from boml.optimizer.opt import BOMLOpt

# import numpy as np

GRADIENT_NONE_MESSAGE = (
    "WARNING: the gradient w.r.t.the tf.Variable\n {}\n is None;\n "
    "Check the computational graph of the inner objective, and be sure you\n"
    "are not considering including variables that should not be there among the\n"
    "inner variables."
)


class BOMLOptMomentum(BOMLOpt, tf.train.MomentumOptimizer):
    def __init__(
        self,
        learning_rate,
        momentum,
        use_locking=False,
        name="Momentum",
        use_nesterov=False,
    ):
        assert use_nesterov is False, "Nesterov momentum not implemented yet..."
        super(BOMLOptMomentum, self).__init__(
            learning_rate, momentum, use_locking, name, use_nesterov
        )

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        # filter_hypers

        update_op = super(BOMLOptMomentum, self).apply_gradients(
            grads_and_vars, global_step, name
        )

        # builds up the dynamics here
        mn = self.get_slot_names()[0]
        dynamics = OrderedDict()
        for g, w in grads_and_vars:
            assert g is not None, GRADIENT_NONE_MESSAGE.format(w)

            m = self.get_slot(w, mn)
            mk = tf.cast(self._momentum_tensor, m.dtype) * m + g
            wk = w - tf.cast(self._learning_rate_tensor, mk.dtype) * mk
            dynamics[w] = wk
            dynamics[m] = mk

        return update_op, dynamics

    def __str__(self):
        return "{}-lr={}-m={}".format(self._name, self._learning_rate, self._momentum)

    @property
    def optimizer_params_tensor(self):
        return super(BOMLOptMomentum, self).optimizer_params_tensor + [
            self._momentum_tensor
        ]
