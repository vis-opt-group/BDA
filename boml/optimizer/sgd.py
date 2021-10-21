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


class BOMLOptSGD(BOMLOpt, tf.train.GradientDescentOptimizer):
    def __init__(self, learning_rate, use_locking=False, name="GradientDescent"):
        super(BOMLOptSGD, self).__init__(learning_rate, use_locking, name)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """

        :param grads_and_vars:
        :param global_step:
        :param name:
        :return: gradient descent step :apply_updates; \n and corresponding dynamics
        """
        # grads_and_vars=self.soft_thresholding(grads_and_vars);
        update_op = super(BOMLOptSGD, self).apply_gradients(
            grads_and_vars, global_step, name
        )

        dynamics = OrderedDict()
        for g, w in grads_and_vars:
            assert g is not None, GRADIENT_NONE_MESSAGE.format(w)
            wk = w - tf.cast(self._learning_rate_tensor, g.dtype) * g
            dynamics[w] = wk
        return update_op, dynamics

    def __str__(self):
        return "{}-lr={}".format(self._name, self._learning_rate)
