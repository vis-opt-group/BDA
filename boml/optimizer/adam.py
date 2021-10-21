from __future__ import print_function, absolute_import, division

from collections import OrderedDict

import tensorflow as tf

from boml.optimizer.opt import BOMLOpt

GRADIENT_NONE_MESSAGE = (
    "WARNING: the gradient w.r.t.the tf.Variable\n {}\n is None;\n "
    "Check the computational graph of the inner objective, and be sure you\n"
    "are not considering including variables that should not be there among the\n"
    "inner variables."
)


class BOMLOptAdam(BOMLOpt, tf.train.AdamOptimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        use_locking=False,
        name="Adam",
    ):
        super(BOMLOptAdam, self).__init__(
            learning_rate, beta1, beta2, epsilon, use_locking, name
        )

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        update_op = super(BOMLOptAdam, self).apply_gradients(
            grads_and_vars, global_step, name
        )

        mn, vn = self.get_slot_names()
        dynamics = OrderedDict()

        with tf.name_scope(name, "Adam_Dynamics"):
            try:
                b1_pow, b2_pow = self._beta1_power, self._beta2_power
            except AttributeError:
                b1_pow, b2_pow = self._get_beta_accumulators()
            lr_k = self._lr_t * tf.sqrt(1.0 - b2_pow) / (1.0 - b1_pow)

            lr_k = tf.cast(lr_k, grads_and_vars[0][0].dtype)
            self._beta1_t = tf.cast(self._beta1_t, grads_and_vars[0][0].dtype)
            self._beta2_t = tf.cast(self._beta2_t, grads_and_vars[0][0].dtype)
            self._epsilon_t = tf.cast(self._epsilon_t, grads_and_vars[0][0].dtype)

            for g, w in grads_and_vars:
                assert g is not None, GRADIENT_NONE_MESSAGE.format(w)

                m = self.get_slot(w, mn)
                v = self.get_slot(w, vn)
                mk = tf.add(
                    self._beta1_t * m, (1.0 - self._beta1_t) * g, name=m.op.name
                )
                vk = tf.add(
                    self._beta2_t * v, (1.0 - self._beta2_t) * g * g, name=v.op.name
                )

                wk = tf.subtract(
                    w, lr_k * mk / (tf.sqrt(vk + self._epsilon_t ** 2)), name=w.op.name
                )
                # IMPORTANT NOTE: epsilon should be outside sqrt as from the original implementation,
                # but this brings to numerical instability of the outergradient.

                dynamics[w] = wk
                dynamics[m] = mk
                dynamics[v] = vk

            b1_powk = b1_pow * self._beta1_t
            b2_powk = b2_pow * self._beta2_t

            dynamics[b1_pow] = b1_powk
            dynamics[b2_pow] = b2_powk
        return update_op, dynamics

    def __str__(self):
        return "{}-lr={}-b1={}-b=2{}-ep={}".format(
            self._name, self._lr, self._beta1, self._beta2, self._epsilon
        )

    @property
    def learning_rate(self):
        return self._lr

    @property
    def learning_rate_tensor(self):
        return self._lr_t

    @property
    def optimizer_params_tensor(self):
        return super(BOMLOptAdam, self).optimizer_params_tensor + [
            self._beta1_t,
            self._beta2_t,
        ]
