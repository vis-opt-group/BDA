from collections import OrderedDict

import tensorflow as tf

from boml import utils
from boml.upper_iter.outer_grad import BOMLOuterGrad


class BOMLOuterGradSimple(BOMLOuterGrad):
    def __init__(self, inner_method="Simple", history=None, name="BMLOuterOptSimple"):
        super(BOMLOuterGradSimple, self).__init__(name)
        self._inner_method = inner_method
        self._history = history or []
        self._reverse_initializer = tf.no_op()
        self.warp_lambda = tf.cast(1.0, tf.float32)
        self.reptile_initializer = tf.no_op()

    # noinspection SpellCheckingInspection

    def compute_gradients(
        self, outer_objective, inner_grad, meta_param=None, param_dict=OrderedDict()
    ):
        """
        Function that adds to the computational graph all the operations needend for computing
        the hypergradients in a "dynamic" way, without unrolling the entire optimization graph.
        The resulting computation, while being roughly 2x more expensive then unrolling the
        optimizaiton dynamics, requires much less (GPU) memory and is more flexible, allowing
        to set a termination condition to the parameters optimizaiton routine.

        :param inner_grad: OptimzerDict object resulting from the inner objective optimization.
        :param outer_objective: A loss function for the outer parameters (scalar tensor)
        :param meta_param: Optional list of outer parameters to consider. If not provided will get all variables in the
                            hyperparameter collection in the current scope.
        :param param_dict: dictionary of parameters to specify different methods
        :return: list of outer parameters involved in the computation
        """
        meta_param = super(BOMLOuterGradSimple, self).compute_gradients(
            outer_objective, inner_grad, meta_param
        )

        with tf.variable_scope(outer_objective.op.name):
            """
            if len(meta_param) == len(list(optimizer_dict.state)):
                doo_dhypers = tf.gradients(outer_objective,list(optimizer_dict.state))
                #doo_dhypers = tf.gradients(outer_objective, list(optimizer_dict.model_fast_weights.values()))
            else:
                """
            if param_dict["use_warp"]:
                doo_dhypers = (
                    [self.warp_lambda * outer_param for outer_param in inner_grad.outer_param_tensor] + inner_grad.model_param_tensor
                )
                doo_dhypers += tf.gradients(
                    outer_objective, meta_param[len(doo_dhypers):]
                )
            else:
                doo_dhypers = tf.gradients(
                    outer_objective,
                    list(inner_grad.state) + meta_param[len(inner_grad.state):],
                )

            for h, doo_dh in zip(meta_param, doo_dhypers):
                assert doo_dh is not None, BOMLOuterGrad._ERROR_HYPER_DETACHED.format(
                    doo_dh
                )
                self._hypergrad_dictionary[h].append(doo_dh)

            return meta_param

    def apply_gradients(
        self,
        inner_objective_feed_dicts=None,
        outer_objective_feed_dicts=None,
        initializer_feed_dict=None,
        param_dict=OrderedDict(),
        global_step=None,
        session=None,
    ):

        ss = session or tf.get_default_session()

        self._history.clear()

        _fd = utils.maybe_call(initializer_feed_dict, utils.maybe_eval(global_step, ss))
        self._save_history(ss.run(self.initialization, feed_dict=_fd))

    def _save_history(self, weights):
        self._history.append(weights)
