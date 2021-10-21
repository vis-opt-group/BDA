from __future__ import absolute_import, print_function, division

from collections import OrderedDict, deque

import numpy as np

# import py_bml.OuterOpt.outer_opt_utils as utils
import tensorflow as tf
from tensorflow.python.training import slot_creator

import boml.extension
from boml import utils
from boml.upper_iter.outer_grad import BOMLOuterGrad

RAISE_ERROR_ON_DETACHED = False


class BOMLOuterGradDarts(BOMLOuterGrad):
    def __init__(self, inner_method="Trad", name="BMLOuterGradDarts"):
        """
       Utility method to initialize truncated reverse HG (not necessarily online),
       :param name: a name for the operations and variables that will be created
       :return: BMLOuterGradDarts object
           """
        super(BOMLOuterGradDarts, self).__init__(name)
        self.epsilon = 0.0
        self._inner_method = inner_method
        self._diff_initializer = tf.no_op()
        self._darts_initializer = tf.no_op()
        self._history = deque(maxlen=2)

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
        :param param_dict: dictionary to store necessary parameters

        :return: list of outer parameters involved in the computation
        """
        meta_param = super(BOMLOuterGradDarts, self).compute_gradients(
            outer_objective, inner_grad, meta_param
        )

        with tf.variable_scope(outer_objective.op.name):

            ex = self.param_dict["experiment"]
            model = self.param_dict["experiment"].model
            loss_func = self.param_dict["loss_func"]
            outer_loss_func = self.param_dict["outer_loss_func"]
            # compute the first-order gradient of updated outer parameters with ones-step forward
            grads_outer = [
                self._create_outergradient(outer_objective, hyper)
                for hyper in meta_param
            ]

            # compute the first-order gradient of  the initial task parameters
            darts_derivatives = [
                grad for grad in tf.gradients(outer_objective, list(inner_grad.state))
            ]
            self.grad_lr =grads_outer[-1]

            # compute the differentiation part, multiplied by Epsilon
            darts_vector = tf.concat(
                axis=0, values=utils.vectorize_all(darts_derivatives)
            )
            self.epsilon = 0.01 / tf.norm(tensor=darts_vector, ord=2)
            darts_derivatives = [
                self.Epsilon * darts_derivative
                for darts_derivative in darts_derivatives
            ]
            fin_diff_part = self._create_darts_derivatives(
                var_list=inner_grad.state, darts_derivatives=darts_derivatives
            )
            self._diff_initializer = tf.group(
                self._diff_initializer,
                tf.variables_initializer(fin_diff_part),
                tf.variables_initializer(grads_outer),
            )

            right_diff_0 = dict(
                zip(
                    model.task_parameter.keys(),
                    [
                        tf.add(state, fin_diff)
                        for state, fin_diff in zip(
                            model.task_parameter.values(), fin_diff_part
                        )
                    ],
                )
            )
            left_diff_0 = dict(
                zip(
                    model.task_parameter.keys(),
                    [
                        tf.subtract(state, fin_diff)
                        for state, fin_diff in zip(
                            model.task_parameter.values(), fin_diff_part
                        )
                    ],
                )
            )

            left_diff = tf.gradients(
                loss_func(
                    pred=model.re_forward(task_parameter=left_diff_0).out,
                    label=ex.y,
                    method="MetaRepr",
                ),
                xs=meta_param,
            )

            right_diff = tf.gradients(
                loss_func(
                    pred=model.re_forward(task_parameter=right_diff_0).out,
                    label=ex.y,
                    method="MetaRepr",
                ),
                xs=meta_param,
            )

            if self._inner_method == 'Aggr':
                left_diff_outer = tf.gradients(
                    outer_loss_func(
                        pred=model.re_forward(new_input=param_dict['meta_learner'].re_forward(ex.x_).out, task_parameter=left_diff_0).out,
                        label=ex.y_,
                        method="MetaRepr",
                    ),
                    xs=meta_param,
                )
                for _, left_diff_grad, left_diff_outer_grad in zip(range(len(left_diff)), left_diff, left_diff_outer):
                    if left_diff_grad is not None:
                        if left_diff_outer_grad is not None:
                            left_diff[_] = (1-param_dict['alpha']) * left_diff_grad + param_dict['alpha'] * left_diff_outer_grad
                        else:
                            left_diff[_] = left_diff_grad
                    else:
                        if left_diff_outer_grad is not None:
                            left_diff[_] = param_dict['alpha'] * left_diff_outer_grad

                right_diff_outer = tf.gradients(
                    outer_loss_func(
                        pred=model.re_forward(new_input=param_dict['meta_learner'].re_forward(ex.x_).out, task_parameter=right_diff_0).out,
                        label=ex.y_,
                        method="MetaRepr",
                    ),
                    xs=meta_param,
                )

                for _, right_diff_grad, right_diff_outer_grad in zip(range(len(right_diff)),right_diff, right_diff_outer):
                    if right_diff_grad is not None:
                        if right_diff_outer_grad is not None:
                            right_diff[_] = (1-param_dict['alpha']) * right_diff_grad + param_dict['alpha'] * right_diff_outer_grad
                        else:
                            right_diff[_] =  right_diff_grad
                    else:
                        if right_diff_outer_grad is not None:
                            right_diff[_] = param_dict['alpha'] * right_diff_outer_grad

            # compute the second-order part and add them to the first-order item
            for grad_outer, left_dif, right_dif in zip(
                grads_outer, left_diff, right_diff
            ):
                if right_dif is not None and left_dif is not None:
                    grad_param = tf.divide(
                        tf.subtract(right_dif, left_dif), 2 * self.epsilon
                    )
                    meta_grad = self.param_dict["learning_rate"] * grad_param
                    self._darts_initializer = tf.group(
                        self._darts_initializer, grad_outer.assign_sub(meta_grad)
                    )

            for h, doo_dh in zip(meta_param, grads_outer):
                assert doo_dh is not None, BOMLOuterGrad._ERROR_HYPER_DETACHED.format(
                    doo_dh
                )
                self._hypergrad_dictionary[h].append(doo_dh)
            return meta_param

    @staticmethod
    def _create_darts_derivatives(var_list, darts_derivatives):
        derivatives = [
            slot_creator.create_slot(
                v.initialized_value(), utils.val_or_zero(der, v), "alpha"
            )
            for v, der in zip(var_list, darts_derivatives)
        ]
        [
            tf.add_to_collection(boml.extension.GraphKeys.DARTS_DERIVATIVES, der)
            for der in derivatives
        ]
        boml.extension.remove_from_collection(
            boml.extension.GraphKeys.GLOBAL_VARIABLES, *derivatives
        )
        # this prevents the 'automatic' initialization with tf.global_variables_initializer.
        return derivatives

    @staticmethod
    def _create_outergradient_from_dodh(hyper, doo_dhypers):
        """
        Creates one hyper-gradient as a variable. doo_dhypers:  initialization, that is the derivative of
        the outer objective w.r.t this hyper
        """
        hgs = slot_creator.create_slot(
            hyper, utils.val_or_zero(doo_dhypers, hyper), "outergradient"
        )
        boml.extension.remove_from_collection(
            boml.extension.GraphKeys.GLOBAL_VARIABLES, hgs
        )
        return hgs

    @staticmethod
    def _create_outergradient(outer_obj, hyper):
        return BOMLOuterGradDarts._create_outergradient_from_dodh(
            hyper, tf.gradients(outer_obj, hyper)[0]
        )

    def apply_gradients(
        self,
        inner_objective_feed_dicts=None,
        outer_objective_feed_dicts=None,
        initializer_feed_dict=None,
        param_dict=OrderedDict(),
        global_step=None,
        session=None,
    ):
        if self._inner_method == "Aggr":
            alpha = param_dict["alpha"]
            t_tensor = param_dict["t_tensor"]

        ss = session or tf.get_default_session()

        self._history.clear()

        _fd = utils.maybe_call(initializer_feed_dict, utils.maybe_eval(global_step, ss))
        self._save_history(ss.run(self.initialization, feed_dict=_fd))
        # perform one-step update to the task parameters and store  weights along the optimization path
        _fd = inner_objective_feed_dicts

        for t in range(param_dict['T']):
            if self._inner_method == "Aggr":
                _fd.update(outer_objective_feed_dicts)
                if not alpha.get_shape().as_list():
                    _fd[t_tensor] = float(1.0)
                else:
                    tmp = np.zeros((alpha.get_shape().as_list()[1], 1))
                    tmp[0][0] = 1.0
                    _fd[t_tensor] = tmp
            ss.run(self.iteration, feed_dict=_fd)
        '''
        for t in range(param_dict['T']):
            if self._inner_method == "Aggr":
                _fd.update(outer_objective_feed_dicts)
                if not alpha.get_shape().as_list():
                    _fd[t_tensor] = float(1.0)
                else:
                    tmp = np.zeros((alpha.get_shape().as_list()[1], 1))
                    tmp[0][0] = 1.0
                    _fd[t_tensor] = tmp
            self._save_history(ss.run(self.iteration, feed_dict=_fd))
    '''
        # compute the differentiation part, multiplied by Epsilon with one-step forward pass
        _fd = utils.maybe_call(
            outer_objective_feed_dicts, utils.maybe_eval(global_step, ss)
        )
        darts_init_fd = utils.merge_dicts(_fd, inner_objective_feed_dicts)
        ss.run(self._diff_initializer, feed_dict=darts_init_fd)

        #del self._history[-1]  # do not consider the final task parameters

        # compute the second-order part and add them to the first-order item
        state_feed_dict = utils.merge_dicts(
            *[
                od.state_feed_dict(h)
                for od, h in zip(sorted(self._optimizer_dicts), self._history[-1])
            ]
        )
        new_fd = utils.merge_dicts(state_feed_dict, inner_objective_feed_dicts)
        if self._inner_method == "Aggr":
            new_fd = utils.merge_dicts(new_fd, outer_objective_feed_dicts)
            # modified - mark
            if not alpha.shape.as_list():
                new_fd[t_tensor] = float(1.0)
            else:
                tmp = np.zeros((alpha.get_shape().as_list()[1], 1))
                tmp[0][0] = 1
                new_fd[t_tensor] = tmp
        new_fd = utils.merge_dicts(new_fd, outer_objective_feed_dicts)
        ss.run(self._darts_initializer, new_fd)

    def _save_history(self, weights):
        self._history.append(weights)
