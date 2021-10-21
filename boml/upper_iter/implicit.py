from __future__ import absolute_import, print_function, division

from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
from tensorflow.python.training import slot_creator

from boml import utils
from boml.upper_iter import BOMLOuterGrad

RAISE_ERROR_ON_DETACHED = False


class BOMLOuterGradImplicit(BOMLOuterGrad):
    """
    Implementation follows Pedregosa's algorithm HOAG
    """

    def __init__(
        self,
        inner_method="Trad",
        linear_system_solver_gen=None,
        name="BMLOuterGradImplicit",
    ):
        super(BOMLOuterGradImplicit, self).__init__(name)
        self._inner_method = inner_method
        if linear_system_solver_gen is None:
            linear_system_solver_gen = lambda _obj, var_list, _tolerance: ScipyOptimizerInterface(
                _obj,
                var_list=var_list,
                options={"maxiter": 100},
                method="cg",
                tol=_tolerance,
            )
        self.linear_system_solver = linear_system_solver_gen

        self.tolerance = lambda _k: 0.1 * (0.9 ** _k)

        self._lin_sys = []
        self._qs = []

    def compute_gradients(
        self, outer_objective, inner_grad, meta_param=None, param_dict=OrderedDict()
    ):
        meta_param = super(BOMLOuterGradImplicit, self).compute_gradients(
            outer_objective, inner_grad, meta_param
        )
        state = list(inner_grad.state)

        with tf.variable_scope(outer_objective.op.name):
            g1 = utils.vectorize_all(
                tf.gradients(outer_objective, state)
            )  # Lower Level gradient of UL objective  w.r.t task parameters
            grads_inner_obj_vec = utils.vectorize_all(
                tf.gradients(inner_grad.objective, state)
            )  #  Lower Level gradient of LL objective  w.r.t task parameters

            q = self._create_q(g1)
            obj = tf.norm(
                utils.vectorize_all(
                    tf.gradients(utils.dot(grads_inner_obj_vec, q), state)
                )
                - g1
            )  # using the norm seems to produce better results then squared norm...
            # (even though is more costly)

            self._lin_sys.append(
                lambda _tolerance: self.linear_system_solver(obj, [q], _tolerance)
            )

            g2s = tf.gradients(outer_objective, meta_param)
            cross_ders = tf.gradients(utils.dot(grads_inner_obj_vec, q), meta_param)
            for g2, cd, hyper in zip(g2s, cross_ders, meta_param):
                assert (
                    g2 is not None or cd is not None
                ), BOMLOuterGrad._ERROR_HYPER_DETACHED.format(hyper)
                hg = utils.maybe_add(-cd, g2)
                if hg is None:  # this would be strange...
                    print(
                        "WARNING, outer objective is only directly dependent on hyperparameter {}. "
                        + "Direct optimization would be better!".format(hyper)
                    )
                    hg = g2
                self._hypergrad_dictionary[hyper].append(hg)

            return meta_param

    def _create_q(self, d_oo_d_state):
        self._qs.append(slot_creator.create_zeros_slot(d_oo_d_state, "q"))
        return self._qs[-1]

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

        inner_objective_feed_dicts = utils.as_tuple_or_list(inner_objective_feed_dicts)

        self._run_batch_initialization(
            ss,
            utils.maybe_call(initializer_feed_dict, utils.maybe_eval(global_step, ss)),
        )

        for t in utils.solve_int_or_generator(param_dict["T"]):
            _fd = utils.maybe_call(inner_objective_feed_dicts[0], t)
            self._forward_step(ss, _fd)

        # end of optimization. Solve linear systems.
        tol_val = utils.maybe_call(
            self.tolerance, utils.maybe_eval(global_step, ss)
        )  # decreasing tolerance (seq.)
        # feed dictionaries (could...in theory, implement stochastic solution of this linear system...)
        _fd = utils.maybe_call(inner_objective_feed_dicts[-1], -1)
        _fd_outer = utils.maybe_call(
            outer_objective_feed_dicts, utils.maybe_eval(global_step, ss)
        )
        _fd = utils.merge_dicts(_fd, _fd_outer)

        for lin_sys in self._lin_sys:
            lin_sys(tol_val).minimize(
                ss, _fd
            )  # implicitly warm restarts with previously found q

    def _forward_step(self, ss, _fd):
        ss.run(self.iteration, _fd)

    def _run_batch_initialization(self, ss, fd):
        ss.run(self.initialization, feed_dict=fd)

    def set_tolerance(self, tolerance=lambda _k: 0.1 * (0.9 ** _k)):
        self.tolerance = tolerance
