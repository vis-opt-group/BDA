from __future__ import absolute_import, print_function, division

from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.training import slot_creator
import sys
import boml.extension
from boml import utils
from boml.upper_iter.outer_grad import BOMLOuterGrad
from boml.utils import maybe_add,reduce_all_sums,dot
RAISE_ERROR_ON_DETACHED = False


class BOMLOuterGradForward(BOMLOuterGrad):
    def __init__(
        self, inner_method="Trad", name="BOMLOuterGradForward"
    ):
        """
       Utility method to initialize truncated reverse HG (not necessarily online),
       :param name: a name for the operations and variables that will be created
       :return: Forward object
           """
        super(BOMLOuterGradForward, self).__init__(name)
        self._forward_initializer = tf.no_op()
        self._zs = {}  # hyperparameter - zs dictionary
        self._z_iter = tf.no_op()
        self._iteration = None
        self.A_dot_zs = {}


    _HYPER_RANK_ERROR_MESSAGE = """
    ForwardHG: Only scalar hyperparameters accepted.\n
     Hyperparameter tensor {} has rank {}.\n
     Use keyword argument far_ho.get_hyperparameter(..., scalar=True) on hyperparameter creation.
    """

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

        :return: list of outer parameters involved in the computation
        """
        meta_param = super(BOMLOuterGradForward, self).compute_gradients(outer_objective, inner_grad, meta_param)

        # scalar_hyper_list

        with tf.variable_scope(outer_objective.op.name):
            # dynamics_vec = vectorize_all(optimizer_dict.dynamics)  # in the new implementation there's no need of
            # vectorizing... it might be more efficient since it's better to avoid too many reshaping operations...
            d_oo_d_state = tf.gradients(outer_objective, list(inner_grad.state))

            with tf.name_scope('DUMMY'):  # variables to compute forward propagation
                # TODO avoid this computation if optimizer_dict has already been seen.
                aux_vs = [tf.zeros_like(v) for v in inner_grad.state]
                dynamics_dot_aux_v = reduce_all_sums(list(inner_grad.dynamics), aux_vs)

                der_dynamics_dot_aux_v = tf.gradients(dynamics_dot_aux_v, list(inner_grad.state))
                # this is a list of jacobians times aux_vs that have the same dimension of states variables.

                init_dynamics_dot_aux_v = None
                if inner_grad.init_dynamics:
                    # init_dynamics_dot_aux_v = dot(vectorize_all(optimizer_dict.init_dynamics), aux_v_vec)  # old impl
                    init_dynamics_dot_aux_v = reduce_all_sums(
                        inner_grad.init_dynamics, aux_vs)

            for meta_par in meta_param:
                assert meta_par.shape.ndims >= 0, BOMLOuterGradForward._HYPER_RANK_ERROR_MESSAGE.format(meta_par, meta_par.shape.ndims)

                d_init_dyn_d_hyp = None if init_dynamics_dot_aux_v is None else \
                    tf.gradients(init_dynamics_dot_aux_v, meta_par)[0]
                d_dyn_d_hyp = tf.gradients(dynamics_dot_aux_v, meta_par)[0]
                d_oo_d_hyp = tf.gradients(outer_objective, meta_par)[0]

                # ------------------------------------------------------------
                # check detached hyperparameters (for which hypergradient would be always null)
                hyper_ok = d_init_dyn_d_hyp is not None or d_dyn_d_hyp is not None or d_oo_d_hyp is not None
                if RAISE_ERROR_ON_DETACHED:
                    # try:
                    assert hyper_ok, BOMLOuterGrad._ERROR_HYPER_DETACHED.format(meta_par)
                    # ex
                else:
                    if not hyper_ok:
                        print(BOMLOuterGrad._ERROR_HYPER_DETACHED.format(meta_par), file=sys.stderr)
                        meta_param.remove(meta_par)
                # -------------------------------------------------------------

                # UPDATE OF TOTAL DERIVATIVE OF STATE W.R.T. HYPERPARAMETER
                zs = BOMLOuterGradForward._create_zs(
                    inner_grad, meta_par, None if d_init_dyn_d_hyp is None else tf.gradients(d_init_dyn_d_hyp, aux_vs)
                )  # this is one z for each variable
                self._zs[meta_par] = zs  # store a reference for the total derivatives for easy access
                Bs = tf.gradients(d_dyn_d_hyp, aux_vs)

                A_dot_zs = tf.gradients(reduce_all_sums(der_dynamics_dot_aux_v, zs), aux_vs)

                self.A_dot_zs[meta_par] = A_dot_zs

                _z_iter = tf.group(*[
                    z.assign(maybe_add(A_dot_z, B)) for z, A_dot_z, B
                    in zip(zs, A_dot_zs, Bs)
                ])
                self._z_iter = tf.group(self._z_iter, _z_iter)

                # -- HYPERGRADIENT -----
                d_E_T = [dot(d_oo_d_s, z) for d_oo_d_s, z in zip(d_oo_d_state, zs)
                         if d_oo_d_s is not None and z is not None]  # list of dot products
                hg = maybe_add(tf.reduce_sum(d_E_T), d_oo_d_hyp)  # sum the partial dot products and possibly ->
                # adds the ''direct derivative'' term d(E( . , \lambda))/d \lambda

                self._hypergrad_dictionary[meta_par].append(hg)
                self._forward_initializer = tf.group(self._forward_initializer,
                                                     tf.variables_initializer(zs))
        return meta_param

    @staticmethod
    def _create_zs(optimizer_dict, hyper, d_init_dynamics_d_hyper):
        if d_init_dynamics_d_hyper is None: d_init_dynamics_d_hyper = [None] * len(optimizer_dict)
        with tf.variable_scope('Z'):
            z = [slot_creator.create_slot(v, utils.val_or_zero(der, v), hyper.op.name) for v, der
                 in zip(optimizer_dict.state, d_init_dynamics_d_hyper)]
            [tf.add_to_collection(boml.extension.GraphKeys.ZS, lm) for lm in z]
            # in this case it is completely fine to keep zs into the global variable...
            return z

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
        return BOMLOuterGradForward._create_outergradient_from_dodh(
            hyper, tf.gradients(outer_obj, hyper)[0]
        )

    def _state_feed_dict_generator(self, history, T_or_generator):
        for t, his in zip(utils.solve_int_or_generator(T_or_generator), history):
            yield t, utils.merge_dicts(
                *[
                    od.state_feed_dict(h)
                    for od, h in zip(sorted(self._optimizer_dicts), his)
                ]
            )

    def apply_gradients(
        self,
        inner_objective_feed_dicts=None,
        outer_objective_feed_dicts=None,
        initializer_feed_dict=None,
        param_dict=OrderedDict(),
        train_batches=None,
        experiments=[],
        global_step=None,
        session=None,
    ):

        ss = session or tf.get_default_session()

        self._run_batch_initialization(ss, utils.maybe_call(
            initializer_feed_dict, utils.maybe_eval(global_step, ss)))

        for t in utils.solve_int_or_generator(param_dict['T']):
            _fd = utils.maybe_call(inner_objective_feed_dicts, t)
            self._forward_step(ss, _fd)

    def _forward_step(self, ss, _fd):
        ss.run(self._z_iter, _fd)
        ss.run(self.iteration, _fd)

    def _run_batch_initialization(self, ss, fd):
        ss.run(self.initialization, feed_dict=fd)
        ss.run(self._forward_initializer, feed_dict=fd)

    @property
    def w_dots(self):
        # if hyper: return self._zs[hyper]
        return [{h: self._zs[h][k] for h in self._zs} for k, _ in enumerate(self.state)]

    def z_callback(self, hyperparameter=None, flatten=True):
        zs_values = []
        zs = list(self._zs.values()) if hyperparameter is None else self._zs[hyperparameter]
        if flatten: zs = utils.vectorize_all(zs)

        # noinspection PyUnusedLocal
        def _callback(_, __, ss):
            zs_values.append(ss.run(zs))  # these should not depend from any feed dictionary

        return zs_values, _callback