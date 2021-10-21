from __future__ import absolute_import, print_function, division

from collections import defaultdict, OrderedDict

import tensorflow as tf

import boml.extension
from boml.lower_iter import BOMLInnerGradTrad, BOMLInnerGradAggr, BOMLInnerGradSimple

RAISE_ERROR_ON_DETACHED = False


class BOMLOuterGrad(object):
    def __init__(self, name):
        self._optimizer_dicts = set()
        self._inner_objectives = None
        self._hypergrad_dictionary = defaultdict(list)
        self._apply_updates = None

        self._initialization = None
        self._iteration = None
        self._state = None
        self._name = name

    _ERROR_NOT_OPTIMIZER_DICT = """
    Looks like {} is not an `OptimizerDict`. Use optimizers in py_bml.optimizers for obtaining an OptimizerDict.
    """

    _ERROR_HYPER_DETACHED = """
    Hyperparameter {} is detached from this optimization dynamics.
    """

    def compute_gradients(self, outer_objective, boml_inner_grad, meta_param=None):
        """
        Function overridden by specific methods.

        :param boml_inner_grad: inner_grad object resulting from the inner objective optimization.
        :param outer_objective: A loss function for the outer parameters
        :param meta_param: Optional list of outer parameters to consider. If not provided will get all variables in the
                            hyperparameter collection in the current scope.

        :return: list of outer parameters involved in the computation
        """
        assert isinstance(
            boml_inner_grad, (BOMLInnerGradAggr, BOMLInnerGradSimple, BOMLInnerGradTrad)
        ), BOMLOuterGrad._ERROR_NOT_OPTIMIZER_DICT.format(boml_inner_grad)
        self._optimizer_dicts.add(boml_inner_grad)

        if meta_param is None:  # get default outer parameters
            meta_param = boml.extension.meta_parameters(tf.get_variable_scope().name)
        return meta_param

    @property
    def initialization(self):
        if self._initialization is None:
            self._initialization = [
                opt_dict.initialization for opt_dict in sorted(self._optimizer_dicts)
            ]
        return self._initialization

    @property
    def iteration(self):
        if self._iteration is None:
            self._iteration = [
                opt_dict.iteration for opt_dict in sorted(self._optimizer_dicts)
            ]
        return self._iteration

    @property
    def state(self):
        for opt_dict in sorted(self._optimizer_dicts):
            for v in opt_dict.state:
                yield v

    @property
    def inner_objectives(self):
        if self._inner_objectives is None:
            self._inner_objectives = [
                opt.objective if hasattr(opt, "objective") else tf.constant(False)
                for opt in sorted(self._optimizer_dicts)
            ]
        return self._inner_objectives

    @property
    def apply_updates(self):
        if self._apply_updates is None:
            self._apply_updates = tf.group(
                *[opt_dict.apply_updates for opt_dict in sorted(self._optimizer_dicts)]
            )
        return self._apply_updates

    def apply_gradients(
        self,
        inner_objective_feed_dicts=None,
        outer_objective_feed_dicts=None,
        initializer_feed_dict=None,
        param_dict=OrderedDict(),
        global_step=None,
        session=None,
    ):
        """
        Runs the inner optimization dynamics for T iterations
        in the meanwhile.

        :param inner_objective_feed_dicts: Optional feed dictionary for the inner objective
        :param outer_objective_feed_dicts: Optional feed dictionary for the outer objective
                                            (note that this is not used in ForwardHG since hypergradients are not
                                            variables)
        :param initializer_feed_dict: Optional feed dictionary for the inner objective
        :param global_step: Optional global step for the optimization process
        :param param_dict: dictionary of parameters passed by `boml.boml_optimizer.BOMLOptimizer`
        :param session: Optional session (otherwise will take the default session)
        """
        raise NotImplementedError()

    def hgrads_hvars(self, meta_param=None, aggregation_fn=None, gradient_clip=None):
        """
        Method for getting outergradient and outer parameters as required by apply_gradient methods from tensorflow
        optimizers.

        :param meta_param: Optional list of outer parameters to consider. If not provided will get all variables in the
                            hyperparameter collection in the current scope.
        :param aggregation_fn: Optional operation to aggregate multiple hypergradients (for the same hyperparameter),
                                by default reduce_mean
        :param gradient_clip: Optional operation like clipping to be applied.
        :return:
        """
        if meta_param is None:
            meta_param = boml.extension.meta_parameters(tf.get_variable_scope().name)

        assert all(
            [h in self._hypergrad_dictionary for h in meta_param]
        ), "FINAL ERROR!"

        if aggregation_fn is None:
            aggregation_fn = lambda hgrad_list: tf.reduce_mean(hgrad_list, axis=0)

        def _aggregate_process_manage_collection(_hg_lst):
            if len(_hg_lst) == 1:  # avoid useless operations...
                aggr = _hg_lst[0]
            else:
                with tf.name_scope(_hg_lst[0].op.name):
                    aggr = aggregation_fn(_hg_lst) if len(_hg_lst) > 1 else _hg_lst[0]
            if gradient_clip is not None:
                with tf.name_scope("process_gradients"):
                    aggr = gradient_clip(aggr)
            tf.add_to_collection(boml.extension.GraphKeys.OUTERGRADIENTS, aggr)
            return aggr

        return [
            (_aggregate_process_manage_collection(self._hypergrad_dictionary[h]), h)
            for h in meta_param
        ]

    @property
    def name(self):
        return self._name

    def __str__(self):
        return self._name
