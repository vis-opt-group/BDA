import sys
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.training import slot_creator

from boml.extension import GraphKeys
from boml.setup_model.network_utils import filter_vars
from boml.utils import remove_from_collection


class BOMLNet(object):
    # definitely deprecated!
    """
    Base object for models
    """

    def __init__(
        self,
        _input,
        outer_param_dict=OrderedDict(),
        model_param_dict=OrderedDict(),
        task_parameter=None,
        var_collections=None,
        name=None,
        reuse=False,
    ):
        """
        Creates an object that creates model parameters and defines the network structure.
        :param _input: the input shape for defined network
        :param outer_param_dict: type of OrderedDict to restore parameters to optimize in the upper level
        :param model_param_dict: type of OrderedDict to restore parameters of special model like Iteration Warping
        :param task_parameter: type of OrderedDict to restore parameters to optimize in the lower level
        """
        super(BOMLNet, self).__init__()

        if not name:
            try:
                name = tf.get_variable_scope().name
            except IndexError:
                print("Warning: no name and no variable scope given", sys.stderr)
        self.outer_param_dict = outer_param_dict
        self.model_param_dict = model_param_dict
        self.task_parameter = task_parameter
        self.var_collections = var_collections
        self.name = name
        self.reuse = reuse
        self._var_list_initial_values = []
        self._var_init_placeholder = None
        self._assign_int = []
        self._var_initializer_op = None

        self.layers = [_input]
        # self.s = None
        self._tf_saver = None

        with self._variable_scope(reuse):
            if (len(self.outer_param_dict) == 0) and (
                callable(getattr(self, "create_meta_parameters", None))
            ):
                self.create_meta_parameters()
            self._forward()

    def _variable_scope(self, reuse):
        """
        May override default variable scope context, but pay attention since it looks like
        initializer and default initializer form contrib.layers do not work well together (I think this is because
        functions in contrib.layers usually specify an initializer....

        :param reuse:
        :return:
        """
        return tf.variable_scope(self.name, reuse=reuse)

    def __getitem__(self, item):
        """
        Get's the `activation`

        :param item:
        :return:
        """
        return self.layers[item]

    def __add__(self, other):
        if self.name not in tf.get_variable_scope().name:
            print(
                "Warning: adding layers outside model variable scope", file=sys.stderr
            )
        self.layers.append(other)
        return self

    @property
    def var_list(self):
        if len(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, self.name)) == 0:
            assert self.task_parameter is not None, (
                "No Model Variables to optimize, "
                "please double check your computational graph"
            )
            return list(self.task_parameter.values())
        else:
            return tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, self.name)

    @property
    def out(self):
        return self[-1]

    def create_initial_parameter(self, primary_outerparameter=None):
        assert primary_outerparameter is not None, (
            "Primary hyperparameters "
            "must be provided for initialization of slot variables"
        )
        # task_weights_keys = [key for key in primary_hyperparameter.keys() if 'prob' not in key and 'z' not in key]
        initial_parameter = OrderedDict(
            [
                (
                    primary_key,
                    slot_creator.create_slot(
                        primary=self.out,
                        val=primary_outerparameter[primary_key].initialized_value(),
                        name=primary_key,
                    ),
                )
                for primary_key in primary_outerparameter.keys()
            ]
        )
        [
            tf.add_to_collection(self.var_collections, initial_param)
            for initial_param in initial_parameter.values()
        ]
        remove_from_collection(GraphKeys.GLOBAL_VARIABLES, *initial_parameter.values())
        return initial_parameter

    def filter_vars(self, var_name):
        return filter_vars(var_name, self.name)

    def initialize(self, session=None):
        """
        Initialize the model. If `deterministic_initialization` is set to true,
        saves the initial weight in numpy which will be used for subsequent initialization.
        This is because random seed management in tensorflow is rather obscure... and I could not
        find a way to set the same seed across different initialization without exiting the session.

        :param session:
        :return:
        """
        # TODO placeholder are not useful at all here... just change initializer of tf.Variables !
        ss = session or tf.get_default_session()
        assert ss, "No default session"
        if not self._var_initializer_op:
            self._var_initializer_op = tf.variables_initializer(self.var_list)
        ss.run(self._var_initializer_op)

    def _variables_to_save(self):
        return self.var_list

    def _forward(self):
        """
        _forward() uses defined convolutional neural networks with initial input
        :return:
        """
        raise NotImplemented()

    def re_forward(self, new_input, task_parameter=OrderedDict()):
        """
        reuses defined convolutional with new input and update the output results
        :param new_input: new input with same shape as the old one
        :param task_parameter: dictionary to store the task specific parameters
        :return:
        """
        raise NotImplemented()

    @property
    def tf_saver(self):
        if not self._tf_saver:
            self._tf_saver = tf.train.Saver(
                var_list=self._variables_to_save(), max_to_keep=1
            )
        return self._tf_saver

    def save(self, file_path, session=None, global_step=None):
        # TODO change this method! save the class (or at least the attributes you can save
        self.tf_saver.save(
            session or tf.get_default_session(), file_path, global_step=global_step
        )

    def restore(self, file_path, session=None, global_step=None):
        if global_step:
            file_path += "-" + str(global_step)
        self.tf_saver.restore(session or tf.get_default_session(), file_path)
