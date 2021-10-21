from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers

from boml.extension import GraphKeys
from boml.setup_model import network_utils
from boml.setup_model.network import BOMLNet
from boml.setup_model.network_utils import as_tuple_or_list
from boml.utils import remove_from_collection


class BOMLNetMetaInitV1(BOMLNet):
    def __init__(
        self,
        _input,
        dim_output,
        name="BOMLNetMetaInitV1",
        outer_param_dict=OrderedDict(),
        model_param_dict=None,
        task_parameter=None,
        use_t=False,
        use_warp=False,
        outer_method="Simple",
        activation=tf.nn.relu,
        var_collections=tf.GraphKeys.MODEL_VARIABLES,
        conv_initializer=tf.contrib.layers.xavier_initializer_conv2d(tf.float32),
        output_weight_initializer=tf.contrib.layers.xavier_initializer(tf.float32),
        norm=layers.batch_norm,
        data_type=tf.float32,
        channels=1,
        dim_hidden=[64, 64, 64, 64],
        kernel=3,
        max_pool=False,
        reuse=False,
    ):
        self.kernel = kernel
        self.channels = channels
        self.dims = as_tuple_or_list(dim_output)
        self.dim_hidden = dim_hidden
        self.datatype = data_type
        self.batch_norm = norm
        self.max_pool = max_pool
        self.stride = [1, 2, 2, 1]
        self.no_stride = [1, 1, 1, 1]
        self.activation = activation
        self.bias_initializer = tf.zeros_initializer(tf.float32)
        self.conv_initializer = conv_initializer
        self.output_weight_initializer = output_weight_initializer
        self.outer_method = outer_method
        self.use_t = use_t
        self.use_warp = use_warp
        self.svd_layer=[]

        super().__init__(
            _input=_input,
            outer_param_dict=outer_param_dict,
            var_collections=var_collections,
            name=name,
            model_param_dict=model_param_dict,
            task_parameter=task_parameter,
            reuse=reuse,
        )

        # variables from batch normalization
        self.betas = self.filter_vars("beta")
        # moving mean and variance (these variables should be used at inference time... so must save them)
        self.moving_means = self.filter_vars("moving_mean")
        self.moving_variances = self.filter_vars("moving_variance")

        if not reuse:  # these calls might print a warning... it's not a problem..
            remove_from_collection(GraphKeys.MODEL_VARIABLES, *self.betas)
            remove_from_collection(GraphKeys.MODEL_VARIABLES, *self.moving_means)
            remove_from_collection(GraphKeys.MODEL_VARIABLES, *self.moving_variances)
            print(name, "MODEL CREATED")

    def create_meta_parameters(self, var_collections=GraphKeys.METAPARAMETERS):
        for i in range(len(self.dim_hidden)):
            self.outer_param_dict["conv" + str(i)] = network_utils.get_conv_weight(
                self, layer=i, initializer=self.conv_initializer
            )
            self.outer_param_dict["bias" + str(i)] = network_utils.get_bias_weight(
                self, layer=i, initializer=self.bias_initializer
            )
        if self.max_pool:
            self.outer_param_dict["w" + str(len(self.dim_hidden))] = tf.get_variable(
                "w" + str(len(self.dim_hidden)),
                [self.dim_hidden[-1] * 5 * 5, self.dims[-1]],
                initializer=self.output_weight_initializer,
            )
            self.outer_param_dict["bias" + str(len(self.dim_hidden))] = tf.get_variable(
                "bias" + str(len(self.dim_hidden)),
                [self.dims[-1]],
                initializer=self.bias_initializer,
                dtype=self.datatype,
            )
        else:
            self.outer_param_dict["w" + str(len(self.dim_hidden))] = tf.get_variable(
                "w" + str(len(self.dim_hidden)),
                [self.dim_hidden[-1], self.dims[-1]],
                initializer=tf.random_normal_initializer,
            )
            self.outer_param_dict["bias" + str(len(self.dim_hidden))] = tf.get_variable(
                "bias" + str(len(self.dim_hidden)),
                [self.dims[-1]],
                initializer=self.bias_initializer,
                dtype=self.datatype,
            )
        [
            tf.add_to_collections(var_collections, hyper)
            for hyper in self.outer_param_dict.values()
        ]

        if len(self.model_param_dict) == 0 and callable(
            getattr(self, "create_model_parameters", None)
        ):
            self.create_model_parameters()

        return self.outer_param_dict

    def create_model_parameters(self, var_collections=GraphKeys.MODELPARAMETERS):
        if self.use_t:
            # hyper parameters of transformation layer
            for i in range(len(self.dim_hidden)):
                self.model_param_dict[
                    "conv" + str(i) + "_z"
                ] = network_utils.get_identity(
                    self.dim_hidden[0], name="conv" + str(i) + "_z", conv=True
                )
            self.model_param_dict[
                "w" + str(len(self.dim_hidden)) + "_z"
            ] = network_utils.get_identity(
                self.dims[-1], name="w" + str(len(self.dim_hidden)) + "_z", conv=False
            )
        elif self.use_warp:
            for i in range(len(self.dim_hidden)):
                self.model_param_dict[
                    "conv" + str(i) + "_z"
                ] = network_utils.get_warp_weight(
                    self, layer=i, initializer=self.conv_initializer
                )
                self.model_param_dict[
                    "bias" + str(i) + "_z"
                ] = network_utils.get_warp_bias(
                    self, layer=i, initializer=self.bias_initializer
                )
        [
            tf.add_to_collections(var_collections, model_param)
            for model_param in self.model_param_dict.values()
        ]
        return self.model_param_dict

    def _forward(self):
        if self.task_parameter is None:
            self.task_parameter = self.create_initial_parameter(
                primary_outerparameter=self.outer_param_dict
            )

        for i in range(len(self.dim_hidden)):
            if self.use_t:
                self + network_utils.conv_block_t(
                    self,
                    self.task_parameter["conv" + str(i)],
                    self.task_parameter["bias" + str(i)],
                    self.model_param_dict["conv" + str(i) + "_z"],
                )
            elif self.use_warp:
                self + network_utils.conv_block_warp(
                    self,
                    self.task_parameter["conv" + str(i)],
                    self.task_parameter["bias" + str(i)],
                    self.model_param_dict["conv" + str(i) + "_z"],
                    self.model_param_dict["bias" + str(i) + "_z"],
                )
            else:
                self + network_utils.conv_block(
                    self,
                    self.task_parameter["conv" + str(i)],
                    self.task_parameter["bias" + str(i)],
                )

        if self.max_pool:
            self + tf.reshape(
                self.out, [-1, np.prod([int(dim) for dim in self.out.get_shape()[1:]])]
            )
            self + tf.add(
                tf.matmul(
                    self.out, self.task_parameter["w" + str(len(self.dim_hidden))]
                ),
                self.task_parameter["bias" + str(len(self.dim_hidden))],
            )
        else:
            self + tf.add(
                tf.matmul(
                    tf.reduce_mean(self.out, [1, 2]),
                    self.task_parameter["w" + str(len(self.dim_hidden))],
                ),
                self.task_parameter["bias" + str(len(self.dim_hidden))],
            )

        if self.use_t:
            self + tf.matmul(
                self.out, self.model_param_dict["w" + str(len(self.dim_hidden)) + "_z"]
            )

    def re_forward(self, new_input=None, task_parameter=OrderedDict()):
        return BOMLNetMetaInitV1(
            _input=new_input if new_input is not None else self.layers[0],
            dim_output=self.dims[-1],
            name=self.name,
            activation=self.activation,
            outer_param_dict=self.outer_param_dict,
            model_param_dict=self.model_param_dict,
            task_parameter=self.task_parameter
            if len(task_parameter.keys()) == 0
            else task_parameter,
            use_t=self.use_t,
            use_warp=self.use_warp,
            outer_method=self.outer_method,
            var_collections=self.var_collections,
            dim_hidden=self.dim_hidden,
            output_weight_initializer=self.output_weight_initializer,
            max_pool=self.max_pool,
            reuse=True,
        )


def BOMLNetOmniglotMetaInitV1(
    _input,
    dim_output,
    outer_param_dict=OrderedDict(),
    model_param_dict=OrderedDict(),
    batch_norm=layers.batch_norm,
    name="BOMLNetOmniglotMetaInitV1",
    outer_method="Simple",
    use_t=False,
    use_warp=False,
    **model_args
):
    return BOMLNetMetaInitV1(
        _input=_input,
        name=name,
        dim_output=dim_output,
        model_param_dict=model_param_dict,
        outer_method=outer_method,
        outer_param_dict=outer_param_dict,
        norm=batch_norm,
        use_t=use_t,
        use_warp=use_warp,
        **model_args
    )


def BOMLNetMiniMetaInitV1(
    _input,
    dim_output,
    outer_param_dict=OrderedDict(),
    model_param_dict=OrderedDict(),
    batch_norm=layers.batch_norm,
    name="BOMLNetMetaInitV1",
    outer_method="Simple",
    use_t=False,
    use_warp=False,
    **model_args
):
    return BOMLNetMetaInitV1(
        _input=_input,
        name=name,
        dim_output=dim_output,
        use_t=use_t,
        use_warp=use_warp,
        outer_param_dict=outer_param_dict,
        model_param_dict=model_param_dict,
        outer_method=outer_method,
        norm=batch_norm,
        channels=3,
        dim_hidden=[32, 32, 32, 32],
        max_pool=True,
        **model_args
    )
