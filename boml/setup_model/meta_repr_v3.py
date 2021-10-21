from collections import OrderedDict
from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers
from boml import extension
from boml.setup_model.network import BOMLNet
from boml.extension import get_outerparameter

def vectorize_all(var_list, name=None):
    """Given a list of tensors returns their concatenated vectorization.
    Note that for matrices the vectorization is row-wise instead of column-wise as
    it should be in Magnus. Could it be a problem?

    :param var_list: **bold**
    :param name: optional name for resulting tensor

    :return: vectorization of `var_list`"""
    with tf.name_scope(name, "Vectorization", var_list) as scope:
        return tf.concat([tf.reshape(_w, [-1]) for _w in var_list], 0, name=scope)


def vectorize(var, name=None):
    """Given a list of tensors returns their concatenated vectorization.
    Note that for matrices the vectorization is row-wise instead of column-wise as
    it should be in Magnus. Could it be a problem?

    :param var_list: **bold**
    :param name: optional name for resulting tensor

    :return: vectorization of `var_list`"""
    with tf.name_scope(name, "Vectorization", var):
        return tf.reshape(var, [-1])


def spectral_norm_regularizer(w, iteration=1):
    name = w.name.split('/')[-1].split(':')[0]
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable(name="%s%s" % ("u", name), shape=[1, w_shape[-1]],
                        initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
        tf.add_to_collection("spectral_norm", w_norm)
    return w_norm


def max_norm_regularizer(threshold=1.0, axes=1, name="max_norm",
                         collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)

        return None  # there is no regularization loss term

    return max_norm


max_norm_reg = max_norm_regularizer(threshold=1.0)

def positive_norm_regularizer(clip_value_min=0.0,clip_value_max=1.0, name="positive_norm",
                         collection="positive_norm"):
    def positive_norm(weights):
        clipped = tf.clip_by_value(weights, clip_value_min=clip_value_min,clip_value_max=clip_value_max)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)

        return None  # there is no regularization loss term

    return positive_norm


positive_norm_reg = positive_norm_regularizer(clip_value_min=0.0,clip_value_max=1.0)


class BOMLNetMnistMetaReprV3(BOMLNet):
    def __init__(
            self,
            _input,
            name="BMLNetC4LMetaRepr",
            outer_param_dict=OrderedDict(),
            model_param_dict=OrderedDict(),
            task_parameter=None,
            use_t=False,
            use_warp=False,
            outer_method="Reverse",
            dim_output=-1,
            activation=tf.nn.relu,
            var_collections=extension.METAPARAMETERS_COLLECTIONS,
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
        self.dim_output = dim_output
        self.kernel = kernel
        self.channels = channels
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
        self.use_t = use_t
        self.use_warp = use_warp
        self.outer_method = outer_method
        self.flatten = False if self.outer_method == "Implicit" else True
        self.svd_layer = []
        self.orthogonality = []

        super(BOMLNetMnistMetaReprV3, self).__init__(
            _input=_input,
            outer_param_dict=outer_param_dict,
            var_collections=var_collections,
            name=name,
            model_param_dict=model_param_dict,
            task_parameter=task_parameter,
            reuse=reuse,
        )

        self.betas = self.filter_vars("beta")

        self.moving_means = self.filter_vars("moving_mean")
        self.moving_variances = self.filter_vars("moving_variance")

        if not reuse:
            extension.remove_from_collection(
                extension.GraphKeys.MODEL_VARIABLES, *self.moving_means
            )
            extension.remove_from_collection(
                extension.GraphKeys.MODEL_VARIABLES, *self.moving_variances
            )
            print(name, "MODEL CREATED")
        extension.remove_from_collection(
            extension.GraphKeys.METAPARAMETERS,
            *self.moving_means,
            *self.moving_variances)

    def create_meta_parameters(self, var_collections=extension.GraphKeys.METAPARAMETERS):
        #self.outer_param_dict['lambda'] = get_outerparameter(name='lambda',shape=(5000),dtype=tf.float32,collections=var_collections,
                                                            # initializer=tf.zeros_initializer,scalar=True,regularizer=positive_norm_reg)
        self.outer_param_dict['lambda'] = tf.get_variable(name='lambda', shape=(5000),initializer=tf.zeros_initializer,regularizer=positive_norm_reg)
        tf.add_to_collections(var_collections, self.outer_param_dict['lambda'])

    def _forward(self):
        return

    def re_forward(self, new_input=None):
        return BOMLNetMnistMetaReprV3(
            _input=new_input if new_input is not None else self.layers[0],
            name=self.name,
            activation=self.activation,
            outer_param_dict=self.outer_param_dict,
            model_param_dict=self.model_param_dict,
            dim_output=self.dim_output,
            task_parameter=self.task_parameter,
            use_warp=self.use_warp,
            use_t=self.use_t,
            var_collections=self.var_collections,
            dim_hidden=self.dim_hidden,
            output_weight_initializer=self.output_weight_initializer,
            max_pool=self.max_pool,
            reuse=tf.AUTO_REUSE,
            outer_method=self.outer_method,
        )

