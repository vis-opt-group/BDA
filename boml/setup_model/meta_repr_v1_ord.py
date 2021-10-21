from collections import OrderedDict
from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers
import tensorflow.contrib.layers as tcl
from boml import extension
from boml.setup_model import network_utils
from boml.setup_model.network import BOMLNet


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


def max_norm_regularizer(threshold=0.5, axes=1, name="max_norm",
                         collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)

        return None  # there is no regularization loss term

    return max_norm


max_norm_reg = max_norm_regularizer(threshold=0.3)


class BOMLNetMetaReprV1(BOMLNet):
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
            dim_hidden=[64, 64, 64,64],
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

        super(BOMLNetMetaReprV1, self).__init__(
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

    def conv_layer(self, layer=-1, hyperparameter=False, activation=tf.nn.relu,
                   var_coll=extension.METAPARAMETERS_COLLECTIONS,
                   conv_initialization=tf.contrib.layers.xavier_initializer_conv2d(tf.float32)):

        bn = lambda _inp: tcl.batch_norm(_inp, variables_collections=var_coll)

        self + tcl.conv2d(self.out, num_outputs=self.dim_hidden[layer],
                          stride=self.no_stride[1] if self.max_pool else self.stride[1],
                          kernel_size=3, normalizer_fn=bn, activation_fn=None,
                          trainable=not hyperparameter,
                          variables_collections=var_coll, weights_initializer=conv_initialization)
        self + activation(self.out)
        if self.max_pool:
            self + tf.nn.max_pool(self.out, self.stride, self.stride, 'VALID')

    def _forward(self):
        for _ in range(len(self.dim_hidden)):
            self.conv_layer(_)

        if len(self.outer_param_dict) == 0:
            for var in self.var_list:
                if var.name == self.name+ '/Conv/'+ 'weights:0':
                    self.outer_param_dict['Conv_0'] = var
                elif var.name.endswith('/weights:0'):
                    self.outer_param_dict[var.name.split('/')[1]] = var
            if len(list(tf.get_collection("max_norm"))) > 0:
                self.clip_weights = tf.norm(list(tf.get_collection("max_norm"))[0], axis=1, ord=2)
                for _ in range(self.dim_hidden[-1]):
                    vector = vectorize_all([list(tf.get_collection("max_norm"))[-1][:, :, :, _]], name='vector')
                    # vector = vector / tf.norm(vector, ord=2)
                    self.orthogonality.append(tf.norm(vector * tf.transpose(vector) - tf.eye(vector.get_shape().as_list()[0]), ord=2))
                self.orthogonality = tf.reduce_mean(self.orthogonality)
            elif len(list(tf.get_collection("spectral_norm"))) > 0:
                for _ in range(self.dim_hidden[0]):
                    temp_matrix = list(tf.get_collection("spectral_norm"))[0][:, :, 0, _]
                    self.svd_layer.append(tf.svd(temp_matrix, compute_uv=False))
                for _ in range(self.dim_hidden[-1]):
                    vector = vectorize_all([list(tf.get_collection("spectral_norm"))[-1][:, :, :, _]], name='vector')
                    # vector = vector / tf.norm(vector, ord=2)
                    self.orthogonality.append(tf.norm(vector * tf.transpose(vector) - tf.eye(vector.get_shape().as_list()[0]), ord=2))
                self.orthogonality = tf.reduce_mean(self.orthogonality)
            else:
                for _ in range(self.dim_hidden[0]):
                    temp_matrix = self.outer_param_dict["Conv_0"][:, :, 0, _]
                    self.svd_layer.append(tf.svd(temp_matrix, compute_uv=False))
                for _ in range(self.dim_hidden[-1]):
                    vector = vectorize_all([ self.outer_param_dict["Conv_"+str(len(self.dim_hidden)-1)][:,:,:,_]], name='vector')
                    # vector = vector / tf.norm(vector, ord=2)
                    self.orthogonality.append(tf.norm(vector * tf.transpose(vector) - tf.eye(vector.get_shape().as_list()[0]), ord=2))
                self.orthogonality = tf.reduce_mean(self.orthogonality)
        flattened_shape = reduce(lambda a, v: a * v, self.layers[-1].get_shape().as_list()[1:])
        self + tf.reshape(self.out, shape=(-1, flattened_shape), name='representation')
        '''
        for i in range(len(self.dim_hidden)):
            if self.use_t:
                self + network_utils.conv_block_t(
                    self,
                    conv_weight=self.outer_param_dict["conv" + str(i)],
                    conv_bias=None,
                    zweight=self.model_param_dict["conv" + str(i) + "_z"],
                )
            elif self.use_warp:
                self + network_utils.conv_block_warp(
                    self,
                    self.outer_param_dict["conv" + str(i)],
                    bweight=None,
                    zweight=self.model_param_dict["conv" + str(i) + "_z"],
                    zbias=None
                )
            else:
                self + network_utils.conv_block(
                    self,
                    self.outer_param_dict["conv" + str(i)],
                    bweight=None
                )

        for _ in range(self.dim_hidden[3]):
            temp_matrix = self.outer_param_dict["conv3"][:, :, _, 0]
            self.svd_layer.append(tf.svd(temp_matrix, compute_uv=False))

        if self.flatten:
            flattened_shape = reduce(
                lambda a, v: a * v, self.layers[-1].get_shape().as_list()[1:]
            )
            self + tf.reshape(
                self.out, shape=(-1, flattened_shape), name="representation"
            )
        else:
        
        if self.max_pool:
            self + tf.reshape(
                self.out,
                [-1, np.prod([int(dim) for dim in self.out.get_shape()[1:]])],
            )
        else:
            self + tf.reduce_mean(self.out, [1, 2])
        '''

    def re_forward(self, new_input=None):
        return BOMLNetMetaReprV1(
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


def BOMLNetOmniglotMetaReprV1(
        _input,
        outer_param_dict=OrderedDict(),
        model_param_dict=OrderedDict(),
        batch_norm=layers.batch_norm,
        name="BMLNetC4LOmniglot",
        use_t=False,
        dim_output=-1,
        use_warp=False,
        outer_method="Reverse",
        **model_args
):
    return BOMLNetMetaReprV1(
        _input=_input,
        name=name,
        model_param_dict=model_param_dict,
        dim_output=dim_output,
        outer_param_dict=outer_param_dict,
        norm=batch_norm,
        use_t=use_t,
        use_warp=use_warp,
        outer_method=outer_method,
        **model_args
    )


def BOMLNetMiniMetaReprV1(
        _input,
        outer_param_dict=OrderedDict(),
        model_param_dict=OrderedDict(),
        dim_output=-1,
        batch_norm=layers.batch_norm,
        name="BOMLNetC4LMini",
        use_t=False,
        use_warp=False,
        outer_method="Reverse",
        **model_args
):
    return BOMLNetMetaReprV1(
        _input=_input,
        name=name,
        use_t=use_t,
        use_warp=use_warp,
        dim_output=dim_output,
        outer_param_dict=outer_param_dict,
        model_param_dict=model_param_dict,
        norm=batch_norm,
        channels=3,
        dim_hidden=[32, 32, 32, 32],
        max_pool=True,
        outer_method=outer_method,
        **model_args
    )
