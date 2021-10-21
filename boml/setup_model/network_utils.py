"""
Contains some misc utility functions
"""

from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

'''
def max_norm_regularizer(threshold=1.0, axes=1, name="max_norm",
                         collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return clip_weights  # there is no regularization loss term
    return max_norm


max_norm_reg = max_norm_regularizer(threshold=1.0)
'''

def max_norm(weights, threshold=1.0, axes=1, name="max_norm",
                     collection="max_norm"):
    clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
    clip_weights = tf.assign(weights, clipped, name=name)
    tf.add_to_collection(collection, clip_weights)
    return clip_weights  # there is no regularization loss term


def spectral_norm(w, iteration=1):
    name = w.name.split('/')[-1].split(':')[0]
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable(name="%s%s" % ("u", name), shape=[1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

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

    return w_norm


def conv_block(boml_net, cweight, bweight):

    """ Perform, conv, batch norm, nonlinearity, and max pool
    :param boml_net: instance of BOMLNet
    :param cweight: parameter of convolutional filter
    :param bweight: bias for covolutional filter
    """
    # cweight = max_norm_reg( weights = cweight )
    # cweight = tf.clip_by_norm(cweight, clip_norm=1.0, axes=1)
    '''
    for _ in range(boml_net.dim_hidden[3]):
        temp_matrix = cweight[:, :, 0, _]
        boml_net.svd_layer.append(tf.svd(temp_matrix, compute_uv=False))
    '''
    if boml_net.max_pool:
        conv_out = tf.nn.conv2d(boml_net.out, cweight, boml_net.no_stride, "SAME",)
    else:
        conv_out = tf.nn.conv2d(boml_net.out, cweight, boml_net.stride, "SAME")
    if bweight is not None:
        conv_out = tf.add(conv_out, bweight)

    if boml_net.batch_norm is not None:
        batch_out = boml_net.batch_norm(
            inputs=conv_out,
            activation_fn=boml_net.activation,
            variables_collections=boml_net.var_collections
        )
    else:
        batch_out = boml_net.activation(conv_out)
    if boml_net.max_pool:
        final_out = tf.nn.max_pool(batch_out, boml_net.stride, boml_net.stride, "VALID")
        return final_out
    else:
        return batch_out


def conv_block_t(boml_net, conv_weight, conv_bias, zweight):
    """ Perform, conv, batch norm, nonlinearity, and max pool
    :param boml_net: instance of BOMLNet
    :param convweight: parameter of convolutional filter
    :param conv_bias: bias for covolutional filter
    :param zweight: parameters of covolutional filter for t-layer"""
    if boml_net.max_pool:
        conv_out = tf.nn.conv2d(boml_net.out, conv_weight, boml_net.no_stride, "SAME")
    else:
        conv_out = tf.nn.conv2d(boml_net.out, conv_weight, boml_net.stride, "SAME")
    if conv_bias is not None:
        conv_out = tf.add(conv_out, conv_bias)

    conv_output = tf.nn.conv2d(conv_out, zweight, boml_net.no_stride, "SAME")
    if boml_net.batch_norm is not None:
        batch_out = layers.batch_norm(
            conv_output,
            activation_fn=boml_net.activation,
            variables_collections=boml_net.var_collections,
        )
    else:
        batch_out = boml_net.activation(conv_output)
    if boml_net.max_pool:
        final_out = tf.nn.max_pool(batch_out, boml_net.stride, boml_net.stride, "VALID")
        return final_out
    else:
        return batch_out


def conv_block_warp(boml_net, cweight, bweight, zweight, zbias):
    """ Perform, conv, batch norm, nonlinearity, and max pool
    :param boml_net: instance of BOMLNet
    :param convweight: parameter of convolutional filter
    :param conv_bias: bias for covolutional filter
    :param zweight: parameters of covolutional filter for warp-layer
    :param zbias: bias of covolutional filter for warp-layer"""

    # cweight = spectral_norm(cweight)
    if boml_net.max_pool:
        conv_out = tf.nn.conv2d(boml_net.out, cweight, boml_net.no_stride, "SAME")
    else:
        conv_out = tf.nn.conv2d(boml_net.out, cweight, boml_net.stride, "SAME")
    if bweight is not None:
        conv_out = tf.add(conv_out, bweight)

    # zweight = spectral_norm(zweight)
    conv_output = tf.nn.conv2d(conv_out, zweight, boml_net.no_stride, "SAME")
    if zbias is not None:
        conv_output = tf.add(conv_output, zbias)
    if boml_net.batch_norm is not None:
        batch_out = layers.batch_norm(
            conv_output,
            activation_fn=boml_net.activation,
            variables_collections=boml_net.var_collections,
        )
    else:
        batch_out = boml_net.activation(conv_output)
    if boml_net.max_pool:
        final_out = tf.nn.max_pool(batch_out, boml_net.stride, boml_net.stride, "VALID")
        return final_out
    else:
        return batch_out


def conv_block_warp_multilayer(boml_net, cweight, bweight,zweight, zbias, mweight,mbias):
    """ Perform, conv, batch norm, nonlinearity, and max pool
    :param boml_net: instance of BOMLNet
    :param convweight: parameter of convolutional filter
    :param conv_bias: bias for covolutional filter
    :param zweight: parameters of covolutional filter for warp-layer
    :param zbias: bias of covolutional filter for warp-layer"""

    #cweight = spectral_norm(cweight)
    if boml_net.max_pool:
        conv_out = tf.nn.conv2d(boml_net.out, cweight, boml_net.no_stride, "SAME")
    else:
        conv_out = tf.nn.conv2d(boml_net.out, cweight, boml_net.stride, "SAME")
    if bweight is not None:
        conv_out = tf.add(conv_out, bweight)

    #zweight = spectral_norm(zweight)
    conv_output = tf.nn.conv2d(conv_out, zweight, boml_net.no_stride, "SAME")
    if zbias is not None:
        conv_output = tf.add(conv_output, zbias)

    #mweight = spectral_norm(mweight)
    conv_output = tf.nn.conv2d(conv_output, mweight, boml_net.no_stride, "SAME")
    if mbias is not None:
        conv_output = tf.add(conv_output, mbias)

    if zbias is not None:
        conv_output = tf.add(conv_output, zbias)
    if boml_net.batch_norm is not None:
        batch_out = layers.batch_norm(
            conv_output,
            activation_fn=boml_net.activation,
            variables_collections=boml_net.var_collections,
        )
    else:
        batch_out = boml_net.activation(conv_output)
    if boml_net.max_pool:
        final_out = tf.nn.max_pool(batch_out, boml_net.stride, boml_net.stride, "VALID")
        return final_out
    else:
        return batch_out


def get_conv_weight(boml_net, layer, initializer):
        return tf.get_variable(
            "conv" + str(layer),
            [
                boml_net.kernel,
                boml_net.kernel,
                boml_net.dim_hidden[layer - 1] if layer > 0 else boml_net.channels,
                boml_net.dim_hidden[layer],
            ],
            initializer=initializer,
            dtype=boml_net.datatype,
        )


def get_warp_weight(boml_net, layer, initializer):
    return tf.get_variable(
        "conv" + str(layer) + "_z",
        [
            boml_net.kernel,
            boml_net.kernel,
            boml_net.dim_hidden[layer - 1],
            boml_net.dim_hidden[layer],
        ],
        initializer=initializer,
        dtype=boml_net.datatype,
    )

def get_multi_weight(boml_net, layer, initializer):
    return tf.get_variable(
        "conv" + str(layer) + "_m",
        [
            boml_net.kernel,
            boml_net.kernel,
            boml_net.dim_hidden[layer - 1],
            boml_net.dim_hidden[layer],
        ],
        initializer=initializer,
        dtype=boml_net.datatype,
    )

def get_warp_bias(boml_net, layer, initializer):
    return tf.get_variable(
        "bias" + str(layer) + "_z",
        [boml_net.dim_hidden[layer]],
        initializer=initializer,
        dtype=boml_net.datatype,
    )


def get_bias_weight(boml_net, layer, initializer):
    return tf.get_variable(
        "bias" + str(layer),
        [boml_net.dim_hidden[layer]],
        initializer=initializer,
        dtype=boml_net.datatype,
    )


def get_identity(dim, name, conv=True):
    return (
        tf.Variable(tf.eye(dim, batch_shape=[1, 1]), name=name)
        if conv
        else tf.Variable(tf.eye(dim), name=name)
    )


def as_list(obj):
    """
    Makes sure `obj` is a list or otherwise converts it to a list with a single element.

    :param obj:
    :return: A `list`
    """
    return obj if isinstance(obj, list) else [obj]


def maybe_call(obj, *args, **kwargs):
    """
    Calls obj with args and kwargs and return its result if obj is callable, otherwise returns obj.
    """
    if callable(obj):
        return obj(*args, **kwargs)
    return obj


def as_tuple_or_list(obj):
    """
    Make sure that `obj` is a tuple or a list and eventually converts it into a list with a single element

    :param obj:
    :return: A `tuple` or a `list`
    """
    return obj if isinstance(obj, (list, tuple)) else [obj]


def maybe_get(obj, i):
    return obj[i] if hasattr(obj, "__getitem__") else obj


def merge_dicts(*dicts):
    return reduce(lambda a, nd: {**a, **nd}, dicts, {})


def to_one_hot_enc(seq, dimension=None):
    da_max = dimension or int(np.max(seq)) + 1
    _tmp = np.zeros((len(seq), da_max))
    _tmp[range(len(_tmp)), np.array(seq, dtype=int)] = 1
    return _tmp
    #
    # def create_and_set(_p):
    #     _tmp = np.zeros(da_max)
    #     _tmp[int(_p)] = 1
    #     return _tmp
    #
    # return np.array([create_and_set(_v) for _v in seq])


def flatten_list(lst):
    from itertools import chain

    return list(chain(*lst))


def filter_vars(var_name, scope):
    import tensorflow as tf

    return [
        v
        for v in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=scope.name if hasattr(scope, "name") else scope,
        )
        if v.name.endswith("%s:0" % var_name)
    ]


def name_from_vars(var_dict, *vars_):
    """
    Unfortunately this method doesn't return a very specific name....It gets a little messy

    :param var_dict:
    :param vars_:
    :return:
    """
    new_k_v = {}
    for v in vars_:
        for k, vv in var_dict.items():
            if v == vv:
                new_k_v[k] = v
    return name_from_dict(new_k_v)


def name_from_dict(_dict, *exclude_names):
    string_dict = {str(k): str(v) for k, v in _dict.items() if k not in exclude_names}
    return _tf_string_replace("_".join(flatten_list(list(sorted(string_dict.items())))))


def _tf_string_replace(_str):
    """
    Replace chars that are not accepted by tensorflow namings (eg. variable_scope)

    :param _str:
    :return:
    """
    return (
        _str.replace("[", "p")
        .replace("]", "q")
        .replace(",", "c")
        .replace("(", "p")
        .replace(")", "q")
        .replace(" ", "")
    )


def get_rand_state(rand):
    """
    Utility methods for getting a `RandomState` object.

    :param rand: rand can be None (new State will be generated),
                    np.random.RandomState (it will be returned) or an integer (will be treated as seed).

    :return: a `RandomState` object
    """
    if isinstance(rand, np.random.RandomState):
        return rand
    elif isinstance(rand, (int, np.ndarray, list)) or rand is None:
        return np.random.RandomState(rand)
    else:
        raise ValueError("parameter rand {} has wrong type".format(rand))


# SOME SCORING UTILS FUNCTIONS

half_int = lambda _m: 1.96 * np.std(_m) / np.sqrt(len(_m) - 1)


def mean_std_ci(measures, mul=1.0, tex=False):
    """
    Computes mean, standard deviation and 95% half-confidence interval for a list of measures.

    :param measures: list
    :param mul: optional multiplication coefficient (e.g. for percentage)
    :param tex: if True returns mean +- half_conf_interval for latex
    :return: a list or a string in latex
    """
    measures = np.array(measures) * mul
    ms = np.mean(measures), np.std(measures), half_int(measures)
    return ms if not tex else r"${:.2f} \pm {:.2f}$".format(ms[0], ms[2])


def leaky_relu(x, alpha, name=None):
    """
    Implements leaky relu with negative coefficient `alpha`
    """
    import tensorflow as tf

    with tf.name_scope(name, "leaky_relu_{}".format(alpha)):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def get_global_step(name="GlobalStep", init=0):
    import tensorflow as tf

    return tf.get_variable(
        name,
        initializer=init,
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES],
    )
