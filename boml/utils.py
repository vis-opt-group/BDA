from __future__ import absolute_import, print_function, division

import sys

import numpy as np
import tensorflow as tf

# noinspection PyClassHasNoInit
import boml

Meta_Init_Method = ["MAML", "FOMAML", "WarpGrad", "MT-net"]
Meta_Init_Method = ["Reverse", "Truncated", "DARTS", "BA", "Implicit"]
METHOD_COLLECTIONS = [Meta_Init_Method, Meta_Init_Method]


def remove_from_collection(key, *lst):
    """
    Remove tensors in lst from collection given by key
    """
    try:
        # noinspection PyProtectedMember
        [tf.get_default_graph()._collections[key].remove(_e) for _e in lst]
    except ValueError:
        print(
            "WARNING: Collection -> {} <- does not contain some tensor in {}".format(
                key, lst
            ),
            file=sys.stderr,
        )


def as_tuple_or_list(obj):
    """
    Make sure that `obj` is a tuple or a list and eventually converts it into a list with a single element

    :param obj:
    :return: A `tuple` or a `list`
    """
    return obj if isinstance(obj, (list, tuple)) else [obj]


def merge_dicts(*dicts):
    """
    Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
    """
    from functools import reduce

    # if len(dicts) == 1: return dicts[0]
    return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})


def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def as_list(obj):
    """
    Makes sure `obj` is a list or otherwise converts it to a list with a single element.
    """
    return obj if isinstance(obj, list) else [obj]


def vectorize_all(var_list, name=None):
    """Given a list of tensors returns their concatenated vectorization.
    Note that for matrices the vectorization is row-wise instead of column-wise as
    it should be in Magnus. Could it be a problem?

    :param var_list: **bold**
    :param name: optional name for resulting tensor

    :return: vectorization of `var_list`"""
    with tf.name_scope(name, "Vectorization", var_list) as scope:
        return tf.concat([tf.reshape(_w, [-1]) for _w in var_list], 0, name=scope)


def add_list(lst1, lst2):
    """
    sum elements in two lists by sequence
    :return: new list that sum elements in two lists by sequence
    """
    assert len(lst1) == len(lst2), "length of two list must be equal"
    return [ls1 + ls2 for ls1, ls2 in zip(lst1, lst2)]


def maybe_call(obj, *args, **kwargs):
    """
    Calls obj with args and kwargs and return its result if obj is callable, otherwise returns obj.
    """
    if callable(obj):
        return obj(*args, **kwargs)
    return obj


def dot(a, b, name=None):
    """
    Dot product between vectors `a` and `b` with optional name.
    If a and b are not vectors, formally this computes <vec(a), vec(b)>.
    """
    with tf.name_scope(name, "Dot", [a, b]):
        return tf.reduce_sum(a * b)


def maybe_eval(a, ss=None):
    """
    Run or eval `a` and returns the result if possible.

    :param a: object, or `tf.Variable` or `tf.Tensor`
    :param ss: `tf.Session` or get default session (if any)
    :return: If a is not a tensorflow evaluable returns it, or returns the
                resulting call
    """
    if ss is None:
        ss = tf.get_default_session()
    if hasattr(a, "eval") or hasattr(a, "run"):
        return ss.run(a)
    return a


def solve_int_or_generator(int_or_generator):
    return range(int_or_generator) if isinteger(int_or_generator) else int_or_generator


def cross_entropy(pred, label, method="MetaInit"):

    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    assert method in ("MetaInit", "MetaRepr"), "Wrong value for argument method"
    if method == "MetaInit":
        return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)
    elif method == "MetaRepr":
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)
        )


def classification_acc(pred, label):
    return tf.contrib.metrics.accuracy(
        tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(label, 1)
    )


def reduce_all_sums(lst1, lst2, name=None):
    with tf.name_scope(name, 'Vectorization', lst1 + lst2) as scope:
        return tf.add_n([tf.reduce_sum(v1*v2) for v1, v2 in zip(lst1, lst2)], name=scope)


def set_gpu():
    # set general configuration
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    return gpu_config


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


def maybe_add(a, b):
    """
    return a if b is None else a + b
    """
    return a if b is None else a + b


def val_or_zero(a, b):
    """
    return a if a is not None else tf.zeros_like(b)
    """
    return a if a is not None else tf.zeros_like(b)


def isinteger(num):
    return isinstance(num, (int, np.int_, np.int8, np.int16, np.int32, np.int64))


def feed_dicts(dat_lst, exs):
    dat_lst = boml.utils.as_list(dat_lst)
    train_fd = boml.utils.merge_dicts(
        *[
            {_ex.x: dat.train.data, _ex.y: dat.train.target}
            for _ex, dat in zip(exs, dat_lst)
        ]
    )
    valid_fd = boml.utils.merge_dicts(
        *[
            {_ex.x_: dat.test.data, _ex.y_: dat.test.target}
            for _ex, dat in zip(exs, dat_lst)
        ]
    )

    return train_fd, valid_fd
