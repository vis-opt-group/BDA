import sys

import numpy as np
import tensorflow as tf

from boml import utils

Hyper_Optim_Method = ["MAML", "FOMAML", "MSGD", "MTNet"]
Bilevel_Optim_Method = ["Reverse", "Truncated", "Forward", "Reverse", "Implicit"]
METHOD_COLLECTIONS = [Hyper_Optim_Method, Bilevel_Optim_Method]


class GraphKeys(tf.GraphKeys):
    """
    adds some meta_parameters and outer_gradients computation related keys
    """

    METAPARAMETERS = "meta_parameters"
    MODELPARAMETERS = "model_parameters"
    LAGRANGIAN_MULTIPLIERS = "lagrangian_multipliers"
    OUTERGRADIENTS = "outergradients"
    DARTS_DERIVATIVES = "darts_derivatives"
    ZS = "zs"


METAPARAMETERS_COLLECTIONS = [GraphKeys.METAPARAMETERS, GraphKeys.GLOBAL_VARIABLES]


def lagrangian_multipliers(scope=None):
    """
    List of variables in the collection LAGRANGIAN_MULTIPLIERS.

    These variables are created by `far.ReverseHG`.

    :param scope: (str) an optional scope.
    :return: A list of tensors (usually variables)
    """
    return tf.get_collection(GraphKeys.LAGRANGIAN_MULTIPLIERS, scope=scope)


def hypergradients(scope=None):
    """
    List of tensors and/or variables in the collection OUTERGRADIENTS.

    These variables are created by `far.HyperGradient`.

    :param scope: (str) an optional scope.
    :return: A list of tensors (usually variables)
    """
    return tf.get_collection(GraphKeys.OUTERGRADIENTS, scope=scope)


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


def outer_parameters(scope=None):
    """
    List of variables in the collection HYPERPARAMETERS.

    Hyperparameters constructed with `get_outerparameter` are in this collection by default.

    :param scope: (str) an optional scope.
    :return: A list of tensors (usually variables)
    """
    return tf.get_collection(GraphKeys.METAPARAMETERS, scope=scope)


def get_outerparameter(
    name,
    initializer=None,
    shape=None,
    dtype=None,
    collections=None,
    scalar=False,
    regularizer=None,
    constraint=None,
):
    """
    Creates an hyperparameter variable, which is a GLOBAL_VARIABLE
    and HYPERPARAMETER. Mirrors the behavior of `tf.get_variable`.

    :param name: name of this hyperparameter
    :param initializer: initializer or initial value (can be also np.array or float)
    :param shape: optional shape, may be not needed depending on initializer
    :param dtype: optional type,  may be not needed depending on initializer
    :param collections: optional additional collection or list of collections, which will be added to
                        METAPARAMETER and GLOBAL_VARIABLES
    :param scalar: default False, if True splits the hyperparameter in its scalar components, i.e. each component
                    will be a single scalar hyperparameter. In this case the method returns a tensor which of the
                    desired shape (use this option with `ForwardHG`)
    :param constraint: optional contstraint for the variable (only if not scalar..)

    :return: the newly created variable, or, if `scalar` is `True` a tensor composed by scalar variables.
    """
    _coll = list(METAPARAMETERS_COLLECTIONS)
    if collections:
        _coll += utils.as_list(collections)
    if not scalar:
        try:
            return tf.get_variable(
                name,
                shape,
                dtype,
                initializer,
                trainable=False,
                collections=_coll,
                constraint=constraint,
                regularizer=regularizer
            )
        except TypeError as e:
            print(e)
            print("Trying to ignore constraints (to use constraints update tensorflow.")
            return tf.get_variable(
                name, shape, dtype, initializer, trainable=False, collections=_coll
            )
    else:
        with tf.variable_scope(name + "_components"):
            _shape = shape or initializer.shape
            if isinstance(_shape, tf.TensorShape):
                _shape = _shape.as_list()
            _tmp_lst = np.empty(_shape, object)
            for k in range(np.multiply.reduce(_shape)):
                indices = np.unravel_index(k, _shape)
                _ind_name = "_".join([str(ind) for ind in indices])
                _tmp_lst[indices] = tf.get_variable(
                    _ind_name,
                    (),
                    dtype,
                    initializer if callable(initializer) else initializer[indices],
                    trainable=False,
                    collections=_coll,
                    regularizer=regularizer
                )
        return tf.convert_to_tensor(_tmp_lst.tolist(), name=name)


def hyperparameters(scope=None):
    """
    List of variables in the collection HYPERPARAMETERS.

    Hyperparameters constructed with `get_outerparameter` are in this collection by default.

    :param scope: (str) an optional scope.
    :return: A list of tensors (usually variables)
    """
    return tf.get_collection(GraphKeys.HYPERPARAMETERS, scope=scope)
