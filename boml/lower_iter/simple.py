from collections import OrderedDict

import tensorflow as tf

from boml.lower_iter import BOMLInnerGradTrad
from boml.utils import add_list


class BOMLInnerGradSimple(BOMLInnerGradTrad):
    def __init__(
        self,
        update_op,
        dynamics,
        objective,
        inner_param_tensor=None,
        outer_param_tensor=[],
        model_param_tensor=[],
    ):
        self._inner_param_fast_tensor = inner_param_tensor
        self._outer_param_tensor = outer_param_tensor
        self._model_param_tensor = model_param_tensor
        super().__init__(update_op=update_op, dynamics=dynamics, objective=objective)

    @staticmethod
    def compute_gradients(
        boml_pot,
        loss_inner,
        loss_outer=None,
        param_dict=OrderedDict(),
        var_list=None,
        **inner_kargs
    ):

        minimize_kargs = {
            inner_arg: inner_kargs[inner_arg]
            for inner_arg in set(inner_kargs.keys()) - set(param_dict.keys())
        }

        assert loss_inner is not None, "argument:inner_objective must be initialized"
        update_op, dynamics = boml_pot.minimize(loss_inner, var_list, *minimize_kargs)
        fast_param, outer_grad, model_grad = BOMLInnerGradSimple.bml_inner_grad_trad(
            loss_inner=loss_inner, param_dict=param_dict, var_list=var_list
        )

        return BOMLInnerGradSimple(
            update_op=update_op,
            dynamics=dynamics,
            objective=loss_inner,
            inner_param_tensor=fast_param,
            outer_param_tensor=outer_grad,
            model_param_tensor=model_grad,
        )

    @staticmethod
    def bml_inner_grad_trad(loss_inner, param_dict=OrderedDict(), var_list=[]):

        assert len(var_list) > 0, "no task_specific variables to optimizer"
        task_model = param_dict["experiment"].model
        task_param = task_model.task_parameter
        outer_param_grad = []
        model_param_grad = []
        grads = tf.gradients(loss_inner, list(task_param.values()))
        if param_dict["use_warp"]:
            outer_param_loss = param_dict["outer_loss_func"](
                pred=task_model.re_forward(param_dict["experiment"].x_).out,
                label=param_dict["experiment"].y_,
                method="MetaInit",
            )
            model_param_loss = param_dict["model_loss_func"](
                pred=task_model.re_forward(param_dict["experiment"].x_).out,
                label=param_dict["experiment"].y_,
                method="MetaInit",
            )
            outer_param_grad = tf.gradients(outer_param_loss, list(var_list))
            model_param_grad = tf.gradients(
                model_param_loss, list(task_model.model_param_dict.values())
            )

        if param_dict["first_order"]:
            grads_dict = dict(
                zip(
                    task_model.task_parameter.keys(),
                    [tf.stop_gradient(grad) for grad in grads],
                )
            )
        else:
            grads_dict = dict(zip(task_model.task_parameter.keys(), grads))

        task_model.task_parameter = OrderedDict(
            zip(
                task_param.keys(),
                [
                    task_param[key] - param_dict["learning_rate"] * grads_dict[key]
                    for key in task_param.keys()
                ],
            )
        )

        for _ in range(param_dict["T"] - 1):
            task_model = task_model.re_forward(param_dict["experiment"].x)
            task_param = task_model.task_parameter

            iter_loss = param_dict["loss_func"](
                pred=task_model.out, label=param_dict["experiment"].y, method="MetaInit"
            )

            if param_dict["use_warp"]:
                outer_param_loss = param_dict["outer_loss_func"](
                    pred=task_model.re_forward(param_dict["experiment"].x_).out,
                    label=param_dict["experiment"].y_,
                    method="MetaInit",
                )
                model_param_loss = param_dict["model_loss_func"](
                    pred=task_model.re_forward(param_dict["experiment"].x_).out,
                    label=param_dict["experiment"].y_,
                    method="MetaInit",
                )
                outer_param_grad = add_list(
                    outer_param_grad, tf.gradients(outer_param_loss, list(var_list))
                )
                model_param_grad = add_list(
                    model_param_grad,
                    tf.gradients(
                        model_param_loss, list(task_model.model_param_dict.values())
                    ),
                )

            grads = tf.gradients(iter_loss, list(task_param.values()))

            if param_dict["first_order"]:
                grads_dict = dict(
                    zip(
                        task_model.task_parameter.keys(),
                        [tf.stop_gradient(grad) for grad in grads],
                    )
                )
            else:
                grads_dict = dict(zip(task_model.task_parameter.keys(), grads))

            task_model.task_parameter = OrderedDict(
                zip(
                    task_param.keys(),
                    [
                        task_param[key] - param_dict["learning_rate"] * grads_dict[key]
                        for key in task_param.keys()
                    ],
                )
            )
        param_dict["experiment"].model = task_model

        return task_model.task_parameter, outer_param_grad, model_param_grad

    @property
    def outer_param_tensor(self):
        """
        :return: temporary weights dictionary used in maml and fomaml for back propagation
        """
        assert (
            self._outer_param_tensor is not None
        ), "temporary weights dictionary must be initialized before being called"
        return self._outer_param_tensor

    @property
    def model_param_tensor(self):
        """
        :return: temporary weights dictionary used in maml and fomaml for back propagation
        """
        assert (
            self._model_param_tensor is not None
        ), "temporary weights dictionary must be initialized before being called"
        return self._model_param_tensor
