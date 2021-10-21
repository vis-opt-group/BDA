from collections import OrderedDict

import tensorflow as tf

from boml.lower_iter import BOMLInnerGradTrad


class BOMLInnerGradAggr(BOMLInnerGradTrad):
    def __init__(self, update_op, dynamics, objective, outer_objective):
        self.outer_objective = outer_objective
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
        assert {
            "alpha",
            "s", "t",
            "gamma",
            "t_tensor",
        } <= param_dict.keys(), (
            "Necessary hyper_parameters must be initialized before calling minimize()"
        )
        # alpha, loss_outer, s, t, t_tensor = sorted(param_dict.items(), key=lambda x: x[0])
        update_op, dynamics = BOMLInnerGradAggr.bml_inner_grad_aggr(
            inner_optimizer=boml_pot,
            loss_inner=loss_inner,
            loss_outer=loss_outer,
            param_dict=param_dict,
            var_list=var_list,
            *minimize_kargs
        )
        return BOMLInnerGradAggr(
            update_op=update_op,
            dynamics=dynamics,
            objective=loss_inner,
            outer_objective=loss_outer,
        )

    @staticmethod
    def bml_inner_grad_aggr(
        inner_optimizer,
        loss_inner,
        loss_outer,
        param_dict=OrderedDict(),
        global_step=None,
        var_list=None,
        gate_gradients=1,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        name=None,
        grad_loss=None,
    ):

        grads_and_vars_inner = inner_optimizer.compute_gradients(
            loss_inner,
            var_list=var_list,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss,
        )
        grads_and_vars_outer = inner_optimizer.compute_gradients(
            loss_outer,
            var_list=var_list,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss,
        )
        grads_and_vars = BOMLInnerGradAggr.combine_grads(
            inner_grads=grads_and_vars_inner,
            outer_grads=grads_and_vars_outer,
            alpha=param_dict["alpha"],
            s=param_dict["s"], t=param_dict["t"],
            gamma=param_dict['gamma'],
            t_tensor=param_dict["t_tensor"],
        )

        return inner_optimizer.apply_gradients(
            grads_and_vars, global_step=global_step, name=name
        )

    @staticmethod
    def combine_grads(inner_grads, outer_grads, alpha, gamma, s, t, t_tensor):

        combine_grads = []
        if not alpha.get_shape().as_list():
            for _ in range(len(inner_grads)):
                ll_part = (1 - alpha) * t * inner_grads[_][0]
                ul_part = alpha * gamma * (0.9**(t_tensor-1)) * s * outer_grads[_][0]
                combine_grads.append((ll_part + ul_part, inner_grads[_][1]))
        else:
            for _ in range(len(inner_grads)):
                ll_part = (
                        (1 - tf.norm(tf.matmul(alpha, t_tensor), ord=1))
                        * inner_grads[_][0]
                )
                ul_part = (
                        tf.norm(tf.matmul(alpha, t_tensor), ord=1)
                        * gamma
                        * outer_grads[_][0]
                )
                combine_grads.append((ll_part + ul_part, inner_grads[_][1]))
        return combine_grads
