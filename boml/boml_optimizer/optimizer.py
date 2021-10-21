from collections import defaultdict, OrderedDict

import numpy as np
import tensorflow as tf

from boml import utils, extension
from boml.load_data import BOMLExperiment

importlib = __import__("importlib")
boml_networks = importlib.import_module("boml.setup_model")
inner_grads = importlib.import_module("boml.lower_iter")
hyper_grads = importlib.import_module("boml.upper_iter")
boml_optimizer = importlib.import_module("boml.optimizer")
tf_optimizer = importlib.import_module("tensorflow.train")


class BOMLOptimizer(object):
    """
    Wrapper for performing gradient-based metaparameter optimization
    """

    def __init__(
        self,
        method=None,
        inner_method=None,
        outer_method=None,
        truncate_iter=-1,
        experiments=[],
    ):
        """
        BMLHOptimizer is the main class in `pybml`, which takes responsibility for
        the whole process of model construnction and back propagation.

        :param Method: define basic method for following training process, it should be included in ['MetaInit', 'MetaRepr'],
         'MetaInit' type includes methods like 'MAML, FOMAML, TNet, WarpGrad'; 'MetaRepr' type includes methods like
         'BDA, RHG, Truncated, Implicit HG, DARTS';
        :param inner_method: method chosen for solving LLproblem, including ['Trad' ,'Simple', 'Aggr'], 'MetaRepr' type choose
        either 'Trad' for traditional optimization strategies or 'Aggr' for Gradient Aggragation optimization 'MetaInit' type
        should choose 'Simple', and set specific parameters for detailed method choices like FOMAML or TNet.
        :param outer_method: method chosen for solving LLproblem, including ['Reverse' ,'Simple', 'Forward', 'Implcit'],
        'MetaInit' type should choose Simple, and set specific parameters for detailed method choices like FOMAML
        :param truncate_iter: specific parameter for Truncated Reverse method, defining number of iterations to truncate
         in the Back propagation process
        :param experiments: list of experiment objects that has already been initialized
        :return :an initialized instance of BMLHOptimizer
        """
        assert method in ("MetaRepr", "MetaInit"), (
            "initialize method arguement, should be in list \an [MetaRepr,MetaInitl] "
            "MetaRepr based methods include [BDA,FHG,RHG,TRHG],"
            "HperOptim based methods include [MAML,FOMAML,MSGD]"
        )
        self._method = method
        assert inner_method in (
            "Aggr",
            "Simple",
            "Trad",
        ), "initialize method arguement, should be in list [Aggr, Simple, Trad]"
        self.inner_method = inner_method

        if self.inner_method == "Simple":
            assert outer_method == "Simple", (
                "Choose simple configuration of lower-level and upper-level "
                "calculation strategy for meta-initialization-based method"
            )
        elif self.inner_method == "Aggr":
            assert outer_method in ("Darts", "Reverse",), (
                "The bilevel aggregation strategy could choose Reverse Auto Differentiation"
                " or DARTS as upper-level calculation strategy"
            )
        else:
            assert outer_method in ("Reverse", "Implicit", "Darts","Forward"), (
                "Invalid combination of inner and upper strategies, "
                "please check initialization for different level of problems or "
                "extend the base classes to formulate your own problems definition"
            )
        self.outer_method = outer_method
        self._inner_gradient = getattr(
            inner_grads, "%s%s" % ("BOMLInnerGrad", self.inner_method)
        )
        if truncate_iter >= 0:
            self._outer_gradient = getattr(
                hyper_grads, "%s%s" % ("BOMLOuterGrad", self.outer_method)
            )(inner_method=self.inner_method, truncate_iter=truncate_iter)
        else:
            self._outer_gradient = getattr(
                hyper_grads, "%s%s" % ("BOMLOuterGrad", self.outer_method)
            )(inner_method=self.inner_method)
        self.io_opt = None
        self._learning_rate = None
        self._meta_learning_rate = None
        self.oo_opt = None
        self.data_set = None
        self._meta_learner = None
        self._fin_hts = None
        self._global_step = None
        self._o_optim_dict = defaultdict(lambda: set())
        self._param_dict = OrderedDict()
        self._global_step = tf.get_variable(
            "global_step", initializer=0, trainable=False
        )
        self._inner_objective = None
        self._truncate_iterations = -1
        self.experiments = experiments
        # self.inner_objectives = []

    def meta_learner(
        self,
        _input,
        dataset,
        meta_model="v1",
        name="Hyper_Net",
        use_t=False,
        use_warp=False,
        **model_args
    ):
        """
        This method must be called once at first to build meta modules and initialize meta parameters and neural networks.
        :param _input: orginal input for neural network construction
        :param dataset: which dataset to use for training and testing. It should be initialized before being passed into the function
        :param meta_model: model chosen for neural network construction, 'v1' for C4L with fully connected layer,
        'v2' for Residual blocks with fully connected layer.
        :param name:  name for Meta model modules used for BMLNet initialization
        :param use_t: whether to use T layer for C4L neural networks
        :return: BMLNet object containing the dict of hyper parameters
        """

        self.param_dict["use_t"] = use_t
        self.param_dict["use_warp"] = use_warp

        self.data_set = dataset
        assert meta_model.startswith(
            "V"
        ), "The dataset does not match the model chosen for meta_learner, V1,V2,...or Vk"
        meta_learner = getattr(
            boml_networks,
            "%s%s%s" % ("BOMLNet", dataset.train.name, self.method + meta_model),
        )(
            _input=_input,
            dim_output=dataset.train.dim_target,
            name=name,
            use_t=use_t,
            use_warp=use_warp,
            outer_method=self.outer_method,
            **model_args
        )
        self.param_dict["meta_model"] = meta_model
        self.param_dict["meta_learner"] = meta_learner
        self._meta_learner = meta_learner

        return meta_learner

    def base_learner(
        self,
        _input,
        meta_learner,
        name="task_net",
        weights_initializer=tf.zeros_initializer,
    ):
        """
        This method has to be called for every experiment and takes responsibility
        for defining task-specific modules and inner optimizers.
        :param _input: orginal input for neural network construction of task-specific module
        :param meta_learner: returned value of Meta_model function, which is a instance of BMLNet or its child classes
        :param name: name for Base model modules used for BMLNet initialization
        :param weights_initializer: initializer function for task_specific network, called by 'MetaRepr' method
        :return: task-specific model part
        """
        if self.method == "MetaInit":
            base_learner = getattr(
                boml_networks,
                "%s%s%s"
                % (
                    "BOMLNet",
                    self.data_set.train.name,
                    self.method + self.param_dict["meta_model"],
                ),
            )(
                _input=_input,
                outer_param_dict=meta_learner.outer_param_dict,
                model_param_dict=meta_learner.model_param_dict,
                dim_output=meta_learner.dims[-1],
                name=name,
                use_t=meta_learner.use_t,
                outer_method=self.outer_method,
                use_warp=meta_learner.use_warp,
            )
        elif self.method == "MetaRepr":
            if self.param_dict["meta_model"]== 'V3':
                base_learner = getattr(boml_networks, "BOMLNetFC")(
                    _input=_input,
                    output_weight_initializer=weights_initializer,
                    name=name,
                    use_t=meta_learner.use_t,
                )  # dim_hidden=self.data_set.train.dim_target,
            else:
                base_learner = getattr(boml_networks, "BOMLNetFeedForward")(
                    _input=_input,
                    output_weight_initializer=weights_initializer,
                    name=name,
                    use_t=meta_learner.use_t,
                    dim_hidden=self.data_set.train.dim_target,
                )
        else:
            print(
                "initialize method arguement, should be in list \an [MetaRepr,MetaInitl]"
            )
            raise AssertionError
        return base_learner

    def ll_problem(
        self,
        inner_objective,
        learning_rate,
        T,
        inner_objective_optimizer="SGD",
        outer_objective=None,
        learn_lr=False,
        alpha_init=0.0,
        s=1.0, t=1.0,
        gamma=1.0,
        learn_alpha=False,
        learn_st=False,
        learn_alpha_itr=False,
        var_list=None,
        first_order=False,
        model_loss_func=utils.cross_entropy,
        outer_loss_func=utils.cross_entropy,
        loss_func=utils.cross_entropy,
        momentum=0.5,
        experiment=None,
        **inner_kargs
    ):
        """
        After construction of neural networks, solutions to lower level problems should be regulated in LL_Problem.
        :param inner_objective: loss function for the inner optimization problem
        :param learning_rate: step size for inner loop optimization
        :param T: numbers of steps for inner gradient descent optimization
        :param inner_objective_optimizer: Optimizer type for the outer parameters, should be in list ['SGD','Momentum','Adam']
        :param outer_objective: loss function for the outer optimization problem, which need to be claimed in BDA agorithm
        :param learn_lr: BOOLEAN, which determines whether to define learning rate as an outer parameter
        :param alpha_init: initial value of ratio of inner objective to outer objective in BDA algorithm
        :param s: coefficients multiplied by UL objectives in BDA algorithm, default to be 1.0
        :param t: coefficients multiplied by LL objectives in BDA algorithm, default to be 1.0
        :param gamma: coefficients multiplied by outer objectives in BDA algorithm, default to be 1.0
        :param learn_alpha: BOOLEAN, specify to decide whether to initialize alpha as an outer parameter for BDA algorithm
        :param learn_st:BOOLEAN, specify whether to initialize alpha as a hyper parameter
        :param learn_alpha_itr: parameter for BDA algorithm to specify whether to initialize alpha as a vector, of which
        every dimension's value is step-wise scale factor fot the optimization process
        :param learn_st: specify parameter for BDA algorithm to decide whether to initialize s and t as hyper parameters
        :param first_order: specific parameter to define whether to use implement first order MAML, default to be `FALSE`
        :param loss_func: specifying which type of loss function is used for the maml-based method, which should be
        consistent with the form to compute the inner objective
        :param model_loss_func: forms of loss functions of task-specific layer corresponding to 'WarpGrad' method, default to be cross-entropy
        :param outer_loss_func: forms of loss functions of task-specific layer corresponding to 'WarpGrad' method, default to be cross-entropy

        :param momentum: specific parameter for Optimizer.BMLOptMomentum to set initial value of momentum
        :param experiment: instance of Experiment to use in the Lower Level Problem, especifially needed in the
         `MetaInit` type of methods.
        :param var_list: optional list of variables (of the inner optimization problem)from
        :param inner_kargs: optional arguments to pass to `py_bml.core.optimizer.minimize`
        :return: `BOMLInnerGrad` from boml.lower_iter.
        """
        if self._learning_rate is None:
            if learn_lr:
                self.param_dict["learn_lr"] = learn_lr
                self._learning_rate = tf.Variable(
                    name="lr",
                    initial_value=learning_rate,
                    dtype=tf.float32,
                    collections=extension.METAPARAMETERS_COLLECTIONS,
                    trainable=False,
                )
            else:
                self._learning_rate = tf.constant(learning_rate, name="lr")
            self.param_dict["learning_rate"] = self._learning_rate
        if self.io_opt is None:
            if self.inner_method == "Simple" or inner_objective_optimizer == "SGD":
                self.io_opt = getattr(
                    boml_optimizer, "%s%s" % ("BOMLOpt", inner_objective_optimizer)
                )(learning_rate=self._learning_rate, name=inner_objective_optimizer)
            elif inner_objective_optimizer == "Momentum":
                self.io_opt = getattr(boml_optimizer, "%s%s" % ("BOMLOpt", "Momentum"))(
                    learning_rate=self._learning_rate,
                    momentum=momentum,
                    name=inner_objective_optimizer,
                )
            else:
                self.io_opt = getattr(
                    boml_optimizer, "%s%s" % ("BOMLOpt", inner_objective_optimizer)
                )(learning_rate=self._learning_rate, name=inner_objective_optimizer)
        assert isinstance(self.io_opt, getattr(boml_optimizer, "BOMLOpt")), (
            "Must use an optimizer that extends "
            "the class boml.optimizers "
            "found {} instead".format(type(self.io_opt))
        )
        assert self._method in (
            "MetaRepr",
            "MetaInit",
        ), "illegal initialization value for argument:method, should be in [MetaRepr, MetaInit]"
        if self.method == "MetaRepr":
            if self.inner_method == "Aggr":
                assert (
                    outer_objective is not None
                ), "BDA must have upper-level loss functions passed to lower-level problems optimization process"

                if "gamma" not in self._param_dict.keys():
                    gamma = tf.constant(gamma, name="gamma")
                    self._param_dict["gamma"] = gamma

                if not (
                    ("s" in self._param_dict.keys()) or ("t" in self._param_dict.keys())
                ):
                    if learn_st:
                        s = extension.get_outerparameter("s", s)
                        t = extension.get_outerparameter("t", t)
                    else:
                        s = tf.constant(s, name="s")
                        t = tf.constant(t, name="t")
                    self._param_dict["s"] = s
                    self._param_dict["t"] = t

                if "alpha" not in self._param_dict.keys():
                    if learn_alpha_itr:
                        alpha_vec = np.ones((1, T), dtype=np.float32) * alpha_init
                        alpha = extension.get_outerparameter(
                            name="alpha", initializer=alpha_vec
                        )
                        t_tensor = tf.placeholder(
                            shape=(T, 1), dtype=tf.float32, name="t_tensor"
                        )
                    else:
                        alpha = (
                            extension.get_outerparameter(
                                initializer=alpha_init, name="alpha"
                            )
                            if learn_alpha
                            else tf.constant(alpha_init, name="alpha")
                        )
                        t_tensor = tf.placeholder(dtype=tf.float32, name="t_tensor")
                    self._param_dict["alpha"] = alpha
                    self._param_dict["t_tensor"] = t_tensor
        elif self.method == "MetaInit":
            self._param_dict["first_order"] = first_order

        assert isinstance(experiment, BOMLExperiment), (
            "MetaInit based methods require specialized "
            "task model for each generated task,"
            "please refer to basic instruction for modules of networks"
        )
        self.param_dict["model_loss_func"] = model_loss_func
        self.param_dict["outer_loss_func"] = outer_loss_func
        self._param_dict["loss_func"] = loss_func
        self._param_dict["experiment"] = experiment

        self._param_dict["T"] = T

        inner_grad = self._inner_gradient.compute_gradients(
            boml_pot=self.io_opt,
            loss_inner=inner_objective,
            loss_outer=outer_objective,
            param_dict=self._param_dict,
            var_list=var_list,
            **{
                inner_arg: inner_kargs[inner_arg]
                for inner_arg in set(inner_kargs.keys()) - set(self.param_dict.keys())
            }
        )
        if hasattr(inner_grad, "objective"):
            inner_objective = inner_grad.objective
            if self._inner_objective is None:
                self._inner_objective = [inner_objective]
            else:
                self._inner_objective = tf.concat(
                    (self._inner_objective, [inner_objective]), axis=0
                )
        else:
            pass

        return inner_grad

    def ul_problem(
        self,
        outer_objective,
        meta_learning_rate,
        inner_grad,
        mlr_decay=1.0e-5,
        meta_param=None,
        outer_objective_optimizer="Adam",
        epsilon=1.0,
        momentum=0.5,
        warp_lambda=1.0,
        tolerance=lambda _k: 0.1 * (0.9 ** _k),
    ):
        """
        Set the outer optimization problem and the descent procedure for the optimization of the
        outer parameters. Can be called at least once for every call of inner_problem, passing the resulting
         `OptimizerDict`. It can be called multiple times with different objective, optimizers and hyper_list s.

        :param outer_objective: scalar tensor for the outer objective
        :param meta_learning_rate: step size for outer loop optimization
        :param inner_grad: Returned value of py_bml.BMLHOptimizer.LLProblem()
        :param meta_param: optional list of outer parameters and model parameters
        :param outer_objective_optimizer: Optimizer type for the outer parameters,
        should be in list ['SGD','Momentum','Adam']
        :param epsilon: Float, cofffecients to be used in DARTS algorithm
        :param momentum: specific parameters to be used to initialize 'Momentum' algorithm
        :param tolerance: specific function template for Implicit HG Algorithm
        :param global_step: optional global step. By default tries to use the last variable
        in the collection GLOBAL_STEP
        :return: itself
        """
        if self._meta_learning_rate is None:
            self._meta_learning_rate = tf.train.inverse_time_decay(
                meta_learning_rate,
                self.global_step,
                decay_steps=1.0,
                decay_rate=mlr_decay,
            )
        if self.oo_opt is None:
            if self.oo_opt is None:
                if outer_objective_optimizer == "Momentum":
                    self.oo_opt = tf.train.MomentumOptimizer(
                        learning_rate=self._meta_learning_rate,
                        momentum=momentum,
                        name=outer_objective_optimizer,
                    )
                elif outer_objective_optimizer == "Adam":
                    self.oo_opt = tf.train.AdamOptimizer(
                        learning_rate=self._meta_learning_rate,
                        name=outer_objective_optimizer,
                    )
                elif outer_objective_optimizer == "SGD":
                    self.oo_opt = tf.train.GradientDescentOptimizer(
                        learning_rate=self._meta_learning_rate,
                        name=outer_objective_optimizer,
                    )
        assert isinstance(
            self._outer_gradient, getattr(hyper_grads, "BOMLOuterGrad")
        ), "Wrong name for inner method,should be in list \n [Reverse, Simple, Forward, Implicit]"
        if 'use_warp' in self.param_dict.keys() and self.param_dict['use_warp']:
            setattr(self.outergradient, "lambda", tf.cast(warp_lambda, tf.float32))

        if self.outer_method == "Darts" and (
            not hasattr(self.outergradient, "Epsilon")
        ):
            assert isinstance(
                self._outer_gradient, getattr(hyper_grads, "BOMLOuterGradDarts")
            ), "Wrong name for outer method,should be in list [Darts]"
            setattr(self.outergradient, "Epsilon", tf.cast(epsilon, tf.float32))
            setattr(self.outergradient, "param_dict", self.param_dict)
        if self.outer_method == "Implicit":
            self.outergradient.set_tolerance(tolerance=tolerance)
        meta_param = self.outergradient.compute_gradients(
            outer_objective,
            inner_grad,
            meta_param=meta_param,
            param_dict=self.param_dict,
        )
        self._o_optim_dict[self.oo_opt].update(meta_param)

        return self

    def minimize(
        self,
        inner_objective,
        learning_rate,
        meta_learning_rate,
        T,
        inner_objective_optimizer="SGD",
        outer_objective=None,
        learn_lr=False,
        alpha_init=0.0,
        gamma=1.0,
        learn_alpha=False,
        learn_st=False,
        learn_alpha_itr=False,
        var_list=None,
        init_dynamics_dict=None,
        first_order=False,
        loss_func=utils.cross_entropy,
        momentum=0.5,
        beta1=0.0,
        beta2=0.999,
        experiment=None,
        meta_param=None,
        outer_objective_optimizer="Adam",
        epsilon=1.0,
        tolerance=lambda _k: 0.1 * (0.9 ** _k),
        global_step=None,
        **inner_kargs
    ):
        """
        calling once 'inner_problem', 'outer_problem' and 'aggregate_all', and optionally set an initial dynamics.
        For more complex uses (like inner problems batching) use the methods separately. This can be used only when
        a single problem setting are expected for your problem definition.
        Returns method `BOMLOptimizer.run`, that runs one hyperiteration.
        """
        inner_grad = self.ll_problem(
            inner_objective=inner_objective,
            learning_rate=learning_rate,
            inner_objective_optimizer=inner_objective_optimizer,
            experiment=experiment,
            outer_objective=outer_objective,
            T=T,
            gamma=gamma,
            first_order=first_order,
            learn_lr=learn_lr,
            learn_alpha=learn_alpha,
            learn_st=learn_st,
            learn_alpha_itr=learn_alpha_itr,
            var_list=var_list,
            alpha_init=alpha_init,
            momentum=momentum,
            beta2=beta2,
            beta1=beta1,
            loss_func=loss_func,
            init_dynamics_dict=init_dynamics_dict,
            **inner_kargs
        )
        self.ul_problem(
            outer_objective=outer_objective,
            meta_learning_rate=meta_learning_rate,
            inner_grad=inner_grad,
            outer_objective_optimizer=outer_objective_optimizer,
            meta_param=meta_param,
            global_step=global_step,
            tolerance=tolerance,
            momentum=momentum,
            epsilon=epsilon,
        )

    def aggregate_all(self, aggregation_fn=None, gradient_clip=None):
        """
        To be called when no more dynamics or problems will be added, computes the updates
        for the outer parameters. This behave nicely with global_variables_initializer.

        :param aggregation_fn: Optional operation to aggregate multiple hypergradients (for the same hyperparameter),
                                by (default: reduce_mean)
        :param gradient_clip: Optional operation like normalizing to be applied to hypergradients before performing
                            a descent step (default: nothing).

        :return: the run method of this object.
        """
        if self._fin_hts is None:
            # in this way also far.optimizer can be used
            _maybe_first_arg = lambda _v: _v[0] if isinstance(_v, tuple) else _v

            def maybe_first_arg(_v):
                return _v[0] if isinstance(_v, tuple) else _v

            self._fin_hts = tf.group(
                *[
                    maybe_first_arg(
                        opt.apply_gradients(
                            self.outergradient.hgrads_hvars(
                                meta_param=hll,
                                aggregation_fn=aggregation_fn,
                                gradient_clip=gradient_clip,
                            )
                        )
                    )
                    for opt, hll in self._o_optim_dict.items()
                ]
            )
            if self._global_step:
                with tf.control_dependencies([self._fin_hts]):
                    self._fin_hts = self._global_step.assign_add(1).op
        return self.run

    def run(
        self,
        inner_objective_feed_dicts=None,
        outer_objective_feed_dicts=None,
        initializer_feed_dict=None,
        session=None,
    ):
        """
        Run an hyper-iteration (i.e. train the model(s) and compute hypergradients) and updates the upper parameters.

        :param inner_objective_feed_dicts: an optional feed dictionary for the inner problem. Can be a function of
                                            step, which accounts for, e.g. stochastic gradient descent.
        :param outer_objective_feed_dicts: an optional feed dictionary for the upper-level optimization problem
                                            (passed to the evaluation of upper objective). Can be a function of
                                            hyper-iterations steps (i.e. global variable), which may account for, e.g.
                                            stochastic evaluation of upper objective.
        :param initializer_feed_dict:  an optional feed dictionary for the initialization of inner problems variables.
                                            Can be a function of hyper-iterations steps (i.e. global variable),
                                            which may account for, e.g. stochastic initialization.
        :param session: optional session
        """
        self._outer_gradient.apply_gradients(
            inner_objective_feed_dicts,
            outer_objective_feed_dicts,
            initializer_feed_dict,
            param_dict=self._param_dict,
            session=session,
            global_step=self._global_step,
        )
        ss = session or tf.get_default_session()

        def _opt_fd():
            # e.g. hyper-learning rate is a placeholder
            _io_fd = (
                utils.maybe_call(
                    inner_objective_feed_dicts, utils.maybe_eval(self._global_step)
                )
                if inner_objective_feed_dicts
                else {}
            )
            _oo_fd = (
                utils.maybe_call(
                    outer_objective_feed_dicts, utils.maybe_eval(self._global_step)
                )
                if outer_objective_feed_dicts
                else {}
            )
            return utils.merge_dicts(_io_fd, _oo_fd)

        ss.run(self._hyperit, _opt_fd())

    @property
    def meta_model(self):
        """
        :return: the created BMLNet object
        """
        return self._meta_learner

    @property
    def outergradient(self):
        """
        :return: the outergradient object underlying this wrapper.
        """
        return self._outer_gradient

    @property
    def innergradient(self):
        """
        :return: the innergradient object underlying this wrapper.
        """
        return self._inner_gradient

    @property
    def learning_rate(self):
        """
        :return: the outergradient object underlying this wrapper.
        """
        return self._learning_rate

    @property
    def meta_learning_rate(self):
        """
        :return: the outergradient object underlying this wrapper.
        """
        return self._meta_learning_rate

    @property
    def method(self):
        """
        :return: the method for whole algorithm.
        """
        return self._method

    @property
    def global_step(self):
        """
        :return: globalstep used in whole graph.
        """
        return self._global_step

    @property
    def param_dict(self):
        """
        :return: dict that holds hyper_params used in the inner optimization process.
        """
        return self._param_dict

    @property
    def _hyperit(self):
        """
        iteration of minimization of upperobjective(s), assuming the hyper-gradients are already computed.
        """
        assert (
            self._fin_hts is not None
        ), "Must call BMLHOptimizer.Aggregate_all before performing optimization."
        return self._fin_hts

    @property
    def inner_objectives(self):
        return self._outer_gradient.inner_objectives
