from collections import OrderedDict
import tensorflow as tf
import tensorflow.contrib.layers as tcl

from boml.extension import GraphKeys, remove_from_collection
from boml.setup_model.network import BOMLNet
from boml.utils import as_tuple_or_list, remove_from_collection


class BOMLNetFC(BOMLNet):
    def __init__(
            self,
            _input,
            task_parameter=None,
            name="BMLNetFeedForward",
            activation=tf.nn.relu,
            data_type=tf.float32,
            var_collections=tf.GraphKeys.MODEL_VARIABLES,
            output_weight_initializer=tf.contrib.layers.xavier_initializer(tf.float32),
            conv_initializer=tf.contrib.layers.xavier_initializer_conv2d(tf.float32),
            norm=tcl.batch_norm,
            channels=1,
            dim_hidden=[10],
            kernel=3,
            max_pool=False,
            reuse=False,
            use_t=False,
            use_warp=False,
            outer_method="Reverse",
    ):
        self.dim_hidden = as_tuple_or_list(dim_hidden)
        self.activation = activation
        self.data_type = data_type
        self.var_collections = var_collections
        self.kernel = kernel
        self.channels = channels
        self.datatype = data_type
        self.batch_norm = norm
        self.max_pool = max_pool
        self.stride = [1, 2, 2, 1]
        self.no_stride = [1, 1, 1, 1]
        self.bias_initializer = tf.zeros_initializer(tf.float32)
        self.conv_initializer = conv_initializer
        self.output_weight_initializer = output_weight_initializer
        self.use_t = use_t
        self.use_warp = use_warp
        self.outer_method = outer_method
        self.flatten = False

        super().__init__(
            _input=_input,
            name=name,
            var_collections=var_collections,
            task_parameter=task_parameter,
            reuse=reuse,
        )

        self.betas = self.filter_vars("beta")

        self.moving_means = self.filter_vars("moving_mean")
        self.moving_variances = self.filter_vars("moving_variance")

        if not reuse:
            remove_from_collection(
                tf.GraphKeys.MODEL_VARIABLES, *self.moving_means
            )
            remove_from_collection(
                tf.GraphKeys.MODEL_VARIABLES, *self.moving_variances
            )
            print(name, "MODEL CREATED")
        remove_from_collection(
            tf.GraphKeys.MODEL_VARIABLES,
            *self.moving_means,
            *self.moving_variances)


    def _forward(self):

        if not isinstance(self.task_parameter, dict):
            self.create_initial_parameter()
        self + tf.add(
            tf.matmul(self.out, self.task_parameter["fc_weight0"]),
            self.task_parameter["fc_bias0"],
        )
        self + self.activation(self.out)
        self + tf.add(
            tf.matmul(self.out, self.task_parameter["fc_weight1"]),
            self.task_parameter["fc_bias1"],
        )
        #self + self.activation(self.out)


    def create_initial_parameter(self):
        self.task_parameter = OrderedDict()
        '''
        for i in range(len(self.dim_hidden)-1):
            self.task_parameter["conv" + str(i)] = self.get_conv_weight(
                layer=i, initializer=self.conv_initializer
            )
        '''
        self.task_parameter["fc_weight0"] = tf.get_variable(
            "fc_weight0",
            shape=[self.layers[-1].shape[1], 300],
            initializer=self.output_weight_initializer,
            dtype=self.data_type,
        )
        self.task_parameter["fc_bias0"] = tf.get_variable(
            "fc_bias0",
            [300],
            initializer=tf.zeros_initializer,
            dtype=self.data_type,
        )
        self.task_parameter["fc_weight1"] = tf.get_variable(
            "fc_weight1",
            shape=[300, self.dim_hidden[-1]],
            initializer=self.output_weight_initializer,
            dtype=self.data_type,
        )
        self.task_parameter["fc_bias1"] = tf.get_variable(
            "fc_bias1",
            [self.dim_hidden[-1]],
            initializer=tf.zeros_initializer,
            dtype=self.data_type,
        )
        [
            tf.add_to_collections(self.var_collections, initial_param)
            for initial_param in self.task_parameter.values()
        ]
        remove_from_collection(
            GraphKeys.GLOBAL_VARIABLES, *self.task_parameter.values()
        )

    def re_forward(self, new_input=None, task_parameter=OrderedDict()):
        return BOMLNetFC(
            new_input if new_input is not None else self.layers[0],
            dim_hidden=self.dim_hidden,
            task_parameter=task_parameter
            if len(task_parameter) > 0
            else self.task_parameter,
            name=self.name,
            activation=self.activation,
            data_type=self.data_type,
            var_collections=self.var_collections,
            output_weight_initializer=self.output_weight_initializer,
            reuse=tf.AUTO_REUSE,
            use_t=self.use_t,
        )
