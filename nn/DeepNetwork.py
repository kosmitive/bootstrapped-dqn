# import necessary packages
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


class DeepNetwork:
    """This represents a general DeepNetwork. It can be used in different
    types of contexts. In general it operates on batches of 1-dimensional
    data. Several copies can be created with the set_scope function."""

    def __init__(self, structure, config):
        """This constructs a new DeepQNetwork. It therefore constructs
        the corresponding layers of the network.

        Args:
            env: The environment to use
            structure: The structure for the hidden layers, a list of integers.
            config:
                - layer-norm: bool, activate Layer Normalization
        """

        # obtain the spaces
        self.scope = None
        self.trainable_vars = None
        self.config = config
        self.structure = structure
        self.log = {}

    def switch(self, scope):
        """Simply switch the scope, e.g. for creating a second version of the network
        """

        self.scope = scope
        self.trainable_vars = self.log[scope] if scope in self.log else None

    def copy_graph(self, target_scope):
        """This copies the values from the passed network to this one.
        Logically the weight matrices need to have the same dimensions.

        Args:
            target_scope: The other DeepQNetwork to take values from.
        """

        assert self.scope is not None

        # create list for updates
        updates = list()
        get_name = lambda x: x.name

        # iterate over
        for var, var_target in zip(
                sorted(self.log[self.scope], key=get_name),
                sorted(self.log[target_scope], key=get_name)):
            updates.append(var_target.assign(var))
        return tf.group(*updates)

    def create_mask_graph(self, probs):
        masks = list()
        mask_assign = list()
        for si in range(1, len(self.structure) - 1):
            s = self.structure[si]
            mask = tf.get_variable(name="m_{}".format(si), shape=[s], dtype=tf.int32)
            masks.append(mask)
            mask_assign.append(tf.assign(mask, tf.distributions.Bernoulli(probs=probs).sample(sample_shape=[s])))

        self.masks = [tf.cast(mask, tf.float32) for mask in masks]
        self.assign_masks = tf.group(*mask_assign)

        return self.assign_masks

    def get_mask_graph(self):

        return self.masks


    def eval_graph(self, input, dropout_masks = None, zoneout_masks = None, layer_norm=True, train=False):
        """This creates the graph it therefore receives the input tensor and
        of course the weights for each layer.

        Args:
            input: The input to the graph. Usually a placeholder

        Returns:
            A fully constructed graph using the weights supplied.

        """

        assert input.shape[1] == self.structure[0]
        assert self.scope is not None
        debug = False
        seed = 17

        with tf.variable_scope(self.scope, reuse=(self.scope in self.log)):

            Q = input
            for hidden_num in range(1, len(self.structure) - 1):
                hidden = self.structure[hidden_num]
                x = layers.fully_connected(Q, num_outputs=hidden, activation_fn=None,
                                           #weights_regularizer=layers.l2_regularizer(0.05),
                                           #biases_regularizer=layers.l2_regularizer(0.05),
                                           weights_initializer=
                                               layers.variance_scaling_initializer(
                                                   factor=2.0/(1+0.1**2),
                                                   mode='FAN_IN',
                                                   seed=seed * hidden_num,
                                                   uniform=False,
                                                   dtype=tf.float32)
                                               )

                if layer_norm: x = layers.layer_norm(x, center=True, scale=True)
                if dropout_masks is not None: x = tf.expand_dims(dropout_masks[hidden_num - 1], 0) * x

                pQ = self.lrelu(x)
                if zoneout_masks is not None and hidden_num > 1:
                    exp_mask = tf.expand_dims(zoneout_masks[hidden_num - 2], 0)
                    Q = exp_mask * pQ + (1 - exp_mask) * Q
                else:
                    Q = pQ

            Q = layers.fully_connected(Q, num_outputs=self.structure[-1], activation_fn=None,
                                       weights_initializer=
                                               layers.variance_scaling_initializer(
                                                   factor=2.0/(1+0.05**2),
                                                   mode='FAN_IN',
                                                   seed=seed * hidden_num,
                                                   uniform=False,
                                                   dtype=tf.float32)
                                               )

        if self.scope not in self.log:
            self.trainable_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/{}".format(tf.get_variable_scope().name, self.scope))

            self.log[self.scope] = self.trainable_vars

        return Q

    def lrelu(self, x, leak=0.05):
        return tf.maximum(x, x * leak)

    def grid_graph(self, grid_dims, int_begin, int_end, masks=None):
        """This creates a grid evaluation graph for the network. This
        can be particular useful, when a heat plot of the q function
        should be generated."""

        assert self.scope is not None
        debug = False

        # calc x and y range
        x_range = tf.range(int_begin[0], int_end[0], (int_end[0] - int_begin[0] - 0.0001) / (grid_dims[0] - 1), dtype=tf.float32)
        y_range = tf.range(int_begin[1], int_end[1], (int_end[1] - int_begin[1] - 0.0001) / (grid_dims[1] - 1), dtype=tf.float32)

        # determine the width itself
        width = grid_dims[0] * grid_dims[1]
        x_elements, y_elements = tf.meshgrid(x_range, y_range)

        # reshape both array
        shaped_x_elements = tf.reshape(x_elements, [width])
        shaped_y_elements = tf.reshape(y_elements, [width])

        # concat them
        all_elements = tf.stack([shaped_x_elements, shaped_y_elements], axis=1)

        # evalute all of them
        q = self.eval_graph(all_elements)

        # split them up
        q_list = tf.unstack(q, axis=1)

        # resize them again
        grid_actions = [tf.reshape(q_list[i], [grid_dims[1], grid_dims[0]]) for i in range(len(q_list))]

        # and pass back
        return tf.Print(grid_actions, [grid_actions], "Actions are ", summarize=100) if debug else grid_actions

    def learn_graph(self, learning_rate, X, Y, count=None):
        """This creates the graph it therefore receives the input tensor and
        of course the weights for each layer.

        Args:
            learning_rate: The learning rate to use
            X: The training graph itself
            Y: The actions which the network should output
            count: The agent step

        Returns:
            A fully constructed graph using the weights supplied.

        """
        debug = False

        # get huber loss from the error
        pX = tf.Print(X, [X], "Input is: ", summarize=100)
        pY = tf.Print(Y, [Y], "Target is: ", summarize=100)
        loss = tf.reduce_mean(self.huber_loss(1.0, (pX if debug else X) - (pY if debug else Y)))

        # build reg
        reg = 0
        for v in self.log[self.scope]:
            reg += tf.nn.l2_loss(v)

        # Create a minimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss, var_list=self.log[self.scope])
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, 3.0), var)

        return optimizer.apply_gradients(gradients, global_step=count)

    def squared_loss(self, x):
        return 0.5 * tf.square(x)

    def huber_loss(self, delta, x):

        # Define the error term
        squared_error = self.squared_loss(x)
        abs_error = tf.abs(x)
        cond = tf.less(abs_error, delta)
        return tf.where(cond, squared_error, delta * (abs_error - 0.5 * delta))