# import necessary packages
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


class MultipleHeadDeepNetwork:
    """This represents a general MultipleHeadDeepNetwork. It can be used in different
    types of contexts. In general it operates on batches of 1-dimensional
    data. Several copies can be created with the set_scope function."""

    def __init__(self, shared_structure, personal_structure, K, config):
        """This constructs a new DeepQNetwork. It therefore constructs
        the corresponding layers of the network.

        Args:
            shared_structure: The structure of the shared neural network
            personal_structure: The structure for each head
            K: The number of heads
            config:
                - layer-norm: bool, activate Layer Normalization
        """

        # obtain the spaces
        self.scope = None
        self.trainable_vars = None
        self.config = config
        self.shared_structure = shared_structure
        self.personal_structure = personal_structure
        self.K = K
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

    def eval_graph(self, input, layer_norm=True, train=False):
        """This creates the graph it therefore receives the input tensor and
        of course the weights for each layer.

        Args:
            input: The input to the graph. Usually a placeholder

        Returns:
            A fully constructed graph using the weights supplied.

        """

        assert input.shape[1] == self.shared_structure[0]
        assert self.scope is not None
        debug = False
        initializ = layers.variance_scaling_initializer(
                                                   factor=2.0/(1+0.1**2),
                                                   mode='FAN_IN',
                                                   uniform=False,
                                                   dtype=tf.float32)

        with tf.variable_scope(self.scope, reuse=(self.scope in self.log)):

            Q = input
            for hidden_num in range(1, len(self.shared_structure)):
                hidden = self.shared_structure[hidden_num]
                x = layers.fully_connected(Q, num_outputs=hidden, activation_fn=None, weights_initializer=initializ)

                if layer_norm: x = layers.layer_norm(x, center=True, scale=True)
                Q = self.lrelu(x)

            # create the different heads
            heads = list()
            for k in range(self.K):
                Qh = Q
                for hidden_num in range(0, len(self.personal_structure) - 1):
                    hidden = self.personal_structure[hidden_num]
                    xh = layers.fully_connected(Qh, num_outputs=hidden, activation_fn=None, weights_initializer=initializ)

                    if layer_norm: xh = layers.layer_norm(xh, center=True, scale=True)
                    Qh = self.lrelu(xh)
                    Qh = layers.fully_connected(Qh, num_outputs=self.personal_structure[-1], activation_fn=None,
                                                weights_initializer=initializ)

                    heads.append(Qh)

        if self.scope not in self.log:
            self.trainable_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/{}".format(tf.get_variable_scope().name, self.scope))

            self.log[self.scope] = self.trainable_vars

        return tf.stack(heads, axis=2)

    def lrelu(self, x, leak=0.05):
        return tf.maximum(x, x * leak)

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
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
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