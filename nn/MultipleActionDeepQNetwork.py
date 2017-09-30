# import necessary packages
import tensorflow as tf

from environments.Environment import Environment
from gym.spaces import Box
from gym.spaces import Discrete


class MultipleActionDeepQNetwork:
    """This represents a basic MultipleActionDeepQNetwork. It basically has the
    common known structure of DQN and in addition, tries to output
    the Q function for each value at once."""

    def __init__(self, env, structure, name_suffix):
        """This constructs a new DeepQNetwork. It therefore constructs
        the corresponding layers of the network.

        Args:
            env: The environment to use
            structure: The structure for the hidden layers, a list of integers.
            name_suffix: The suffix to append to the name
        """

        with tf.variable_scope("multiple_action_deep_q_network_{}".format(name_suffix)):

            assert isinstance(env, Environment)

            # obtain the spaces
            self.state_space = env.observation_space()
            self.action_space = env.action_space()

            assert isinstance(self.state_space, Box)
            assert isinstance(self.action_space, Discrete)

            # get tge action and state space sizes
            state_size = self.state_space.shape[0]
            action_size = self.action_space.n

            # Build the graph
            self.W, self.b, self.reg = self.__create_weights([state_size] + structure + [action_size])

    def copy_graph(self, network):
        """This copies the values from the passed network to this one.
        Logically the weight matrices need to have the same dimensions.

        Args:
            network: The other DeepQNetwork to take values from.
        """

        # for the ide only
        assert isinstance(network, MultipleActionDeepQNetwork)
        assert len(network.W) == len(self.W)

        # create operation list and fill accordingly
        ops = list()

        # iterate over the weights
        for i in range(len(self.W)):
            ops.append(self.W[i].assign(network.W[i]))

        for i in range(len(self.b)):
            ops.append(self.b[i].assign(network.b[i]))

        return tf.group(*ops)

    def eval_graph(self, states):
        """This creates the graph it therefore receives the input tensor and
        of course the weights for each layer.

        Args:
            states: The input to the graph. Usually a placeholder

        Returns:
            A fully constructed graph using the weights supplied.

        """

        assert states.shape[1] == self.state_space.shape[0]

        # set the start value
        Q = states

        # create network
        for i in range(len(self.W) - 1):
            x = Q @ self.W[i] + self.b[i]
            Q = tf.maximum(x, 0.01 * x)

        Q = Q @ self.W[-1]

        # pass back the relevant elements
        return Q

    def grid_graph(self, grid_dims):
        """This creates a grid evaluation graph for the network. This
        can be particular useful, when a heat plot of the q function
        should be generated."""

        # calc x and y range
        x_range = tf.range(self.state_space.low[0], self.state_space.high[0],
                           (self.state_space.high[0] - self.state_space.low[0] - 0.0001) / (grid_dims[0] - 1), dtype=tf.float32)

        y_range = tf.range(self.state_space.low[1], self.state_space.high[1],
                           (self.state_space.high[1] - self.state_space.low[1] - 0.0001) / (grid_dims[1] - 1), dtype=tf.float32)

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
        return grid_actions

    def learn_graph(self, learning_rate, Q, target_actions):
        """This creates the graph it therefore receives the input tensor and
        of course the weights for each layer.

        Args:
            learning_rate: The learning rate to use
            Q: The training graph itself
            target_actions: The actions which the network should output

        Returns:
            A fully constructed graph using the weights supplied.

        """

        # Define the error term
        hl = 1.0
        error = Q - target_actions
        squared_error = 0.5 * tf.square(error)
        abs_error = tf.abs(error)
        cond = abs_error < hl
        loss = tf.where(cond, squared_error, hl * (abs_error - 0.5 * hl))

        # Create a minimizer
        minimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(tf.reduce_mean(loss), var_list=self.W + self.b)

        # pass back the relevant elements
        return minimizer

    def __create_weights(self, structure):
        """This method can be used to create the weights for this
        model for the specified structure. These weights can then
        be used to create the remaining network.

        Args:
            structure: A list of integers describing the structure for the weights
        """

        # create two lists
        W = list()
        reg = tf.zeros([], dtype=tf.float32)

        in_size = structure[0]
        count = 0

        # create weights
        for layer in structure[1:]:

            # create the weights
            single_w = self.__init_single_weight_prelu([in_size, layer], ("W" + str(count)), 0.01)

            # add l2 loss
            reg += tf.nn.l2_loss(single_w)

            # create weights and update control variables
            W.append(single_w)
            in_size = layer
            count = count + 1

        # create biases
        b = list()
        count = 0

        for layer in structure[1:-1]:

            # create the bias
            single_b = self.__init_single_weight_with_init([1, layer], ("b" + str(count)), tf.zeros_initializer())
            reg += tf.nn.l2_loss(single_b)
            b.append(single_b)
            count = count + 1

        return W, b, reg

    def __init_single_weight_with_init(self, size, name, init):
        return tf.get_variable(name, shape=size, initializer=init)

    def __init_single_weight_prelu(self, size, name, value):

        var_term = tf.sqrt(2 / ((1 + tf.square(value)) * size[1]))
        rand_dist = tf.random_normal(size, 0, var_term, dtype=tf.float32)
        return tf.Variable(rand_dist, name=name)