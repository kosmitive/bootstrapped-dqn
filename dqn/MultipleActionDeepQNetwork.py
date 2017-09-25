# import necessary packages
import tensorflow as tf


from gym.spaces import Box
from gym.spaces import Discrete


class MultipleActionDeepQNetwork:
    """This represents a basic MultipleActionDeepQNetwork. It basically has the
    common known structure of DQN and in addition, tries to output
    the Q function for each value at once."""

    def __init__(self, state_space, action_space, structure, name_suffix):
        """This constructs a new DeepQNetwork. It therefore constructs
        the corresponding layers of the network.

        Args:
            state_space: The state space.
            structure: The structure for the hidden layers, a list of integers.
            action_space: Which actions can be performed per step.
            name_suffix: The suffix to append to the name
        """

        assert isinstance(state_space, Box)
        assert isinstance(action_space, Discrete)

        with tf.variable_scope("multiple_action_deep_q_network_{}".format(name_suffix)):

            # This holds the state and action space
            self.state_space = state_space
            self.action_space = action_space

            # get tge action and state space sizes
            state_size = state_space.shape[0]
            action_size = action_space.n

            # Build the graph
            self.W, self.b = self.__create_weights([state_size] + structure + [action_size])

    def create_weight_copy_op(self, network):
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
            ops.append(self.b[i].assign(network.b[i]))

        return tf.group(*ops)

    def __create_weights(self, structure):
        """This method can be used to create the weights for this
        model for the specified structure. These weights can then
        be used to create the remaining network.

        Args:
            structure: A list of integers describing the structure for the weights
        """

        # get the first value of the structure
        in_size = structure[0]

        # create two lists
        W = list()
        b = list()
        count = 0

        # iterate over the remaining layers
        for layer in structure[1:]:

            # create weights and update control variables
            W.append(self.__init_single_weight([in_size, layer], name=("W" + str(count))))
            b.append(self.__init_single_weight([1, layer], name=("b" + str(count))))
            in_size = layer
            count = count + 1

        return W, b

    def create_eval_graph(self, states):
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
            Q = tf.nn.relu(Q @ self.W[i] + self.b[i])

        Q = Q @ self.W[-1] + self.b[-1]

        # pass back the relevant elements
        return Q

    def create_learn_graph(self, learning_rate, Q, target_actions):
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
        error = 0.5 * tf.square(Q - target_actions)

        # Create a minimizer
        minimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(error)

        # pass back the relevant elements
        return error, minimizer

    def __init_single_weight(self, size, name):
        return tf.get_variable(name, shape=size, initializer=tf.contrib.layers.xavier_initializer())
