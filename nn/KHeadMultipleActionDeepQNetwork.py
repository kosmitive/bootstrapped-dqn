# import necessary packages
import numpy as np
import tensorflow as tf

from src.util.Encoder import Encoder
from src.util.spaces.DiscreteSpace import DiscreteSpace


class KHeadMultipleActionDeepQNetwork:
    """This represents a basic MultipleActionDeepQNetwork. It basically has the
    common known structure of DQN and in addition, tries to output
    the Q function for each value at once."""

    def __init__(self, state_space, shared_structure, unique_structure, action_space, num_heads, dropout=1.0,
                 batch_normalization=False):
        """This constructor can be used to create a KHeadMultipleActionDeepQNetwork.
        It has K different heads, resulting in K different Q functions as well

        Args:
            state_space: The state space.
            shared_structure: The structure of the shared network
            unique_structure: The structure of the unique network.
            action_space: Which actions can be performed per step.
            num_heads: The number of different heads for this network.
        """

        assert isinstance(state_space, DiscreteSpace)
        assert isinstance(action_space, DiscreteSpace)

        # Define the seed for the model
        self.seed = 3
        self.K = num_heads

        # This holds the state and action space
        self.state_space = state_space
        self.action_space = action_space

        # get tge action and state space sizes
        state_size = state_space.get_log2_size()
        action_size = action_space.get_size()

        # define the variables and placeholders
        self.states = tf.placeholder(tf.float32, [state_size, None], name="states")

        # Build the shared network
        self.shared_W, self.shared_b = self._create_weights([state_size] + shared_structure)
        self.intermediate_layer = self._create_graph(self.states, self.shared_W, self.shared_b,
                                                     dropout, batch_normalization)

        # Build num_head times a head and combine them in a 3-dimensional tensor
        self.unique_weights = [self._create_weights(unique_structure + [action_size]) for _ in range(num_heads)]
        self.unique_outputs = [self._create_graph(self.intermediate_layer, W, b, dropout, batch_normalization)
                               for W, b in self.unique_weights]
        self.output_tensor = tf.stack(self.unique_outputs)

        # these are the target variables inside
        self.target_action_values = tf.placeholder(tf.float32, [num_heads, action_size, None], name="target_action_values")
        self.state_actions_count = tf.placeholder(tf.float32, [num_heads, action_size, None], name="state_actions_count")

        # Define the error on this
        squared_error = self.state_actions_count * tf.pow(self.output_tensor - self.target_action_values, 2)
        self.error = 0.5 * tf.reduce_sum(squared_error)

        # Create a minimizer
        self.minimizer = tf.train.AdagradOptimizer(0.01).minimize(self.error)

        # create new session
        for x in self.shared_W + self.shared_b:
            x.initializer.run()

        # create new session
        for W, b in self.unique_weights:
            for rw, rb in zip(W, b):
                rw.initializer.run()
                rb.initializer.run()

        tf.global_variables_initializer().run()

    def copy_weights(self, network):
        """This copies the values from the passed network to this one.
        Logically the weight matrices need to have the same dimensions.

        Args:
            network: The other DeepQNetwork to take values from.
        """

        # for the ide only
        assert isinstance(network, KHeadMultipleActionDeepQNetwork)
        assert len(network.shared_W) == len(self.shared_W)

        # iterate over the weights
        for i in range(len(self.shared_W)):
            self.shared_W[i].assign(network.shared_W[i]).eval()
            self.shared_b[i].assign(network.shared_b[i]).eval()

        # Copy the unique heads as well
        for i in range(len(self.unique_weights)):
            [W, b] = self.unique_weights[i]
            [oW, ob] = network.unique_weights[i]

            # assign the values
            for w1, w2 in zip(W, oW):
                w1.assign(w2).eval()

            for b1, b2 in zip(b, ob):
                b1.assign(b2).eval()

    def eval(self, states):
        """This method takes a vector of size [state_space_size, n], where
        n is the number of states, which should be evaluated at once.

        Args:
            states: The states to determine the Q-Value for

        Returns:
            The Q function, for each action for each supplied state.
        """

        # Encode as binary
        b_states = self.__states_to_binary_matrix(states)
        return self.output_tensor.eval(feed_dict={self.states: b_states})

    def __states_to_binary_matrix(self, states):
        """This method converts a list of states to a binary training tensor.

        Args:
            states: The states to convert to binary
        """

        # get the state size and length
        state_size = self.state_space.get_log2_size()
        state_len = len(states)

        # binary states
        result = np.empty((state_size, state_len), dtype=np.int32)

        # iterate over
        for k in range(state_len):
            result[:, k:k + 1] = Encoder.int_to_bin(states[k], state_size)

        return result

    def learn(self, states, actions, action_values, k):
        """This method takes the given state, action values and training mask.
        and consequently calculates the Q value for each action.

        Args:
            states: The states to check.
            actions: These are the actions used.
            action_values: The expected target values.
            k: Represents the masks which define how important a action value is for each head
        """

        # get needed fields
        tensor_states, tensor_counts, tensor_action_value \
            = self.__count_unique_states(states, actions, action_values, k)

        # create binary states
        binary_states = self.__states_to_binary_matrix(tensor_states)

        # run minimizer
        self.minimizer.run(feed_dict={self.states: binary_states,
                                      self.target_action_values: tensor_action_value,
                                      self.state_actions_count: tensor_counts})

    def __count_unique_states(self, states, actions, action_values, ks):
        """This method counts the unique states.

        Args:
            states: These are the unique states extracted from.
            actions: These are the actions used.
            action_values: These are the corresponding action values

        Returns:
            A tuple (states, count_tensor, action_value_tensor, drift)
        """

        action_size = self.action_space.get_size()

        # determine the number of distinct states, and
        # create a mapping form states to indices
        mapping = {}
        lst_states = list()
        N = 0
        for state in states:
            if state not in mapping:
                lst_states.append(state)
                mapping[state] = N
                N += 1

        # now we can build up the state tensor
        tensor_states = np.stack(lst_states)
        tensor_counts = np.zeros((self.K, action_size, N), dtype=np.float32)
        tensor_action_value = np.zeros((self.K, action_size, N), dtype=np.float32)

        # iterate over all states
        for (state, action, action_value, k) in zip(states, actions, action_values, ks):
            index = mapping[state]

            # count up in the count tensor
            tensor_counts[k, action, index] += 1
            tensor_action_value[k, action, index] =\
                ((tensor_counts[k, action, index] - 1) * tensor_action_value[k, action, index] + action_value) \
                / tensor_counts[k, action, index]

        # pass back the list with the tuples
        return tensor_states, tensor_counts, tensor_action_value

    def _create_weights(self, structure):
        """This method creates weights for the given structure.

        Args:
            structure: A list of integers describing the structure of the weights
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
            W.append(self._init_single_weight([layer, in_size], name=("W" + str(count))))
            b.append(self._init_single_weight([layer, 1], name=("b" + str(count))))
            in_size = layer
            count = count + 1

        return W, b

    def _create_graph(self, x, W, b, dropout, batch_normalization):
        """This creates the graph it therefore receives the input tensor and
        of course the weights for each layer.

        Args:
            x: The input to the graph. Usually a placeholder
            W: The list of weight matrices for each layer
            b: The list of biases for each layer

        Returns:
            A fully constructed graph using the weights supplied.

        """
        # set the start value
        tree = x

        # create network
        for i in range(len(W) - 1):

            # Use the passed dropout layer list, if available
            # to input the probabilities for the various drop
            # out layers
            if isinstance(dropout, list):
                if dropout[i] < 1.0:
                    tree = tf.nn.dropout(tree, dropout[i])

            elif dropout < 1.0:
                tree = tf.nn.dropout(tree, dropout)

            tree = self._lrelu(W[i] @ tree + b[i])

            if batch_normalization:
                mean = tf.zeros(b[i].get_shape())
                var = tf.ones(b[i].get_shape())
                tree = tf.nn.batch_normalization(tree, mean, var, None, None, 0.001)

        # add a final regression layer
        Q = W[-1] @ tree + b[-1]
        return Q

    @staticmethod
    def _lrelu(x):
        """This creates a new prelu unit using tensorflow. It takes
        two arguments as an input.

        Args:
            x: This represents the input to the layer.

        Returns:
            The tensorflow object, with that specific layer
        """
        return tf.maximum(x, 0.01 * x)

    def _init_single_weight(self, size, name):
        return tf.Variable(tf.random_normal(size, mean=0.0, stddev=0.05, seed=self.seed), name=name)
