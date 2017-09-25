# simply import the numpy package.
import tensorflow as tf

from dqn.ExperienceReplayMemory import ExperienceReplayMemory
from dqn.MultipleActionDeepQNetwork import MultipleActionDeepQNetwork
from gym.spaces import Box
from gym.spaces import Discrete
from policies.Policy import Policy


class DDQNAgent:
    """this is the agent playing the game and trying to maximize the reward."""

    def __init__(self, state_space, action_space, memory, structure, offset):
        """Constructs a BootstrappedDDQNAgent.

        Args:
            state_space: Give the discrete state space
            action_space: Give the discrete action space
            memory: The memory to use for retrieving information
            offset: Number of steps till the value should be copied
            structure: The structure to use for the neural networks
        """

        # check if the spaces are valid
        assert isinstance(state_space, Box)
        assert isinstance(action_space, Discrete)
        assert isinstance(memory, ExperienceReplayMemory)

        # set the internal debug variable
        self.memory = memory
        self.copy_offset = offset
        self.iteration = 0

        # Save the epsilon for the greedy policies_nn.
        self.state_space = state_space
        self.action_space = action_space

        # init necessary objects
        self.dqn = MultipleActionDeepQNetwork(state_space, action_space, structure, "selected")
        self.target_dqn = MultipleActionDeepQNetwork(state_space, action_space, structure, "target")

        # create iteration counter
        self.counter_init = tf.zeros([], dtype=tf.int32)
        self.iteration_counter = tf.Variable(self.counter_init)
        self.global_step = tf.Variable(0, trainable=False)

    def create_action_graph(self, current_observation, policy):
        """This method creates the action graph using the current observation. The policy
        has to be of type Policy.

        Args:
            current_observation: The current observation as a graph.
            policy: The policy to use.
        """

        assert isinstance(policy, Policy)

        # choose appropriate action
        eval_graph = self.dqn.create_eval_graph(tf.expand_dims(current_observation, 0))
        return policy.choose_action(eval_graph[0])

    def create_learn_graph(self, current_observation, next_observation, action, reward, discount, learning_rate, done):

        def false_fn():
            copy_op = self.target_dqn.create_weight_copy_op(self.dqn)
            with tf.control_dependencies([copy_op]):
                return tf.assign(self.iteration_counter, self.counter_init)

        def true_fn():
            return tf.assign_add(self.iteration_counter, 1)

        # create the copy operation
        condition = tf.less(self.iteration_counter, self.copy_offset)
        copy_op = tf.cond(condition, true_fn, false_fn)

        with tf.control_dependencies([copy_op]):

            # retrieve all samples
            cs, ns, ac, re, ds = self.memory.create_insert_sample_op(current_observation, next_observation, action, reward, done)

            # get both q functions
            q = self.dqn.create_eval_graph(cs)
            dqn_nq = self.dqn.create_eval_graph(ns)
            tdqn_nq = self.target_dqn.create_eval_graph(ns)
            dqn_ba = tf.cast(tf.argmax(dqn_nq, axis=1), tf.int32)

            rng = tf.range(0, tf.shape(dqn_ba)[0], dtype=tf.int32)
            stacked_indices = tf.stack((rng, dqn_ba), axis=1)
            tdqn_nq = tf.gather_nd(tdqn_nq, stacked_indices)

            stacked_indices2 = tf.stack((rng, ac), axis=1)
            q_nq = tf.gather_nd(q, stacked_indices2)

            # calculate targets
            targets = re + tf.where(ds, tf.zeros_like(tdqn_nq), discount * tdqn_nq)
            learn = self.dqn.create_learn_graph(learning_rate, q_nq, targets)

            return learn
