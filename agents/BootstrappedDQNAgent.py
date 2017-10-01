# simply import the numpy package.
import tensorflow as tf
from gym.spaces import Box
from gym.spaces import Discrete

import extensions.tensorflowHelpers as tfh
from memory.Memory import Memory
from nn.MultipleActionDeepQNetwork import MultipleActionDeepQNetwork
from policies.Policy import Policy
from environments.Environment import Environment
from agents.Agent import Agent

class BootstrappedDQNAgent(Agent):
    """this is the agent playing the game and trying to maximize the reward."""

    def __init__(self, env, structure, offset, discount, learning_rate):
        """Constructs a BootstrappedDDQNAgent.

        Args:
            state_space: Give the discrete state space
            action_space: Give the discrete action space
            offset: Number of steps till the value should be copied
            structure: The structure to use for the neural networks
        """

        assert isinstance(env, Environment)

        # obtain the spaces
        self.state_space = env.observation_space()
        self.action_space = env.action_space()

        # check if the spaces are valid
        assert isinstance(self.state_space, Box)
        assert isinstance(self.action_space, Discrete)

        # set the internal debug variable
        self.memory = None
        self.policy = None
        self.copy_offset = offset
        self.iteration = 0

        # save these numbers
        self.discount = discount
        self.learning_rate = learning_rate

        # init necessary objects
        self.dqn = MultipleActionDeepQNetwork(env, structure, "selected")
        self.target_dqn = MultipleActionDeepQNetwork(env, structure, "target")

        # create iteration counter
        self.counter_init = tf.zeros([], dtype=tf.int32)
        self.iteration_counter = tf.Variable(0, trainable=False)
        self.global_step = tf.Variable(0, trainable=False)

    def register_memory(self, memory):
        assert isinstance(memory, Memory)
        self.memory = memory

    def register_policy(self, policy):
        assert isinstance(policy, Policy)
        self.policy = policy

    def action_graph(self, current_observation):
        """This method creates the action graph using the current observation. The policy
        has to be of type Policy.

        Args:
            current_observation: The current observation
        """
        assert self.policy is not None

        # choose appropriate action
        eval_graph = self.dqn.eval_graph(tf.expand_dims(current_observation, 0))
        return self.policy.choose_action(eval_graph[0])

    def observe_graph(self, current_observation, next_observation, action, reward, done):
        assert self.memory is not None

        # create a clocked executor
        copy_weights_op = tfh.clocked_executor(
            self.iteration_counter,
            self.copy_offset,
            self.target_dqn.copy_graph(self.dqn)
        )

        with tf.control_dependencies([copy_weights_op]):

            # retrieve all samples
            current_states, next_states, actions, rewards, dones = self.memory.store_and_sample_graph(current_observation, next_observation, action, reward, done)

            # get both q functions
            current_q = self.dqn.eval_graph(current_states)
            next_q = self.dqn.eval_graph(next_states)
            target_next_q = self.target_dqn.eval_graph(next_states)
            best_next_actions = tf.cast(tf.argmax(next_q, axis=1), tf.int32)

            sample_rng = tf.range(0, tf.size(actions), dtype=tf.int32)
            indices_best_actions = tf.stack((sample_rng, best_next_actions), axis=1)
            target_best_q_values = tf.gather_nd(target_next_q, indices_best_actions)

            indices_actions = tf.stack((sample_rng, actions), axis=1)
            exec_q_values = tf.gather_nd(current_q, indices_actions)

            # calculate targets
            targets = rewards + tf.where(dones, tf.zeros_like(rewards), self.discount * target_best_q_values)
            learn = self.dqn.learn_graph(self.learning_rate, exec_q_values, targets)

            # execute only if in learning mode
            return learn
