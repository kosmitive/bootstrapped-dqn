# simply import the numpy package.
import tensorflow as tf

import extensions.tensorflowHelpers as tfh
from memory.Memory import Memory
from nn.MultipleActionDeepQNetwork import MultipleActionDeepQNetwork
from policies.Policy import Policy
from environments.Environment import Environment
from agents.Agent import Agent
from spaces.ContinuousSpace import ContinuousSpace
from spaces.DiscreteSpace import DiscreteSpace

class DDQNAgent(Agent):
    """this is the agent playing the game and trying to maximize the reward."""

    def __init__(self, N, env, structure, config):
        """Constructs a BootstrappedDDQNAgent.

        Args:
            N: The number of independent agents
            env: The environment
            structure: Define the number of layers
            config:
                - offset: Number of steps till the value should be copied
        """

        assert isinstance(env, Environment)

        # obtain the spaces
        self.state_space = env.observation_space()
        self.action_space = env.action_space()

        # set the internal debug variable
        self.memory = None
        self.policy = None
        self.copy_offset = config['target_offset']
        self.iteration = 0
        self.N = N

        # save these numbers
        self.discount = config['discount']
        self.learning_rate = config['learning_rate']

        # init necessary objects
        self.dqn = MultipleActionDeepQNetwork(N, env, structure, "selected")
        self.target_dqn = MultipleActionDeepQNetwork(N, env, structure, "target")

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

    def action_graph(self, current_observations):
        """This method creates the action graph using the current observation. The policy
        has to be of type Policy.

        Args:
            current_observation: The current observation
        """
        assert self.policy is not None

        # choose appropriate action
        eval_graph = self.dqn.eval_graph(tf.expand_dims(current_observations, axis=1))
        return self.policy.choose_action(eval_graph[:, 0, :])

    def observe_graph(self, current_observation, next_observation, action, reward, done):

        # create a clocked executor
        copy_weights_op = tfh.clocked_executor(
            self.iteration_counter,
            self.copy_offset,
            self.target_dqn.copy_graph(self.dqn)
        )

        # retrieve all samples
        current_states, next_states, actions, rewards, dones = self.memory.store_and_sample_graph(current_observation, next_observation, action, reward, done)

        # get both q functions
        current_q = self.dqn.eval_graph(current_states)
        next_q = self.dqn.eval_graph(next_states)
        target_next_q = self.target_dqn.eval_graph(next_states)
        best_next_actions = tf.cast(tf.argmax(next_q, axis=2), tf.int32)

        # create indices
        A = self.memory.sample_size
        batch_rng = tf.expand_dims(tf.range(0, A, dtype=tf.int32), 0)
        model_rng = tf.expand_dims(tf.range(0, self.N, dtype=tf.int32), 1)

        # scale to matrices
        til_batch_rng = tf.tile(batch_rng, [self.N, 1])
        til_model_rng = tf.tile(model_rng, [1, A])

        # create index matrix
        ind_best_actions_matrix = tf.stack([til_model_rng, til_batch_rng, best_next_actions], axis=2)
        indices_best_actions = tf.reshape(ind_best_actions_matrix, [self.N * A, 3])
        target_best_q_values = tf.reshape(tf.gather_nd(target_next_q, indices_best_actions), [self.N, A])

        ind_actions_matrix = tf.stack([til_model_rng, til_batch_rng, actions], axis=2)
        indices_actions = tf.reshape(ind_actions_matrix, [self.N * A, 3])
        exec_q_values = tf.reshape(tf.gather_nd(current_q, indices_actions), [self.N, A])

        # calculate targets
        targets = rewards + self.discount * tf.cast(1 - dones, tf.float32) * target_best_q_values
        learn = self.dqn.learn_graph(self.learning_rate, exec_q_values, targets, self.global_step)

        # execute only if in learning mode
        return learn
