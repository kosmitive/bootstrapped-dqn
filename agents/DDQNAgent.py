# simply import the numpy package.
import tensorflow as tf

import extensions.tensorflowHelpers as tfh
from memory.Memory import Memory
from nn.DeepNetwork import DeepNetwork
from policies.Policy import Policy
from environments.Environment import Environment
from agents.Agent import Agent

class DDQNAgent(Agent):
    """this is the agent playing the game and trying to maximize the reward."""

    def __init__(self, env, config):
        """Constructs a BootstrappedDDQNAgent.

        Args:
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

        # save these numbers
        self.discount = config['discount']
        self.learning_rate = config['learning_rate']

        # init necessary objects
        net_struct = [env.observation_space().dim()] + config['structure'] + [env.action_space().dim()]
        self.network = DeepNetwork(net_struct, {"layer-norm" : True})

        # create iteration counter
        self.counter_init = tf.zeros([], dtype=tf.int32)
        self.iteration_counter = tf.Variable(0, trainable=False)
        self.global_step = tf.Variable(0, trainable=False)

        use_dropout = 'regularization' in config and config['regularization'] == 'dropout'
        use_zoneout = 'regularization' in config and config['regularization'] == 'zoneout'
        use_shakeout = 'regularization' in config and config['regularization'] == 'shakeout'

        if use_dropout or use_zoneout or use_shakeout:
            self.mask_value = config['reg_rate_end'] - tfh.linear_decay(config['reg_steps_red'], config['reg_rate_end'] - config['reg_rate_begin'], 0.0, self.global_step)
            self.sample = self.network.create_mask_graph(self.mask_value)
            self.masks = self.network.get_mask_graph()

        # create feed dict for zoneout and dropout
        self.settings = {}
        if use_dropout:
            self.settings['dropout_masks'] = self.masks
        elif use_zoneout:
            self.settings['zoneout_masks'] = self.masks
        elif use_shakeout:
            self.settings['shakeout_masks'] = self.masks

    def copy_graph(self):
        self.network.switch('dqn')
        return self.network.copy_graph('target')

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
        self.network.switch('dqn')
        eval_graph = self.network.eval_graph(tf.expand_dims(current_observation, 0), **self.settings)
        return self.policy.choose_action(eval_graph)

    def observe_graph(self, current_observation, next_observation, action, reward, done):
        assert self.memory is not None

        # retrieve all samples
        current_states, next_states, actions, rewards, dones = self.memory.store_and_sample_graph(current_observation, next_observation, action, reward, done)

        # get both q functions
        self.network.switch('dqn')
        current_q = self.network.eval_graph(current_states, train=True, **self.settings)
        next_q = self.network.eval_graph(next_states, train=True)

        self.network.switch('target')
        target_next_q = self.network.eval_graph(next_states, **self.settings)
        best_next_actions = tf.cast(tf.argmax(next_q, axis=1), tf.int32)

        sample_rng = tf.range(0, tf.size(actions), dtype=tf.int32)
        indices_best_actions = tf.stack((sample_rng, best_next_actions), axis=1)
        target_best_q_values = tf.gather_nd(target_next_q, indices_best_actions)

        indices_actions = tf.stack((sample_rng, actions), axis=1)
        exec_q_values = tf.gather_nd(current_q, indices_actions)

        # calculate targets
        targets = rewards + self.discount * tf.cast(1 - dones, tf.float32) * target_best_q_values
        self.network.switch('dqn')
        learn = self.network.learn_graph(self.learning_rate, exec_q_values, tf.stop_gradient(targets), self.global_step)

        # execute only if in learning mode
        return learn
