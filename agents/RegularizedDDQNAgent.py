# simply import the numpy package.
import tensorflow as tf

import extensions.tensorflowHelpers as tfh
from memory.Memory import Memory
from nn.RegularizedDeepNetwork import RegularizedDeepNetwork
from policies.Policy import Policy
from environments.Environment import Environment
from agents.Agent import Agent

class RegularizedDDQNAgent(Agent):
    """this is the agent playing the game and trying to maximize the reward."""

    def __init__(self, env, K, H, config):
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
        self.K = K
        self.H = H

        # save these numbers
        self.discount = config['discount']
        self.learning_rate = config['learning_rate']

        # init necessary objects
        net_struct = [env.observation_space().dim()] + config['structure'] + [env.action_space().dim()]
        self.network = RegularizedDeepNetwork(net_struct, {"layer-norm" : True})

        # create iteration counter
        self.counter_init = tf.zeros([], dtype=tf.int32)
        self.iteration_counter = tf.Variable(0, trainable=False)
        self.global_step = tf.Variable(0, trainable=False)

        use_dropout = 'regularization' in config and config['regularization'] == 'dropout'
        use_zoneout = 'regularization' in config and config['regularization'] == 'zoneout'
        use_shakeout = 'regularization' in config and config['regularization'] == 'shakeout'

        if use_dropout or use_zoneout or use_shakeout:

            # create selection masks
            self.as_masks = list()
            self.sample_as_masks = list()
            for k in range(K):
                mask_value = config['reg_rate_end'] - tfh.linear_decay(config['reg_steps_red'], config['reg_rate_end'] - config['reg_rate_begin'], 0.0, self.global_step)
                masks, sample = self.network.create_mask_graph(mask_value)
                self.as_masks.append(masks)
                self.sample_as_masks.append(sample)

            # create selection masks
            self.lr_masks = list()
            self.sample_lr_masks = list()
            for k in range(K):
                mask_value = config['reg_rate_end'] - tfh.linear_decay(config['reg_steps_red'],
                                                                       config['reg_rate_end'] - config[
                                                                           'reg_rate_begin'], 0.0, self.global_step)
                masks, sample = self.network.create_mask_graph(mask_value)
                self.lr_masks.append(masks)
                self.sample_lr_masks.append(sample)

            self.sample_as_masks = tf.group(*self.sample_as_masks)
            self.sample_lr_masks = tf.group(*self.sample_lr_masks)
        # create feed dict for zoneout and dropout
        self.settings = None
        if use_dropout:
            self.settings = 'dropout_masks'
        elif use_zoneout:
            self.settings = 'zoneout_masks'
        elif use_shakeout:
            self.settings = 'shakeout_masks'

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
        evals = list()
        for k in range(0, self.K):
            mask_dict = {self.settings: self.as_masks[k]}
            evals.append(self.network.eval_graph(tf.expand_dims(current_observation, 0), **mask_dict))

        # build the mean
        mean_graph = evals[0]
        for k in range(1, self.K):
            mean_graph += evals[0]
        mean_graph /= self.K

        var_graph = tf.pow(evals[0] - mean_graph, 2)
        for k in range(1, self.K):
            var_graph += tf.pow(evals[k] - mean_graph, 2)
        var_graph /= self.K

        action = self.policy.choose_action(mean_graph + 0.005 * var_graph)
        # calculate info gain
        T = 0.9
        boltz_list = list()
        for k in range(self.K):
            softmax = tf.nn.softmax(evals[k] / T)
            boltz_list.append(softmax)

        # build the average
        avg = boltz_list[0]
        for k in range(1, self.K):
            avg += boltz_list[k]

        avg /= self.K

        kl_term = tf.constant(0.0)
        for k in range(self.K):
            kl_term += self.kl_divergence(tf.squeeze(boltz_list[k]), tf.squeeze(avg))

        return action, kl_term

    def kl_divergence(self, p, q):
        return tf.reduce_sum(p * tf.log(p / q))

    def observe_graph(self, current_observation, next_observation, action, reward, done):
        assert self.memory is not None

        # retrieve all samples
        current_states, next_states, actions, rewards, dones = self.memory.store_and_sample_graph(current_observation, next_observation, action, reward, done)
        all_exec_q_values = list()
        all_targets = list()

        with tf.control_dependencies([self.sample_lr_masks]):

            # over all  masks
            for h in range(self.H):

                mask_dict = {self.settings: self.lr_masks[h]}

                # get both q functions
                self.network.switch('dqn')
                current_q = self.network.eval_graph(current_states, train=True, **mask_dict)
                next_q = self.network.eval_graph(next_states, train=True, **mask_dict)

                self.network.switch('target')
                target_next_q = self.network.eval_graph(next_states, **mask_dict)
                best_next_actions = tf.cast(tf.argmax(next_q, axis=1), tf.int32)

                sample_rng = tf.range(0, tf.size(actions), dtype=tf.int32)
                indices_best_actions = tf.stack((sample_rng, best_next_actions), axis=1)
                target_best_q_values = tf.gather_nd(target_next_q, indices_best_actions)

                indices_actions = tf.stack((sample_rng, actions), axis=1)
                exec_q_values = tf.gather_nd(current_q, indices_actions)

                # calculate targets
                targets = rewards + self.discount * tf.cast(1 - dones, tf.float32) * target_best_q_values
                all_exec_q_values.append(exec_q_values)
                all_targets.append(targets)

            all_exec_q_values = tf.stack(all_exec_q_values)
            all_targets = tf.stack(all_targets)

            self.network.switch('dqn')
            learn = self.network.learn_graph(self.learning_rate, all_exec_q_values, tf.stop_gradient(all_targets), self.global_step)

            # execute only if in learning mode
            return learn
