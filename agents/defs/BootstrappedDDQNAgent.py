# MIT License
#
# Copyright (c) 2017 Markus Semmler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf

from agents.GeneralAgent import GeneralAgent
from environments.open_ai_envs.ContinualStateEnv import ContinualStateEnv
from memory.Memory import Memory
from policies.Policy import Policy


class BootstrappedDDQNAgent(GeneralAgent):
    """This represents a bootstrapped DQN agent. It basically approximates a
    posterior over Q."""

    def __init__(self, env, shared_structure, head_structure, num_heads, config):
        """Constructs a BootstrappedDDQNAgent.

        Args:
            env: The environment
            shared_structure: Define the number of layers
            config:
                - offset: Number of steps till the value should be copied
        """

        assert isinstance(env, ContinualStateEnv)
        super().__init__("bootstrapped_dqn", env, config)

        # set the internal debug variable
        self.memory = None
        self.policy = None
        self.copy_offset = config['target_offset']
        self.iteration = 0

        # save these numbers
        self.discount = config['discount']
        self.learning_rate = config['learning_rate']
        self.heads = num_heads

        # init necessary objects
        self.network = MultipleHeadDeepNetwork([env.observation_space().dim()] + shared_structure,
                                               head_structure + [env.action_space().dim()], num_heads, {"layer-norm" : True})

        # create iteration counter
        self.counter_init = tf.zeros([], dtype=tf.int32)
        self.iteration_counter = tf.Variable(0, trainable=False)
        self.global_step = tf.Variable(0, trainable=False)
        self.current_head = tf.Variable(0, trainable=False, dtype=tf.int64)

    def sample_head_graph(self):
        random_head = tf.random_uniform([], 0, self.heads, dtype=tf.int64)
        return tf.assign(self.current_head, random_head)

    def copy_graph(self):
        self.network.switch('bootstrappeddqn')
        return self.network.copy_graph('bootstrappedtarget')

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
        self.network.switch('bootstrappeddqn')
        eval_graph = self.network.eval_graph(tf.expand_dims(current_observation, 0))
        return self.policy.choose_action(eval_graph[:, :, self.current_head])

    def observe_graph(self, current_observation, next_observation, action, reward, done):
        assert self.memory is not None

        # retrieve all samples
        current_states, next_states, actions, rewards, dones = self.memory.store_and_sample_graph(current_observation, next_observation, action, reward, done)

        # get both q functions
        current_q = self.network.eval_graph(current_states, train=True)
        next_q = self.network.eval_graph(next_states, train=True)

        target_next_q = self.network.eval_graph(next_states)
        best_next_actions = tf.reshape(tf.cast(tf.argmax(next_q, axis=1), tf.int32), [self.memory.sample_size * self.heads])

        # build initial ranges
        sample_rng = tf.range(0, self.memory.sample_size, dtype=tf.int32)
        head_rng = tf.range(0, self.heads, dtype=tf.int32)

        # mod the ranges
        sample_rng = tfh.duplicate_each_element(sample_rng, self.heads)
        head_range = tf.tile(tf.expand_dims(head_rng, 1), [self.memory.sample_size, 1])
        indices_best_actions = tf.stack((sample_rng, head_range, best_next_actions), axis=1)
        target_best_q_values = tf.gather_nd(target_next_q, indices_best_actions)

        duplicated_actions = tfh.duplicate_each_element(actions, self.heads)
        indices_actions = tf.stack((sample_rng, head_range, duplicated_actions), axis=1)
        exec_q_values = tf.gather_nd(current_q, indices_actions)

        # now duplicate the rewards and dones
        dupl_rewards = tfh.duplicate_each_element(rewards, self.heads)
        dupl_dones = tfh.duplicate_each_element(dones, self.heads)

        # calculate targets
        targets = dupl_rewards + self.discount * tf.cast(1 - dupl_dones, tf.float32) * target_best_q_values
        learn = self.network.learn_graph(exec_q_values, tf.stop_gradient(targets), 'dqn', self.learning_rate, self.global_step)

        # execute only if in learning mode
        return learn
