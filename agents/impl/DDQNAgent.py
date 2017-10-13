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

# simply import the numpy package.
import tensorflow as tf
from nn.FeedForwardNetwork import FeedForwardNetwork
from agents.RLAgent import RLAgent


class DDQNAgent(RLAgent):
    """this is the agent playing the game and trying to maximize the reward."""

    def __init__(self, env, structure, config):
        """Constructs a BootstrappedDDQNAgent.

        Args:
            env: The environment
            structure: Define the number of layers
            config:
                - offset: Number of steps till the value should be copied
        """
        super().__init__("DDQN", env, config)

        # set the internal debug variable
        self.memory = None
        self.policy = None
        self.copy_offset = config['target_offset']
        self.iteration = 0

        # save these numbers
        self.discount = config['discount']
        self.learning_rate = config['learning_rate']

        # init necessary objects
        net_struct = [env.state_space.D] + structure + [env.action_space.D]
        self.network = FeedForwardNetwork(net_struct, {"layer-norm" : True})

        # create iteration counter
        self.counter_init = tf.zeros([], dtype=tf.int32)
        self.iteration_counter = tf.Variable(0, trainable=False)
        self.global_step = tf.Variable(0, trainable=False)

    def action_eval_graph(self, current_observation):
        """This method creates the action graph using the current observation. The policy
        has to be of type Policy.

        Args:
            current_observation: The current observation
        """

        # choose appropriate action
        return self.network.eval_graph(tf.expand_dims(current_observation, 0), 'dqn')

    def learn_graph(self, current_states, next_states, actions, rewards, dones):
        """This graph takes N experience tuple  uses these to create a minimizer.

        Args:
            current_states: The current observations
            next_states: The next observations
            actions: The chosen actions
            rewards: The reward received for performing the actions
            dones: Whether the episode is finished or not.

        Returns:
            A operation which learns from the samples.
        """

        # get both q functions
        current_q = self.network.eval_graph(current_states, train=True, scope='dqn')
        next_q = self.network.eval_graph(next_states, train=True, scope='dqn')

        target_next_q = self.network.eval_graph(next_states, scope='target')
        best_next_actions = tf.cast(tf.argmax(next_q, axis=1), tf.int32)

        sample_rng = tf.range(0, tf.size(actions), dtype=tf.int32)
        indices_best_actions = tf.stack((sample_rng, best_next_actions), axis=1)
        target_best_q_values = tf.gather_nd(target_next_q, indices_best_actions)

        indices_actions = tf.stack((sample_rng, actions), axis=1)
        exec_q_values = tf.gather_nd(current_q, indices_actions)

        # calculate targets
        targets = rewards + self.discount * tf.cast(1 - dones, tf.float32) * target_best_q_values
        learn = self.network.learn_graph(exec_q_values, tf.stop_gradient(targets), 'dqn', self.learning_rate, self.global_step)

        # execute only if in learning mode
        return learn
