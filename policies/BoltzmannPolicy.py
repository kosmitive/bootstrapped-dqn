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

from policies.Policy import Policy


class BoltzmannPolicy(Policy):
    """Represents a simple boltzmann policies_nn."""

    def choose_action(self, q, config):
        """This method of a policies_nn basically gets a Q function
        and has to return the action to take now. Here you can
        specify behaviour like taking the best action, or sometimes
        different actions to them.

        Args:
            q: The q function to use for evaluating.

        Returns: The index of the action that should be taken
        """

        # get the number of states
        soft_max = tf.nn.softmax(q / config['temperature'])

        # create the categorical
        dist = tf.distributions.Categorical(probs=soft_max)
        actions = tf.cast(dist.sample(), tf.int64)
        actions = actions

        # pass back the actions and corresponding q values
        model_range = tf.range(0, config['num_models'], 1, dtype=tf.int64)
        indices = tf.stack([model_range, actions], axis=1)
        q_values = tf.gather_nd(q, indices)
        return actions, q_values
