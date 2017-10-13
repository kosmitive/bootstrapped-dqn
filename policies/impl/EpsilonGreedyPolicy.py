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
from policies.impl.RandomPolicy import RandomPolicy
from policies.impl.GreedyPolicy import GreedyPolicy


class EpsilonGreedyPolicy(Policy):
    """This represents a simple epsilon greedy policies_nn. What it basically does,
    is, that it selects the best action in 1 - epsilon of the cases and in
    epsilon cases it wil basically select a random one.
    """

    def __init__(self, action_space, epsilon):
        """Initializes a new EpsilonGreedyPolicy.

        Args:
            action_space: The action space to use
            epsilon: The probability of choosing a random action.
        """

        self.max_action = action_space.D
        self.epsilon = epsilon

    def choose_action(self, q):
        """Create the tree for epsilon greedy policies_nn selection.

        Args:
            q: The q function to use for evaluating.

        Returns: The tensorflow graph
        """

        # get the number of states
        best_actions = GreedyPolicy().choose_action(q)
        random_actions = RandomPolicy(self.max_action).choose_action(q)

        # combine both
        random_decision_vector = tf.less(tf.random_uniform([1], dtype=tf.float64), self.epsilon)
        final_actions = tf.where(random_decision_vector, random_actions, best_actions)

        # pass back the actions and corresponding q values
        return final_actions
