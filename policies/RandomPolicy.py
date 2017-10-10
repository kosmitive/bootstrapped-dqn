import tensorflow as tf

from policies.Policy import Policy
from policies.GreedyPolicy import GreedyPolicy
from gym.spaces import Discrete


class RandomPolicy(Policy):

    def __init__(self, max_num):
        """Inits a policy, therefore one has to supply a
        value for the times of dimensions"""

        self.max = max_num

    def choose_action(self, q):
        """Create the tree for epsilon greedy policies_nn selection.

        Args:
            q: The q function to use for evaluating.

        Returns: The tensorflow graph
        """

        return tf.random_uniform([1], maxval=self.max, dtype=tf.int32)