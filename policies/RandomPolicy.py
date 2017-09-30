import tensorflow as tf

from policies.Policy import Policy
from policies.GreedyPolicy import GreedyPolicy
from gym.spaces import Discrete


class RandomPolicy(Policy):

    def choose_action(self, max):
        """Create the tree for epsilon greedy policies_nn selection.

        Args:
            max: The q function to use for evaluating.

        Returns: The tensorflow graph
        """

        return tf.random_uniform([1], maxval=max, dtype=tf.int32)