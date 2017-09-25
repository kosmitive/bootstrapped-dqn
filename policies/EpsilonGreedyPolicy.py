import tensorflow as tf

from policies.Policy import Policy
from policies.GreedyPolicy import GreedyPolicy
from gym.spaces import Discrete


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

        assert isinstance(action_space, Discrete)
        self.max = action_space.n
        self.epsilon = epsilon

    def choose_action(self, q):
        """Create the tree for epsilon greedy policies_nn selection.

        Args:
            q: The q function to use for evaluating.

        Returns: The tensorflow graph
        """

        # get the number of states
        best_action = GreedyPolicy().choose_action(q)

        random_action = tf.random_uniform([1], maxval=self.max, dtype=tf.int32)
        random_decision_vector = tf.less(tf.random_uniform([1]), self.epsilon)

        # let the coin decide about the final actions
        final_action = tf.where(random_decision_vector, best_action, random_action)

        # pass back the actions and corresponding q values
        return final_action