import tensorflow as tf

from policies.Policy import Policy
from policies.GreedyPolicy import GreedyPolicy
from policies.RandomPolicy import RandomPolicy
from gym.spaces import Discrete


class EpsilonGreedyPolicy(Policy):
    """This represents a simple epsilon greedy policies_nn. What it basically does,
    is, that it selects the best action in 1 - epsilon of the cases and in
    epsilon cases it wil basically select a random one.
    """

    def __init__(self, N, action_space, epsilon):
        """Initializes a new EpsilonGreedyPolicy.

        Args:
            action_space: The action space to use
            epsilon: The probability of choosing a random action.
        """

        super().__init__(N)
        self.action_space = action_space
        self.epsilon = epsilon

    def choose_action(self, q):
        """Create the tree for epsilon greedy policies_nn selection.

        Args:
            q: The q function to use for evaluating.

        Returns: The tensorflow graph
        """

        # get the number of states
        best_actions = GreedyPolicy(self.N).choose_action(q)
        random_actions = RandomPolicy(self.N, self.action_space.dim()).choose_action(q)

        # combine both
        random_decision_vector = tf.less(tf.random_uniform([self.N]), self.epsilon)
        final_actions = tf.where(random_decision_vector, best_actions, random_actions)

        # pass back the actions and corresponding q values
        return final_actions