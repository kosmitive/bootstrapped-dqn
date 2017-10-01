import tensorflow as tf

from policies.Policy import Policy


class GreedyPolicy(Policy):
    """This policy selects the action greedily."""

    def __init__(self, N):
        """Inits a policy, therefore one has to supply a
        value for the times of dimensions"""

        super().__init__(N)

    def choose_action(self, q):
        """Creates a graph, where the best action is selected.

        Args:
            q: The q function to use for evaluating.
        """

        # get the number of states
        return tf.cast(tf.argmax(q, axis=1), tf.int32)
