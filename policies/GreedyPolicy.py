import tensorflow as tf

from policies.Policy import Policy


class GreedyPolicy(Policy):
    """This policy selects the action greedily."""

    def choose_action(self, q):
        """Creates a graph, where the best action is selected.

        Args:
            q: The q function to use for evaluating.
        """

        # get the number of states
        return tf.expand_dims(tf.cast(tf.argmax(q), tf.int32), 0)
