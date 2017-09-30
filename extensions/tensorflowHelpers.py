import tensorflow as tf


def if_execute(condition, action):
    """Execute the action if the condition holds.

    Args:
        condition: The condition which should be evaluated to true.
        action: The action to execute.

    """

    def fn():
        with tf.control_dependencies([action]):
            return tf.no_op()

    return tf.cond(condition, lambda: fn(), lambda: tf.no_op())


def clocked_executor(counter, max, action):
    """This represents a clocked executor. It can be used to count up and
    execute the action each time the counter reaches max."""

    incounter = tf.assign(counter, tf.mod(counter + 1, max))
    with tf.control_dependencies([incounter]):
        return if_execute(tf.equal(counter, 1), action)


def exp_decay(exp_max, exp_min, decay_lambda, step):

    # create linear decay learning rate
    return exp_min \
          + (exp_max - exp_min) \
            * tf.exp(-decay_lambda * tf.cast(step, tf.float32))