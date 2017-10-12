import tensorflow as tf
import tensorflow.contrib.layers as layers

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

def linear_decay(schedule_timesteps, initial_p, final_p, step):

    # create linear decay learning rate
    fraction = tf.minimum(step / schedule_timesteps, 1.0)
    return initial_p + fraction * (final_p - initial_p)

def duplicate_each_element(vector: tf.Tensor, repeat: int):
    """This method takes a vector and duplicates each element the number of times supplied."""

    height = tf.shape(vector)[0]
    exp_vector = tf.expand_dims(vector, 1)
    tiled_states = tf.tile(exp_vector, [1, repeat])
    mod_vector = tf.reshape(tiled_states, [repeat * height])
    return mod_vector


# --- Neural Network Functions ---


def squared_loss(x):
    return 0.5 * tf.square(x)


def huber_loss(delta, x):

    # Define the error term
    squared_error = squared_loss(x)
    abs_error = tf.abs(x)
    cond = tf.less(abs_error, delta)
    return tf.where(cond, squared_error, delta * (abs_error - 0.5 * delta))


def leakyrelu(leak=0.05):
    """This method creates a LReLU activation function and a correct initializer.

    Args:
        leak: The derivative for the negative part.

    Returns:
        Tuple (activation_fn, initializer_fn)
    """

    def general_xavier_initializer(seed=None):
        return layers.variance_scaling_initializer(
            factor=2.0/(1+leak**2),
            mode='FAN_IN',
            seed=seed,
            uniform=False,
            dtype=tf.float32)

    return lambda x: tf.maximum(x, x * leak), general_xavier_initializer


def identity():
    """This method creates a identity activation function and a correct initializer.

    Returns:
        Tuple (activation_fn, initializer)
    """
    return lambda x: tf.identity(x), layers.xavier_initializer


def eval_fc_layer(Q, shape, activation=identity(), layer_norm=False, mask=None, mask_type=None, seed=None):

    if mask is not None and mask_type is None:
        raise ValueError("You have to specify either 'dropout', 'zoneout' or 'shakeout' as the mask_type")

    if mask is not None and mask_type is not None:

        # expand mask
        exp_mask = tf.expand_dims(mask, 0)

        # define some control variables
        use_dropout = mask_type is 'dropout'
        use_zoneout = mask_type is 'zoneout'
        use_shakeout = mask_type is 'shakeout'

    else:
        use_dropout = False
        use_zoneout = False
        use_shakeout = False

    # get activation and weights initializer
    activation_fn, init = activation
    W = tf.get_variable("W", shape=shape, initializer=init(seed=seed))
    b = tf.get_variable("b", shape=shape[1], initializer=tf.zeros_initializer())

    # first of all shakeout has to be applied
    if use_shakeout:
        q = 0.3
        W = W @ exp_mask + q * tf.sign(W) @ (exp_mask - 1)

    # create the network
    u = Q @ W + b

    # apply regularization
    if layer_norm: u = layers.layer_norm(u, center=True, scale=True)
    if use_dropout: u = u * exp_mask
    v = activation_fn(u)

    # last but not least apply zoneout
    return exp_mask * v + (1 - exp_mask) * Q if use_zoneout else v
