import tensorflow as tf

from gym.spaces import Box
from gym.spaces import Discrete
from environments.Environment import Environment
from memory.Memory import Memory


class ExperienceReplayMemory(Memory):
    """This class represent a basic replay memory. It can basically
    store the last N tuples.
    """

    def __init__(self, size, sample_size, env):
        """Constructs a new ReplayMemory.

        Args:
            size: The size of the replay memory.
            sample_size: How much samples should be derived.
        """

        # check if this is a valid size
        assert size > 0
        assert sample_size <= size
        assert isinstance(env, Environment)

        # obtain the spaces
        state_space = env.observation_space()
        action_space = env.action_space()

        # check if the spaces are valid
        assert isinstance(state_space, Box)
        assert isinstance(action_space, Discrete)

        # create a new variable scope
        with tf.variable_scope("replay_memory"):

            # init action and reward value
            self.actions = tf.Variable(tf.zeros([size], dtype=tf.int32))
            self.rewards = tf.Variable(tf.zeros([size], dtype=tf.float32))
            self.dones = tf.Variable(tf.zeros([size], dtype=tf.bool))

            # init state space size
            state_size = state_space.shape[0]
            state_init = tf.zeros([size, state_size], dtype=tf.float32)
            self.current_states = tf.Variable(state_init)
            self.next_states = tf.Variable(state_init)

            # save the current position
            self.size = size

            self.counter_init = tf.zeros([], dtype=tf.int32)
            self.count = tf.Variable(self.counter_init)
            self.current = tf.Variable(self.counter_init)

            # create necessary operations
            self.sample_size = sample_size

    # ---------------------- Memory Interface ---------------------------

    def reset_graph(self):
        """This method delivers the operation for resetting the
        replay memory."""

        reset_count = tf.assign(self.count, self.counter_init)
        reset_current = tf.assign(self.current, self.counter_init)
        return tf.group(reset_count, reset_current)

    def store_graph(self, current_state, next_state, action, reward, done):
        """This method inserts a new tuple into the replay memory.

        Args:
            current_state: The current_state in a binary encoded fashion.
            reward: The reward for the action taken
            action: The action that was taken for the reward
            next_state: The state after the action was executed.
            done: Whether the the episode was finished or not.
        """

        # create a new variable scope
        with tf.variable_scope("replay_memory"):

            # insert values
            exp_reward = tf.expand_dims(reward, 0)
            exp_current = tf.expand_dims(self.current, 0)
            exp_current_state = tf.expand_dims(current_state, 0)
            exp_next_state = tf.expand_dims(next_state, 0)
            exp_done = tf.expand_dims(done, 0)

            insert_action = tf.scatter_update(self.actions, exp_current, action)
            insert_reward = tf.scatter_update(self.rewards, exp_current, exp_reward)
            insert_done = tf.scatter_update(self.dones, exp_current, exp_done)
            insert_current_state = tf.scatter_update(self.current_states, exp_current, exp_current_state)
            insert_next_state = tf.scatter_update(self.next_states, exp_current, exp_next_state)

            # create increase counters
            increase_current = tf.assign(self.current, tf.mod(self.current + 1, self.size))
            increase_count = tf.cond(tf.less(self.count, self.size), lambda: tf.assign_add(self.count, 1), lambda: tf.assign(self.count, self.count))

            # pass back all operations melted together
            insert_op = tf.group(insert_action, insert_reward, insert_current_state, insert_next_state, insert_done,
                                 increase_current, increase_count)

            with tf.control_dependencies([insert_op]):
                return tf.identity(self.size - self.count)

    def store_and_sample_graph(self, current_state, next_state, action, reward, done):
        """This method inserts a new tuple into the replay memory.

        Args:
            current_state: The current_state in a binary encoded fashion.
            reward: The reward for the action taken
            action: The action that was taken for the reward
            next_state: The state after the action was executed.
            done: Whether the the episode was finished or not.
        """

        insert_count = self.store_graph(current_state, next_state, action, reward, done)

        # create a new variable scope
        with tf.variable_scope("replay_memory"):
            insert_op = tf.group(insert_count)
            with tf.control_dependencies([insert_op]):
                samples = self.__create_samples(self.sample_size)

            return samples

    # ---------------------- Private Functions ---------------------------

    def __create_samples(self, sample_size):
        """This method creates the sampling operation.

        Args:
            sample_size: How much samples should be derived.
        """

        # get unique indices
        permutation = tf.random_shuffle(tf.range(0, self.count, dtype=tf.int32))
        indices = permutation[:tf.minimum(sample_size, self.count)]

        # gather the values
        gathered_actions = tf.gather(self.actions, indices)
        gathered_rewards = tf.gather(self.rewards, indices)
        gathered_dones = tf.gather(self.dones, indices)
        gathered_current_states = tf.gather(self.current_states, indices)
        gathered_next_states = tf.gather(self.next_states, indices)

        # pass back the sampled actions
        return gathered_current_states, gathered_next_states, gathered_actions, gathered_rewards, gathered_dones
