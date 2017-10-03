import tensorflow as tf

from environments.Environment import Environment
from memory.Memory import Memory
from spaces.ContinuousSpace import ContinuousSpace
from spaces.DiscreteSpace import DiscreteSpace

class ExperienceReplayMemory(Memory):
    """This class represent a basic replay memory. It can basically
    store the last N tuples.
    """

    def __init__(self, N, size, sample_size, env):
        """Constructs a new ReplayMemory.

        Args:
            size: The size of the replay memory.
            sample_size: How much samples should be derived.
            env: The environment used for initialization
            N: Define how much memories should be managed
        """

        # check if this is a valid size
        assert size > 0
        assert sample_size <= size
        assert isinstance(env, Environment)

        # obtain the spaces
        state_space = env.observation_space()
        action_space = env.action_space()

        # create a new variable scope
        with tf.variable_scope("replay_memory"):

            # init action and reward value
            self.actions = tf.Variable(tf.zeros([size, N], dtype=tf.int32))
            self.rewards = tf.Variable(tf.zeros([size, N], dtype=tf.float32))
            self.dones = tf.Variable(tf.zeros([size, N], dtype=tf.int32))

            # init state space size
            state_size = state_space.dim()
            state_init = tf.zeros([size, N, state_size], dtype=tf.float32)
            self.current_states = tf.Variable(state_init)
            self.next_states = tf.Variable(state_init)

            # save the current position
            self.size = size
            self.counter_init = tf.zeros([], dtype=tf.int32)
            self.count = tf.Variable(self.counter_init)
            self.current = tf.Variable(self.counter_init)

            # create necessary operations
            self.sample_size = sample_size
            self.N = N

    # ---------------------- Memory Interface ---------------------------

    def fill_graph(self, current_states, next_states, actions, rewards, dones):

        with tf.variable_scope("replay_memory"):
            fill_count = tf.assign(self.count, self.size)
            reset_current = tf.assign(self.current, self.counter_init)

            assign_cs_values = tf.assign(self.current_states, current_states)
            assign_ns_values = tf.assign(self.next_states, next_states)
            assign_a_values = tf.assign(self.actions, actions)
            assign_r_values = tf.assign(self.rewards, rewards)
            assign_d_values = tf.assign(self.dones, dones)

            return tf.group(fill_count, reset_current,
                            assign_a_values, assign_cs_values, assign_ns_values, assign_r_values, assign_d_values)

    def reset_graph(self):
        """This method delivers the operation for resetting the
        replay memory."""
        with tf.variable_scope("replay_memory"):
            reset_count = tf.assign(self.count, self.counter_init)
            reset_current = tf.assign(self.current, self.counter_init)
            return tf.group(reset_count, reset_current)

    def store_graph(self, current_states, next_states, actions, rewards, dones):
        """This method inserts a new tuple into the replay memory.

        Args:
            current_states: The current_state in a binary encoded fashion.
            rewards: The reward for the action taken
            actions: The action that was taken for the reward
            next_states: The state after the action was executed.
            dones: Whether the the episode was finished or not.
        """

        # create a new variable scope
        with tf.variable_scope("replay_memory"):

            # insert values
            exp_current = tf.expand_dims(self.current, 0)

            insert_action = tf.scatter_update(self.actions, exp_current, tf.expand_dims(actions, 0))
            insert_reward = tf.scatter_update(self.rewards, exp_current, tf.expand_dims(rewards, 0))
            insert_done = tf.scatter_update(self.dones, exp_current, tf.expand_dims(tf.cast(dones, tf.int32), 0))
            insert_current_state = tf.scatter_update(self.current_states, exp_current, tf.expand_dims(current_states, 0))
            insert_next_state = tf.scatter_update(self.next_states, exp_current, tf.expand_dims(next_states, 0))

            # create increase counters
            increase_current = tf.assign(self.current, tf.mod(self.current + 1, self.size))
            increase_count = tf.cond(tf.less(self.count, self.size), lambda: tf.assign_add(self.count, 1), lambda: tf.assign(self.count, self.count))

            # pass back all operations melted together
            insert_op = tf.group(insert_action, insert_reward, insert_current_state, insert_next_state, insert_done,
                                 increase_current, increase_count)

            with tf.control_dependencies([insert_op]):
                return tf.identity(self.size - self.count)

    def store_and_sample_graph(self, current_states, next_states, actions, rewards, dones):
        """This method inserts a new tuple into the replay memory.

        Args:
            current_states: The current_state in a binary encoded fashion.
            rewards: The reward for the action taken
            actions: The action that was taken for the reward
            next_states: The state after the action was executed.
            dones: Whether the the episode was finished or not.
        """

        insert_count = self.store_graph(current_states, next_states, actions, rewards, dones)

        # create a new variable scope
        with tf.variable_scope("replay_memory"):
            with tf.control_dependencies([insert_count]):
                samples = self.__create_samples(self.sample_size)

            return samples

    # ---------------------- Private Functions ---------------------------

    def __create_samples(self, sample_size):
        """This method creates the sampling operation.

        Args:
            sample_size: How much samples should be derived.
        """

        # create the indices for use with gather
        ind_range = tf.range(0, self.count, dtype=tf.int32)
        ind_list = list()

        # iterate over
        for n in range(self.N):

            # get unique indices
            permutation = tf.random_shuffle(ind_range)
            first_indices = n * tf.ones([sample_size], dtype=tf.int32)
            indices = permutation[:sample_size]
            combined_ind = tf.stack([indices, first_indices], axis=1)
            ind_list.append(combined_ind)

        # obtain all indices
        all_indices = tf.concat(ind_list, axis=0)

        # simple function to reduce code size
        def gather_and_split(tensor, g_indices):
            gathered_values = tf.gather_nd(tensor, g_indices)
            splitted_values = tf.split(gathered_values, num_or_size_splits=self.N, axis=0)
            return tf.stack(splitted_values, axis=0)

        # gather the values
        gathered_actions = gather_and_split(self.actions, all_indices)
        gathered_rewards = gather_and_split(self.rewards, all_indices)
        gathered_dones = gather_and_split(self.dones, all_indices)
        gathered_current_states = gather_and_split(self.current_states, all_indices)
        gathered_next_states = gather_and_split(self.next_states, all_indices)

        # pass back the sampled actions
        return gathered_current_states, gathered_next_states, gathered_actions, gathered_rewards, gathered_dones

