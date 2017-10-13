# import the necessary packages
import numpy as np
import tensorflow as tf

from utils.Space import Space
from environments.Environment import Environment


class DeterministicMDP(Environment):
    """This represents a deterministic MDP"""

    def __init__(self, name, state_space, action_space, N, dynamics):
        """Construct a new general deterministic MDP.

        Args:
            name: The name of the MDP
            state_space: The discrete state space
            action_space The discrete action space
            N: Number of states to maintain
            dynamics: List of transitio, reward and initial state initially
        """

        # unpack the dynamics
        transition_function, reward_function, initial_state = dynamics

        # check if the parameters are indeed the correct instances
        assert isinstance(action_space, Space)
        assert isinstance(state_space, Space)
        assert all(state_space.is_discrete)
        assert all(action_space.is_discrete)
        assert isinstance(initial_state, int) and 0 <= initial_state < state_space.get_size()

        name = "mdp_{}".format(name)
        super().__init__(name, state_space, action_space, N)

        # save the number of states
        self.initial_state = initial_state

        # first of all normalize the rewards
        mi = np.min(reward_function)
        ma = np.max(reward_function)
        reward_function = (reward_function - mi) / (ma - mi)

        # Do some assertions on the passed reward and transition functions.
        # They need to have the height of the state space and the width of
        state_action_shape = (state_space.D, action_space.D)
        assert np.shape(transition_function) == state_action_shape
        assert np.shape(reward_function) == state_action_shape

        # check if transition function is valid
        for i in range(np.size(transition_function, 0)):
            for j in range(np.size(transition_function, 1)):
                assert 0 <= transition_function[i, j] < state_space.get_size()

        # create tensorflow constants for the dynamics
        with tf.variable_scope(name):
            self.transition = tf.constant(transition_function, dtype=tf.int64)
            self.rewards = tf.constant(reward_function, dtype=tf.float64)
            self.reward_function = reward_function
            self.transition_function = transition_function

# --- NP ---

    def _reset(self, w):
        """Simply overwrite and return an initial state."""
        return self.initial_state

    def _step(self, w, state, action):
        """Pass back reward, next state and if the episode
        is finished. (r, ns, d)"""

        # access the state action values inside of the transition function
        next_state = self.transition[state, action]
        rewards = self.rewards[state, action]
        done = False
        return rewards, next_state, done

    def get_optimal(self, steps, discount):
        """This gets the optimal reward using value iteration."""

        state_size = self.state_space.get_size()
        action_size = self.action_space.get_size()

        # init q function
        q_shape = (state_size, action_size)
        q_function = -np.ones(q_shape)
        next_q_function = np.zeros(q_shape)

        # repeat until converged
        while np.max(np.abs(q_function - next_q_function)) >= 0.001:

            # create next bootstrapped q function
            q_function = next_q_function
            bootstrapped_q_function = np.empty(q_shape)

            # iterate over all fields
            for s in range(state_size):
                for a in range(action_size):
                    next_state = self.transition_function[s, a]
                    bootstrapped_q_function[s, a] = np.max(q_function[next_state, :])

            # update the q function correctly
            next_q_function = self.reward_function + discount * bootstrapped_q_function

        # create new environment and simulate
        optimal_policy = np.argmax(q_function, axis=1)
        reward = 0
        current_state = self.initial_state

        # run for specified number of steps
        for k in range(steps):
            reward += self.reward_function[current_state, optimal_policy[current_state]]
            current_state = self.transition_function[current_state, optimal_policy[current_state]]

        optimal_reward = reward

        # create new environment and simulate
        minimal_policy = np.argmin(q_function, axis=1)
        reward = 0
        current_state = self.initial_state

        # run for specified number of steps
        for k in range(steps):
            reward += self.reward_function[current_state, minimal_policy[current_state]]
            current_state = self.transition_function[current_state, minimal_policy[current_state]]

        minimal_reward = reward

        return optimal_reward, minimal_reward, np.min(q_function), np.max(q_function), q_function


# --- TF ---

    def _next_observation_graph(self, actions):
        """Creates the graph from actions to the next state.

        Returns:
            actions: A tf.Tensor specifying the actions that should be taken.
        """

        # access the state action values inside of the transition function
        selection = tf.stack([self.current_states, actions], axis=1)
        next_state = tf.gather_nd(self.transition, selection)
        rewards = tf.gather_nd(self.rewards, selection)

        # save the reward and update state
        return next_state, rewards, tf.zeros([self.num_states], dtype=tf.bool)

    def _reset_graph(self):
        """This should reset everything which is not managed by this class."""
        tf.constant(self.initial_state, [self.num_states])
