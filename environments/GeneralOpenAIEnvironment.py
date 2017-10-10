import gym
import numpy as np
import tensorflow as tf

from gym.spaces import Box
from gym.spaces import Discrete
from environments.Environment import Environment
from spaces.ContinuousSpace import ContinuousSpace
from spaces.DiscreteSpace import DiscreteSpace

class GeneralOpenAIEnvironment(Environment):

    def __init__(self, env_name):
        """Constructs a new general environment with a specified name. In detail one can
        create multiple environments, which are all capable of retaining their own state.

        Args:
            name - The name of the environment as registered in open ai gym.
            N - The number of models to initialize
        """

        super().__init__("OpenAI-{}".format(env_name))

        # save properties
        self.env_name = env_name

        # create the environment and verify spaces
        self.env = gym.make(env_name)

        # check if the spaces are valid
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Discrete)

        # save the spaced correctly
        self.o_space = ContinuousSpace(self.env.observation_space.shape[0],
                                       self.env.observation_space.low,
                                       self.env.observation_space.high)
        self.a_space = DiscreteSpace(self.env.action_space.n)

        # save the observation space
        with tf.variable_scope(self.env_name):

            # create pointer for the observation space itself
            init = tf.zeros([self.o_space.dim()], dtype=tf.float32)
            self.current_observation = tf.Variable(init, trainable=False)
            self.next_observations = None

    # ---------------------- Environment Interface ---------------------------

    def random_walk(self, steps):
        """This executes a random walk for x steps"""

        current_observations = list()
        next_observations = list()
        rewards = list()
        dones = list()

        # add to current observations
        c_obs = self.env.reset()
        np.random.seed(0)
        all_actions = np.random.random_integers(0, self.a_space.dim() - 1, steps)

        for s in range(steps):
            if s % 1000 == 0: print("\tStep {}/{}".format(int(s / 1000), int((steps + 999) / 1000)))
            n_obs, reward, done, _ = self.env.step(all_actions[s])
            rewards.append(reward)
            dones.append(done)
            current_observations.append(c_obs)
            next_observations.append(n_obs)
            if done: self.env.reset()
            c_obs = n_obs

        # finally reset it again
        self.env.reset()

        # now stack the lists up
        fin_cobs = np.stack(next_observations)
        fin_nobs = np.stack(current_observations)
        fin_rews = np.stack(rewards)
        fin_dones = np.stack(dones)

        return fin_cobs, fin_nobs, all_actions, fin_rews, fin_dones

    def observation_space(self):
        return self.o_space

    def action_space(self):
        return self.a_space

    def current_observation_graph(self): return self.current_observation

    def step_graph(self, actions):
        """This method receives a vector of N actions."""
        with tf.variable_scope(self.env_name):
            next_observations, rewards, dones = tf.py_func(self.__one_step, [tf.cast(actions, tf.int64)], [tf.float64, tf.float64, tf.bool])
            self.next_observations = tf.cast(next_observations, tf.float32)
            return self.next_observations, tf.cast(rewards, tf.float32),  tf.cast(dones, tf.int32)

    def apply_step_graph(self):
        with tf.variable_scope(self.env_name):
            return tf.assign(self.current_observation, self.next_observations)

    def render(self):
        """Render the first D environments."""

        self.env.render()

    def reset_graph(self):
        """This method simply executed the reset function for each environment"""

        with tf.variable_scope(self.env_name):

            # map over all observations
            observation = tf.py_func(self.env.reset, [], tf.float64)

            # afterwards group them in one operation
            return tf.assign(self.current_observation, tf.cast(observation, tf.float32))

    # ---------------------- Private Functions ---------------------------

    def __one_step(self, action):

        observation, reward, done, info = self.env.step(action[0])
        return observation, reward, done
