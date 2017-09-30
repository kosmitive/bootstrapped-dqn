import gym
import tensorflow as tf
import os

from gym.spaces import Box
from gym.spaces import Discrete
from environments.Environment import Environment


class GeneralOpenAIEnvironment(Environment):

    def __init__(self, name):
        """Constructs a new general environment with a specified name. In detail one can
        create multiple environments, which are all capable of retaining their own state.

        Args:
            name - The name of the environment as registered in open ai gym.
        """

        # save properties
        self.name = name

        # create the environment and verify spaces
        self.env = gym.make(name)

        # check if the spaces are valid
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Discrete)

        with tf.variable_scope(self.name):

            # create pointer for the observation space itself
            self.current_observation = tf.Variable(tf.zeros(self.env.observation_space.shape, dtype=tf.float32), trainable=False)
            self.next_observation = None

    # ---------------------- Environment Interface ---------------------------

    def observation_space(self): return self.env.observation_space

    def action_space(self): return self.env.action_space

    def current_observation_graph(self): return self.current_observation

    def step_graph(self, action):
        with tf.variable_scope(self.name):
            next_observation, reward, done = tf.py_func(self.__one_step, [tf.cast(action, tf.int64)], [tf.float64, tf.float64, tf.bool])
            self.next_observation = tf.cast(next_observation, tf.float32)
            return self.next_observation, tf.cast(reward, tf.float32), done

    def apply_step_graph(self):
        with tf.variable_scope(self.name):
            return tf.assign(self.current_observation, self.next_observation)

    def render(self): self.env.render()

    def reset_graph(self):
        """This method simply executed the reset function from the environment"""
        with tf.variable_scope(self.name):
            observation = tf.py_func(self.env.reset, [], tf.float64)
            return tf.assign(self.current_observation, tf.cast(observation, tf.float32))

    # ---------------------- Private Functions ---------------------------

    def __one_step(self, action):
        observation, reward, done, info = self.env.step(action[0])
        return observation, reward, done

