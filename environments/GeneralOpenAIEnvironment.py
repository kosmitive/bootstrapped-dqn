import gym
import tensorflow as tf
import os

from gym.spaces import Box
from gym.spaces import Discrete


class GeneralOpenAIEnvironment:

    def __init__(self, name, dir, render, monitor):
        """Constructs a new general environment with a specified name."""

        # save properties
        self.name = name
        self.save_dir = os.path.join(dir, name)
        self.render = render

        # create if not exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # create the environment and verify spaces
        self.env = gym.make(name)
        if monitor:
            self.env = gym.wrappers.Monitor(self.env, self.save_dir, force=True, video_callable=False)

        # check if the spaces are valid
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Discrete)

        with tf.variable_scope(self.name):

            # create pointer for the observation space itself
            self.current_observation = tf.Variable(tf.zeros(self.env.observation_space.shape, dtype=tf.float32))
            self.next_observation = None

    def reset_op(self):
        with tf.variable_scope(self.name):
            observation = tf.py_func(self.env.reset, [], tf.float64)
            return tf.assign(self.current_observation, tf.cast(observation, tf.float32))

    def perform_step_op(self, action):
        with tf.variable_scope(self.name):
            next_observation, reward, done = tf.py_func(self.__one_step, [tf.cast(action, tf.int64)], [tf.float64, tf.float64, tf.bool])
            self.next_observation = tf.cast(next_observation, tf.float32)
            return self.next_observation, tf.cast(reward, tf.float32), done

    def apply_op(self):
        with tf.variable_scope(self.name):
            return tf.assign(self.current_observation, self.next_observation)

    def __one_step(self, action):
        observation, reward, done, info = self.env.step(action[0])
        return observation, reward, done

    def render_if_activated(self):
        """This method renders the internally saved environment."""

        if self.render:
            self.env.render()