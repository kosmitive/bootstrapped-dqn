import gym
import numpy as np
import tensorflow as tf
import threading
import time

from gym.spaces import Box
from gym.spaces import Discrete
from gym.envs.classic_control.mountain_car import MountainCarEnv
from environments.Environment import Environment
from spaces.ContinuousSpace import ContinuousSpace
from spaces.DiscreteSpace import DiscreteSpace


class GeneralOpenAIEnvironment(Environment):

    def __init__(self, env_name, num_models):
        """Constructs a new general environment with a specified name. In detail one can
        create multiple environments, which are all capable of retaining their own state.

        Args:
            name - The name of the environment as registered in open ai gym.
            N - The number of models to initialize
        """

        super().__init__("OpenAI-{}".format(env_name))

        # save properties
        self.env_name = env_name
        self.num_models = num_models
        self.real_dones = num_models * [False]

        # save the observation space
        self.envs = [gym.make(env_name) for _ in range(num_models)]

        # check if the spaces are valid
        assert isinstance(self.envs[0].observation_space, Box)
        assert isinstance(self.envs[0].action_space, Discrete)

        # save the spaced correctly
        self.o_space = ContinuousSpace(self.envs[0].observation_space.shape[0],
                                       self.envs[0].observation_space.low,
                                       self.envs[0].observation_space.high)
        self.a_space = DiscreteSpace(self.envs[0].action_space.n)

        with tf.variable_scope(self.env_name):

            # create pointer for the observation space itself
            init = tf.zeros([num_models, self.o_space.dim()], dtype=tf.float32)
            self.current_observation = tf.Variable(init, trainable=False)
            self.next_observations = None

    # ---------------------- Environment Interface ---------------------------

    def random_walk(self, steps):
        """This executes a random walk for x steps"""

        # get the maximum steps of all
        interval = 1000
        max_steps = int((steps * self.num_models + interval - 1) / interval)

        # define random walk function
        def single_random_walk(model, results, rand_steps):

            np.random.seed(model)
            current_observations = list()
            next_observations = list()
            rewards = list()
            dones = list()

            # add to current observations
            c_obs = self.envs[model].reset()
            np.random.seed(0)
            all_actions = np.random.random_integers(0, self.a_space.dim() - 1, steps)

            for s in range(steps):
                n_obs, reward, done, _ = self.envs[model].step(all_actions[s])
                rewards.append(reward)
                dones.append(done)
                current_observations.append(c_obs)
                next_observations.append(n_obs)
                #if done: self.envs[model].reset()
                c_obs = n_obs

                # every 1000 steps
                if s % interval == 0:

                    # increase rand step list
                    rand_steps[model] += 1
                    summed_steps = np.sum(rand_steps)
                    print("\tModel_{}; Step {}/{}".format(model, summed_steps, max_steps))

            # finally reset it again
            self.envs[model].reset()

            # now stack the lists up
            fin_cobs = np.stack(next_observations)
            fin_nobs = np.stack(current_observations)
            fin_rews = np.stack(rewards)
            fin_dones = np.stack(dones)
            results[model] = [fin_cobs, fin_nobs, all_actions, fin_rews, fin_dones]

        # iterate over all models
        results = [None] * self.num_models
        rand_steps = [0] * self.num_models

        threads = list()
        for k in range(self.num_models):
            threads.append(threading.Thread(target=single_random_walk, args=(k, results, rand_steps)))
            threads[k].start()

        for k in range(len(threads)):
            threads[k].join()

        return results

    def observation_space(self):
        return self.o_space

    def action_space(self):
        return self.a_space

    def current_observation_graph(self): return self.current_observation

    def step_graph(self, actions):
        """This method receives a vector of N actions."""
        with tf.variable_scope(self.env_name):
            next_observations, rewards, dones, stdone = tf.py_func(self.__one_step, [tf.cast(actions, tf.int64)], [tf.float64, tf.float64, tf.bool, tf.bool])
            self.next_observations = tf.cast(next_observations, tf.float32)
            return self.next_observations, tf.cast(rewards, tf.float32),  tf.cast(dones, tf.int32),  tf.cast(stdone, tf.int32)

    def apply_step_graph(self):
        with tf.variable_scope(self.env_name):
            return tf.assign(self.current_observation, self.next_observations)

    def render(self, D):
        """Render the first D environments."""

        self.envs[0].render()

    def reset_graph(self):
        """This method simply executed the reset function for each environment"""

        with tf.variable_scope(self.env_name):

            # map over all observations
            observation = tf.py_func(self.__reset_envs, [], tf.float64)

            # afterwards group them in one operation
            return tf.assign(self.current_observation, tf.cast(observation, tf.float32))

    # ---------------------- Private Functions ---------------------------

    def __reset_envs(self):
        val =np.stack([self.envs[k].reset() for k in range(self.num_models)], axis=0)
        self.real_dones = self.num_models * [False]
        return val

    def __one_step(self, action):
        observations = list()
        rewards = list()
        dones = list()
        stop_train_done = list()

        for k in range(self.num_models):
            observation, reward, done, _ = self.envs[k].step(action[k][0])
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            stop_train_done.append(done and self.real_dones[k])
            self.real_dones[k] = done
        return np.stack(observations, 0), np.stack(rewards, 0), np.stack(dones, 0), np.stack(stop_train_done, 0)
