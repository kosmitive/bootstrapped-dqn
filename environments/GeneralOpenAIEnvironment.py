import gym
import numpy as np
import tensorflow as tf

from gym.spaces import Box
from gym.spaces import Discrete
from environments.Environment import Environment
from spaces.ContinuousSpace import ContinuousSpace
from spaces.DiscreteSpace import DiscreteSpace

class GeneralOpenAIEnvironment(Environment):

    def __init__(self, N, env_name):
        """Constructs a new general environment with a specified name. In detail one can
        create multiple environments, which are all capable of retaining their own state.

        Args:
            name - The name of the environment as registered in open ai gym.
            N - The number of models to initialize
        """

        super().__init__("OpenAI-{}".format(env_name), N)

        # save properties
        self.env_name = env_name

        # create the environment and verify spaces
        self.envs = [gym.make(env_name) for _ in range(N)]

        # check if the spaces are valid
        assert isinstance(self.envs[0].observation_space, Box)
        assert isinstance(self.envs[0].action_space, Discrete)

        # save the spaced correctly
        self.o_space = ContinuousSpace(self.envs[0].observation_space.shape[0],
                                       self.envs[0].observation_space.low,
                                       self.envs[0].observation_space.high)
        self.a_space = DiscreteSpace(self.envs[0].action_space.n)

        # save the observation space
        with tf.variable_scope(self.env_name):

            # create pointer for the observation space itself
            init = tf.zeros([N, self.o_space.dim()], dtype=tf.float32)
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
        step_cobs = np.stack([self.envs[i].reset() for i in range(self.N)])
        step_actions = np.random.random_integers(0, self.a_space.dim() - 1, (steps, self.N))

        for s in range(steps):
            if s % 1000 == 0: print("\tStep {}/{}".format(int(s / 1000), int((steps + 999) / 1000)))
            step_nobs = list()
            step_rews = list()
            step_dones = list()
            for n in range(self.N):
                sn_obs, sn_rew, sn_dn, _ = self.envs[n].step(step_actions[s, n])
                step_nobs.append(sn_obs)
                step_rews.append(sn_rew)
                step_dones.append(sn_dn)
                if sn_dn: self.envs[n].reset()

            rewards.append(np.stack(step_rews))
            dones.append(np.stack(step_dones))
            current_observations.append(step_cobs)
            step_nobs = np.stack(step_nobs)
            next_observations.append(step_nobs)

            step_cobs = step_nobs

        # finally reset it again
        [self.envs[i].reset() for i in range(self.N)]

        # now stack the lists up
        fin_cobs = np.stack(next_observations)
        fin_nobs = np.stack(current_observations)
        fin_rews = np.stack(rewards)
        fin_dones = np.stack(dones)

        return fin_cobs, fin_nobs, step_actions, fin_rews, fin_dones

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
            return self.next_observations, tf.cast(rewards, tf.float32), dones

    def apply_step_graph(self):
        with tf.variable_scope(self.env_name):
            return tf.assign(self.current_observation, self.next_observations)

    def render(self, D):
        """Render the first D environments."""

        assert 0 < D <= self.N
        for d in range(D):
            self.envs[d].render()

    def reset_graph(self):
        """This method simply executed the reset function for each environment"""

        with tf.variable_scope(self.env_name):

            # map over all observations
            observation_assign_list = list()
            for n in range(self.N):
                with tf.variable_scope(str(n)):
                    observation = tf.py_func(self.envs[n].reset, [], tf.float64)
                    observation_assign_list.append(observation)

            # afterwards group them in one operation
            return tf.assign(self.current_observation, tf.cast(tf.stack(observation_assign_list), tf.float32))

    # ---------------------- Private Functions ---------------------------

    def __one_step(self, action):

        # iterate over all environments
        obs_list = list()
        rew_list = list()
        done_list = list()

        # perform the steps at each environment
        for n in range(self.N):
            observation, reward, done, info = self.envs[n].step(action[n])

            obs_list.append(observation)
            rew_list.append(reward)
            done_list.append(done)

        # stack them all up
        observations = np.stack(obs_list)
        rewards = np.stack(rew_list)
        dones = np.stack(done_list)

        return observations, rewards, dones
