# MIT License
#
# Copyright (c) 2017 Markus Semmler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gym
import numpy as np
import tensorflow as tf

from gym.spaces import Box
from gym.spaces import Discrete
from environments.Environment import Environment
from util.Space import Space
from multiprocessing.dummy import Pool as ThreadPool


class GeneralOpenAIEnvironment(Environment):

    def __init__(self, env_name, N):
        """Constructs a new general environment with. In detail one can
        create multiple environments, which are all capable of retaining their own state.

        Args:
            name - The name of the environment as registered in open ai gym.
            N - The number of models to initialize
        """

        # save the observation space
        env = gym.make(env_name)
        eos = env.observation_space
        aos = env.action_space

        assert isinstance(eos, Box)
        assert isinstance(aos, Discrete)

        # create the continuous space
        state_space = Space(list(zip(eos.low, eos.high)))
        action_space = Space([aos.n])
        super().__init__("openai_{}".format(env_name), state_space, action_space, N)

        # init the other environments
        self.envs = [env] + [gym.make(env_name) for _ in range(N - 1)]

        # set up a thread pool
        threads = 16
        self.chunks = np.maximum(int(N / threads + 0.5), 10)
        self.pool = ThreadPool(threads)
        self.indices = list(range(N))

# --- NP ---

    def render(self, w):
        """Simply render the environment of the passed ind."""
        self.envs[w].render()

# --- TF ---

    def _next_observation_graph(self, actions):
        """This method receives a vector of N actions."""
        next_observations, rewards, dones =\
            tf.py_func(self._one_step, [tf.cast(actions, tf.int64)],
                       [tf.float64, tf.float64, tf.bool])
        self.next_observations = tf.cast(next_observations, tf.float32)
        return self.next_observations, tf.cast(rewards, tf.float32),  tf.cast(dones, tf.int32)

    def _reset_graph(self):
        """This method simply executed the reset function for each environment"""

        # map over all observations
        return tf.py_func(self._reset_envs, [], tf.float64)

# --- PY Funcs ---

    def _reset(self, w):
        """Simply overwrite and return an initial state."""
        return self.envs[w].reset()

    def _step(self, w, state, action):
        """Pass back reward, next state and if the episode
        is finished. (r, ns, d)"""
        return self.envs[w].step(action)

    def _reset_envs(self):
        obs = self.pool.imap(self._reset, self.indices, self.chunks)
        self.pool.close()
        self.pool.join()
        return np.stack(obs, axis=0)

    def _one_step(self, action):
        obs = self.pool.imap(lambda k: self._step(k, None, action[k][0]), self.indices, self.chunks)
        self.pool.close()
        self.pool.join()

        tobs = [np.stack([o[k] for o in obs], axis=0) for k in range(3)]
        return tobs