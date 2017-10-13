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

import numpy as np
import tensorflow as tf


from utils.Space import Space


class Environment:
    """This represents the base for all environments in this project."""

    def __init__(self, name, state_space, action_space, N=1, config=None):
        """Creates a new environment, based on the state and action
        space.

        Args:
            name: One unique name is given.
            state_space: Specifies the state space
            action_space: Specifies the action space
            N: The number of different states.
            config: Nothing yet
        """
        assert isinstance(state_space, Space)
        assert isinstance(action_space, Space)
        self.state_space = state_space
        self.action_space = action_space

        # save other settings
        self.num_states = N
        self.name = name
        self.conf = config

        with tf.variable_scope(name):

            # some general information
            self.expired_steps = tf.get_variable("expired_steps", [N, 1], dtype=tf.int32)
            self.dones = tf.get_variable("dones", [N, 1], dtype=tf.bool)
            self.inc_expired_steps = tf.assign_add(self.expired_steps, tf.to_int32(tf.logical_not(self.dones)))

            # build variables for current observations
            self.current_states = tf.get_variable("cur_states", [N, state_space.D])
            self.next_states = tf.get_variable("next_states", [N, state_space.D])

            # pipe only for active agents
            next_current_state = tf.where(self.dones, self.current_states, self.next_states)
            self.pipe_state = tf.assign(self.current_states, next_current_state)

            self.cum_rewards = tf.get_variable("rewards", [N, 1])
            self.actions = tf.get_variable("actions", [N, 1], dtype=tf.int32)

# --- NP ---

    def _reset(self, w):
        """Simply overwrite and return an initial state."""
        raise NotImplementedError()

    def random_walk(self, x):
        """This executes a random walk for x steps and passes
        back the trajectory over observations and the obtained reward

        Args:
            x: The number of steps

        Returns:
            A list of tuples [cs, a, r, ns, d] where each element is numpy
            tensor over the time.
        """

        walk_exps = list()
        reset = self._reset()
        for w in range(self.num_states):

            # create the list of experiences
            cst = reset(w)
            exps = list()
            # run the graph for the number of steps
            for s in range(x):
                act = self.action_space.sample_element()
                rew, nst, d = self._step(w, cst, act)
                exps.append([cst, act, rew, nst, d])
                cst = reset(w) if d else nst

            walk_exps.append([np.stack([e[k] for e in exps]) for k in range(5)])

        return walk_exps

    def _step(self, w, states, actions):
        """Pass back reward, next state and if the episode
        is finished. (r, ns, d)"""
        raise NotImplementedError()

    def render(self, w):
        """This method can be used to render the environment. No
        implementation is given and thus it stays open if
        a visualization is integrated.
        """
        raise NotImplementedError()

# --- TF ---

    def init_episode_graph(self):
        """This graph should be executed in prior to executing any episode.

        Returns:
            A operation which once executed initialized the graph
        """
        with tf.variable_scope(self.name):
            ns = tf.assign(self.current_states, self._reset_graph())
            ds = tf.assign(self.dones, False)
            es = tf.assign(self.expired_steps, 0)
            cr = tf.assign(self.cum_rewards, 0)
            return tf.group(ns, ds, es, cr)

    def experience_graph(self, actions):
        """Creates the graph for obtaining the next state, reward
        and if all agent are done. The reward is accumulated for later obtaining

        Returns:
            An action which observes and presents it to the outside world.
        """
        with tf.variable_scope(self.name):
            nsts, rews, dns = self._next_observation_graph(actions)

            # increase exp step and
            with tf.control_dependencies([self.inc_expired_steps, self.pipe_state]):
                pipe_next = tf.assign(self.next_states, tf.where(self.dones, self.next_states, nsts))
                pipe_dns = tf.assign(self.dones, tf.logical_or(self.dones, dns))

                count_rews = tf.assign_add(self.cum_rewards, tf.to_float(tf.logical_not(self.dones)) * rews)
                move_observation = tf.group(pipe_next, count_rews, pipe_dns)

            # make the observations available to the callee.
            with tf.control_dependencies([move_observation]):
                return [tf.identity(rews), tf.identity(self.next_states), tf.reduce_all(dns)]

    def collect_run_data_graph(self):
        """This graph makes some evaluations to pass back to the callee, so that it can
        be logged by the executing environment.
        """
        with tf.variable_scope(self.name):

            id_cum_rews = tf.identity(self.cum_rewards)
            id_exp_steps = tf.identity(self.expired_steps)
            success_ratio = tf.to_float(tf.bincount(tf.equal(self.expired_steps, tf.reduce_min(self.expired_steps)))) / tf.constant(self.num_states)

            return [id_cum_rews, id_exp_steps, success_ratio]

    def _next_observation_graph(self, actions):
        """Creates the graph from actions to the next state.

        Returns:
            actions: A tf.Tensor specifying the actions that should be taken.
        """
        raise NotImplementedError()

    def _reset_graph(self):
        """This should reset everything which is not managed by this class."""
        raise NotImplementedError()