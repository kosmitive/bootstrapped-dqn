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

from environments.open_ai_envs.ContinualStateEnv import ContinualStateEnv
from memory.Memory import Memory
from policies.Policy import Policy


class GeneralAgent:

    def __init__(self, name, env, conf):
        """Constructs an interface of NeuralAgent.

        Args:
            name: The name of the agent.
            env: The environment to play.
            conf: The configuration for additional settings
        """

        # save vars
        self.name = name
        self.conf = conf

        # check if correct instances
        assert isinstance(env, ContinualStateEnv)

        # save the spaces
        self.state_space = env.observation_space()
        self.action_space = env.action_space()

        # some initial standard settings
        self.memory = None
        self.policy = None

    # --- Register Points -----------------------------------------------

    def register_memory(self, memory):
        assert isinstance(memory, Memory)
        self.memory = memory

    def register_policy(self, policy):
        assert isinstance(policy, Policy)
        self.policy = policy

    # --- Graphs -----------------------------------------------

    def pre_episode_graph(self):
        """This graph is executed before an episode is executed before each episode.

        Returns:
            The graph performing some actions.
        """
        raise NotImplementedError()

    def action_graph(self, current_observations):
        """This graph takes NxD observations and outputs the chosen actions.

        Args:
            current_observations: The current observations.

        Returns:
            The chosen N actions
        """
        raise NotImplementedError()

    def observe_graph(self, current_observation, next_observation, action, reward, done):
        """This graph takes an experience tuple and either samples a batch from
        the supplied replay memory or takes the tuple and uses that to create a minimizer.

        Args:
            current_observation: The current observation
            next_observation: The next observation
            action: The chosen action
            reward: The reward received for performing the action
            done: Whether the episode is finished or not.

        Returns:
            A operation which learns from the samples.
        """
        exp_tuple = (current_observation, next_observation, action, reward, done)

        if self.memory is None:
            return self.learn_graph(*exp_tuple)
        else:
            exp_batch = self.memory.store_and_sample_graph(*exp_tuple)
            return self.learn_graph(*exp_batch)

    def learn_graph(self, current_states, next_states, actions, rewards, dones):
        """This graph takes N experience tuple  uses these to create a minimizer.

        Args:
            current_states: The current observations
            next_states: The next observations
            actions: The chosen actions
            rewards: The reward received for performing the actions
            dones: Whether the episode is finished or not.

        Returns:
            A operation which learns from the samples.
        """
        raise NotImplementedError()
