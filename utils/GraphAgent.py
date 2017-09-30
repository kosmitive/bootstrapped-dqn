import tensorflow as tf

from agents.DDQNAgent import DDQNAgent
from environments.GeneralOpenAIEnvironment import GeneralOpenAIEnvironment
from memory.ExperienceReplayMemory import ExperienceReplayMemory
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy


class GraphCreator:
    def __init__(self, name,
                 replay_size, sample_size,
                 structure, copy_offset,
                 starter_exploration_rate, end_exploration_rate,
                 discount, learning_rate, decay_lambda):

        self.name = name
        self.replay_size = replay_size
        self.sample_size = sample_size
        self.structure = structure
        self.copy_offset = copy_offset
        self.starter_exploration_rate = starter_exploration_rate
        self.end_exploration_rate = end_exploration_rate
        self.discount = discount
        self.learning_rate = learning_rate
        self.decay_lambda = decay_lambda

    def create_agent(self, render):



        # build the final graph
        current_observation = env.current_observation
        action = agent.action_graph(env.current_observation, exploration_strategy)

        # execute action
        next_observation, reward, done = env.perform_step_op(action)

        # create the learn graph
        minimizer = agent.observe_graph(current_observation, next_observation,
                                        action, reward, self.discount, self.learning_rate, done)

        # execute the minimizer before "applying" the action
        with tf.control_dependencies([minimizer]):
            one_step = env.apply_op()

        return agent, env, [one_step, done, reward]

    def create_multiple_agents(self, count):
        agent_lst = list()
        env_lst = list()
        env_output = list()

        for k in range(count):
            with tf.variable_scope("agent{}".format(k)):
                new_agent, new_env, new_output = self.create_agent(k == 0)
                env_lst.append(new_env)
                env_output.append(new_output)
                agent_lst.append(new_agent)

        return agent_lst, env_lst, env_output