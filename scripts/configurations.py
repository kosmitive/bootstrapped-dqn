from agents.DDQNAgent import DDQNAgent
from environments.GeneralOpenAIEnvironment import GeneralOpenAIEnvironment
from memory.ExperienceReplayMemory import ExperienceReplayMemory
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from policies.GreedyPolicy import GreedyPolicy

import extensions.tensorflowHelpers as tfh


def ddqn_general_ddqn_eps_config(env_name):

        # Replay memory
        replay_size = 100000
        sample_size = 64

        # DQN
        discount = 0.99
        learning_rate = 0.00025
        structure = [256, 256]
        copy_offset = 10000

        # Exploration reduction
        exp_rate_begin = 1.0
        exp_rate_end = 0.05
        decay_lambda = 0.001

        env = GeneralOpenAIEnvironment(env_name)
        memory = ExperienceReplayMemory(replay_size, sample_size, env)
        agent = DDQNAgent(env, structure, copy_offset, discount, learning_rate)

        # obtain an exploration strategy
        exploration_parameter = tfh.exp_decay(exp_rate_begin, exp_rate_end, decay_lambda, agent.global_step)
        exploration_strategy = EpsilonGreedyPolicy(env.action_space(), exploration_parameter)

        return [env, agent, memory, exploration_strategy]


def ddqn_general_ddqn_greedy_config(env_name):

        # Replay memory
        replay_size = 100000
        sample_size = 64

        # DQN
        discount = 0.99
        learning_rate = 0.00025
        structure = [256, 256]
        copy_offset = 10000

        # the basic environment
        env = GeneralOpenAIEnvironment(env_name)
        memory = ExperienceReplayMemory(replay_size, sample_size, env)
        agent = DDQNAgent(env, structure, copy_offset, discount, learning_rate)

        # obtain an exploration strategy
        exploration_strategy = GreedyPolicy()

        return [env, agent, memory, exploration_strategy]