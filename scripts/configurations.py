from agents.DDQNAgent import DDQNAgent
from environments.GeneralOpenAIEnvironment import GeneralOpenAIEnvironment
from memory.ExperienceReplayMemory import ExperienceReplayMemory
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from policies.GreedyPolicy import GreedyPolicy

import extensions.tensorflowHelpers as tfh
import tensorflow as tf

def ddqn_general_ddqn_eps_config(env_name, max_timesteps):

        # Replay memory
        replay_size = 100000
        sample_size = 128

        # DQN
        structure = [256, 256]
        agent_config = {
                'discount': 0.99,
                'learning_rate': 0.00025,
                'target_offset': 2000,
        }

        # Exploration reduction
        exp_rate_begin = 1.0
        exp_rate_end = 0.1
        decay_lambda = 0.1

        env = GeneralOpenAIEnvironment(env_name)
        memory = ExperienceReplayMemory(replay_size, sample_size, env)
        agent = DDQNAgent(env, structure, agent_config)

        # obtain an exploration strategy
        exploration_parameter = tfh.exp_decay(exp_rate_begin, exp_rate_end, decay_lambda, agent.global_step)
        exploration_parameter = tfh.linear_decay(int(decay_lambda * 100000),
                                                 exp_rate_begin, exp_rate_end, agent.global_step)
        exploration_strategy = EpsilonGreedyPolicy(env.action_space(), exploration_parameter)
        # exploration_strategy = GreedyPolicy()
        return [env, agent, memory, exploration_strategy, exploration_parameter], agent_config


def ddqn_general_ddqn_greedy_config(env_name):

        # Replay memory
        replay_size = 100000
        sample_size = 32

        # DQN
        structure = [32, 32, 32, 32, 32]
        agent_config = {
                'discount': 0.99,
                'learning_rate': 0.000025,
                'target_offset': 1000,
        }

        env = GeneralOpenAIEnvironment(env_name)
        memory = ExperienceReplayMemory(replay_size, sample_size, env)
        agent = DDQNAgent(env, structure, agent_config)

        exploration_strategy = GreedyPolicy()
        return [env, agent, memory, exploration_strategy, tf.constant(0.0)], agent_config