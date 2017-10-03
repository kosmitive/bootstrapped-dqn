from agents.DDQNAgent import DDQNAgent
from environments.GeneralOpenAIEnvironment import GeneralOpenAIEnvironment
from memory.ExperienceReplayMemory import ExperienceReplayMemory
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from policies.GreedyPolicy import GreedyPolicy

import extensions.tensorflowHelpers as tfh
import tensorflow as tf

def ddqn_general_ddqn_eps_config(env_name, num_models):

        # Replay memory
        replay_size = 100000
        sample_size = 32

        # DQN
        structure = [256, 256]
        agent_config = {
                'discount': 0.99,
                'learning_rate': 0.00025,
                'target_offset': 10000,
        }

        # Exploration reduction
        exp_rate_begin = 1.0
        exp_rate_end = 0.1
        decay_lambda = 0.00001

        env = GeneralOpenAIEnvironment(num_models, env_name)
        memory = ExperienceReplayMemory(num_models, replay_size, sample_size, env)
        agent = DDQNAgent(num_models, env, structure, agent_config)

        # obtain an exploration strategy
        exploration_parameter = tfh.exp_decay(exp_rate_begin, exp_rate_end, decay_lambda, agent.global_step)
        exploration_strategy = EpsilonGreedyPolicy(num_models, env.action_space(), exploration_parameter)

        return [env, agent, memory, exploration_strategy, exploration_parameter]


def ddqn_general_ddqn_greedy_config(env_name, num_models):

        # Replay memory
        replay_size = 100000
        sample_size = 32

        # DQN
        structure = [256, 256]
        agent_config = {
                'discount': 0.99,
                'learning_rate': 0.00025,
                'target_offset': 10000,
        }

        env = GeneralOpenAIEnvironment(num_models, env_name)
        memory = ExperienceReplayMemory(num_models, replay_size, sample_size, env)
        agent = DDQNAgent(num_models, env, structure, agent_config)

        exploration_strategy = GreedyPolicy(num_models)
        return [env, agent, memory, exploration_strategy, tf.constant(0.0)]