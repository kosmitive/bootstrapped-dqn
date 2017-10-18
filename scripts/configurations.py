from agents.DDQNAgent import DDQNAgent
from environments.GeneralOpenAIEnvironment import GeneralOpenAIEnvironment
from memory.ExperienceReplayMemory import ExperienceReplayMemory
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from policies.GreedyPolicy import GreedyPolicy

import extensions.tensorflowHelpers as tfh
import tensorflow as tf

def ddqn_general_ddqn_eps_config(env_name, max_timesteps, num_models):

        # DQN
        config = {
                'structure': [256, 256],

                'replay_size': 50000,
                'sample_size': 128,

                'discount': 0.99,
                'learning_rate': 0.00075,
                'target_offset': 500,

                'policy': 'eps_greedy',
                'exp_rate_decay': 'linear',
                'exp_rate_begin': 1.0,
                'exp_rate_end': 0.1,
                'decay_lambda': 0.1
        }

        env = GeneralOpenAIEnvironment(env_name, num_models)
        memories = [ExperienceReplayMemory(env, config) for _ in range(num_models)]
        agents = [DDQNAgent(env, config) for _ in range(num_models)]

        # obtain an exploration strategy
        exploration_parameters = [tfh.linear_decay(int(config['decay_lambda'] * 100000),
                                                   config['exp_rate_begin'],
                                                   config['exp_rate_end'], agents[k].global_step) for k in range(num_models)]

        exploration_strategies = [EpsilonGreedyPolicy(env.action_space(), exploration_parameters[k]) for k in range(num_models)]
        return [env, agents, memories, exploration_strategies, exploration_parameters], config


def ddqn_general_ddqn_zoneout_config(env_name, max_timesteps, num_models):

        # Replay memory
        config = {
            'structure': [256, 256],
            'replay_size': 50000,
            'sample_size': 128,

            'discount': 0.99,
            'learning_rate': 0.00025,
            'target_offset': 500,

            'regularization': 'zoneout',
            'reg_rate_increase': 'linear',
            'reg_rate_begin': 0.90,
            'reg_rate_end': 1.0,
            'reg_steps_red': 300000
        }

        env = GeneralOpenAIEnvironment(env_name, num_models)
        memories = [ExperienceReplayMemory(env, config) for _ in range(num_models)]
        agents = [DDQNAgent(env, config) for _ in range(num_models)]
        exploration_strategies = [GreedyPolicy() for k in range(num_models)]
        exploration_parameters = [agent.mask_value for agent in agents]
        return [env, agents, memories, exploration_strategies, exploration_parameters], config


def ddqn_general_ddqn_shakeout_config(env_name, max_timesteps, num_models):
    # DQN
    # Replay memory
    config = {
        'structure': [256, 256],
        'replay_size': 50000,
        'sample_size': 128,

        'discount': 0.99,
        'learning_rate': 0.00025,
        'target_offset': 500,

        'regularization': 'shakeout',
        'reg_rate_increase': 'linear',
        'reg_rate_begin': 0.8,
        'reg_rate_end': 1.0,
        'reg_steps_red': 100000
    }

    env = GeneralOpenAIEnvironment(env_name, num_models)
    memories = [ExperienceReplayMemory(env, config) for _ in range(num_models)]
    agents = [DDQNAgent(env, config) for _ in range(num_models)]
    exploration_strategies = [GreedyPolicy() for _ in range(num_models)]
    exploration_parameters = [agent.mask_value for agent in agents]
    return [env, agents, memories, exploration_strategies, exploration_parameters], config


def ddqn_general_ddqn_dropout_config(env_name, max_timesteps, num_models):

        # DQN
        # Replay memory
        config = {
            'structure': [256, 256],
            'replay_size': 50000,
            'sample_size': 128,

            'discount': 0.99,
            'learning_rate': 0.00025,
            'target_offset': 500,

            'regularization': 'dropout',
            'reg_rate_increase': 'linear',
            'reg_rate_begin': 0.8,
            'reg_rate_end': 1.0,
            'reg_steps_red': 300000
        }

        env = GeneralOpenAIEnvironment(env_name, num_models)
        memories = [ExperienceReplayMemory(env, config) for _ in range(num_models)]
        agents = [DDQNAgent(env, config) for _ in range(num_models)]
        exploration_strategies = [GreedyPolicy() for k in range(num_models)]
        exploration_parameters = [agent.mask_value for agent in agents]
        return [env, agents, memories, exploration_strategies, exploration_parameters], config


def ddqn_general_ddqn_greedy_config(env_name, max_timesteps, num_models):

        # Replay memory
        replay_size = 50000
        sample_size = 64

        # DQN
        structure = [128, 128]
        agent_config = {
            'discount': 0.99,
            'learning_rate': 0.00025,
            'target_offset': 500,
        }

        env = GeneralOpenAIEnvironment(env_name, num_models)
        memories = [ExperienceReplayMemory(replay_size, sample_size, env) for _ in range(num_models)]
        agents = [DDQNAgent(env, structure, agent_config) for _ in range(num_models)]

        exploration_strategy = [GreedyPolicy() for k in range(num_models)]
        return [env, agents, memories, exploration_strategy, [tf.constant(0.0)]], agent_config