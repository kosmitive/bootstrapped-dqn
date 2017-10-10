import tensorflow as tf

from agents.Agent import Agent
from memory.Memory import Memory
from environments.Environment import Environment
from policies.Policy import Policy
from policies.RandomPolicy import RandomPolicy

def build_general_configuration(env, agent, memory, policy):

    # build the final graph
    current_observation = env.current_observation_graph()

    # obtain basically the best action
    agent.register_policy(policy)
    sample = agent.network.create_mask_graph(0.85)
    action = agent.action_graph(current_observation)

    # execute action
    next_observation, rewards, dones = env.step_graph(action)

    # create the learn graph
    agent.register_memory(memory)
    minimizer = agent.observe_graph(current_observation, next_observation, action, rewards, dones)

    # execute the minimizer before "applying" the action
    with tf.control_dependencies([minimizer]):
        one_step = env.apply_step_graph()
    feedback = one_step, rewards, dones
    return feedback, sample