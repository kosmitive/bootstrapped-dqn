import tensorflow as tf

from agents.Agent import Agent
from memory.Memory import Memory
from environments.Environment import Environment
from policies.Policy import Policy
from policies.RandomPolicy import RandomPolicy

def build_general_configuration(env, agent, memory, policy):


    # build the final graph
    current_observation = env.current_observation

    # obtain basically the best action
    agent.register_policy(policy)
    action = agent.action_graph(current_observation)

    # execute action
    next_observation, reward, done = env.step_graph(action)

    # create the learn graph
    agent.register_memory(memory)
    minimizer = agent.observe_graph(current_observation, next_observation, action, reward, done)

    # execute the minimizer before "applying" the action
    with tf.control_dependencies([minimizer]):
        one_step = env.apply_step_graph()

    # additionally create an op for filling the memory
    random_action = RandomPolicy().choose_action(env.action_space().n)
    rand_next_observation, rand_reward, rand_done = env.step_graph(random_action)
    count = memory.store_graph(current_observation, rand_next_observation, random_action, rand_reward, rand_done)

    with tf.control_dependencies([env.apply_step_graph()]):
        fill_step = tf.identity(count)

    feedback = [one_step, done, reward]
    return feedback, fill_step