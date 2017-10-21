import tensorflow as tf
import extensions.tensorflowHelpers as tfh

from agents.Agent import Agent
from memory.Memory import Memory
from environments.Environment import Environment
from policies.Policy import Policy
from policies.RandomPolicy import RandomPolicy

def build_general_configuration(env, agents, memories, policies):

    num_models = len(agents)

    # build the final graph
    current_observation = env.current_observation_graph()

    # obtain basically the best action
    [agents[k].register_policy(policies[k]) for k in range(num_models)]

    actionlst = list()
    info_bonus = list()
    for k in range(num_models):
        with tf.variable_scope(str(k)):
            act, info = agents[k].action_graph(current_observation[k, :])
            actionlst.append(act)
            info_bonus.append(info)

    actions = tf.stack(actionlst, 0)

    # execute action
    next_observation, trewards, dones, real_dones = env.step_graph(actions)

    rewards = list()
    for k in range(num_models):
        rewards.append(trewards[k] + 0.1 * info_bonus[k])

    # create the learn graph
    minimizer = list()
    cond_minimizer = list()
    for k in range(num_models):
        with tf.variable_scope(str(k)):
            agents[k].register_memory(memories[k])
            minimizer.append(agents[k].observe_graph(current_observation[k, :], next_observation[k, :], actions[k], rewards[k], dones[k]))
            cond_minimizer.append(tf.cond(tf.cast(real_dones[k], tf.bool), lambda: tf.no_op(), lambda: minimizer[k]))

    # execute the minimizer before "applying" the action
    with tf.control_dependencies(cond_minimizer):
        one_step = env.apply_step_graph()
    feedback = one_step, rewards, dones
    return feedback