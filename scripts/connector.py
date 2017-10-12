import tensorflow as tf
import extensions.tensorflow_extensions as tfh

from agents.GeneralAgent import GeneralAgent
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
    for k in range(num_models):
        with tf.variable_scope(str(k)):
            actionlst.append(agents[k].action_graph(current_observation[k, :]))

    actions = tf.stack(actionlst, 0)

    # execute action
    next_observation, rewards, dones, real_dones = env.step_graph(actions)

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