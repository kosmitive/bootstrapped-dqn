import tensorflow as tf
from environments.Environment import Environment
from memory.Memory import Memory
from policies.Policy import Policy
from agents.RLAgent import RLAgent


class GraphGenerator:
    """This class can be used to create a graph."""

    def __init__(self, N, env):
        assert isinstance(env, Environment)
        self.env = env
        self.N = N

    def register_memory(self, memory):
        assert memory in ['exp_replay', 'none']
        self.memory = memory

    def register_policy(self, policy):
        assert isinstance(policy, Policy)
        self.policy = policy

    def register_agent(self, agent):
        assert isinstance(agent, RLAgent)
        self.agent = agent

    def build_model(self):

        # builder the final graph
        actions = list()
        current_states = list()

        for m in range(self.N):
            with tf.variable_scope("model_{}".format(m)):
                current_observation = self.env.current_states[m]
                act_eval_graph = self.agent.action_eval_graph(current_observation)
                actions.append(self.policy.choose_action(act_eval_graph))
                current_states.append(current_observation)

        # execute the actions
        actions = tf.stack(actions)
        rews, next_states, dones = self.env.experience_graph(actions)

        # unstack the samples againg
        rews = tf.unstack(rews)
        next_states = tf.unstack(next_states)
        actions = tf.unstack(actions)
        dones = tf.unstack(dones)

        # split them up again
        for m in range(self.N):
            with tf.variable_scope("model_{}".format(m)):
                exp_tuple = (current_states[m], next_states[m], actions[m], rews[m], dones[m])
                available_sampling = (exp_tuple if self.memory is None else self.memory.store_and_sample_graph(*exp_tuple))
                learn = self.agent.learn_graph(*)

        # create the learn graph
        minimizer = list()
        cond_minimizer = list()
        for k in range(num_models):
            with tf.variable_scope(str(k)):
                agents[k].register_memory(memories[k])
                minimizer.append(agents[k].experience_graph(current_observation[k, :], next_observation[k, :], actions[k], rewards[k], dones[k]))
                cond_minimizer.append(tf.cond(tf.cast(real_dones[k], tf.bool), lambda: tf.no_op(), lambda: minimizer[k]))

        # execute the minimizer before "applying" the action
        with tf.control_dependencies(cond_minimizer):
            one_step = env.apply_step_graph()
        feedback = one_step, rewards, dones
        return feedback