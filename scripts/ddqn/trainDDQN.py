import datetime
import time

import numpy as np
import tensorflow as tf

import extensions.tensorflowHelpers as tfh
from agents.DDQNAgent import DDQNAgent
from environments.GeneralOpenAIEnvironment import GeneralOpenAIEnvironment
from memory.ExperienceReplayMemory import ExperienceReplayMemory
from plots.RewardValueFunctionPlot import RewardValueFunctionPlot
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from utils.DirectoryManager import DirectoryManager

from scripts.configurations import ddqn_general_ddqn_eps_config
from scripts.configurations import ddqn_general_ddqn_greedy_config
from scripts.connector import build_general_configuration


# the run settings
name = "MountainCar-v0"
run_folder = "./run/"

batch = [["ddqn_eps_base", ddqn_general_ddqn_eps_config, "Epsilon-Greedy"],
         ["ddqn_greedy_base", ddqn_general_ddqn_greedy_config, "Greedy"]]

# the settings for the framework
seed = 28
epochs = 100000
save_epoch = 500
save_plot = True
num_models = 50

# change display settings, note that these only apply
# to the first model created internally.
render = False
plot_interactive = False
grid_granularity = 300

# ---------------------------------------------------------------

for [agent_name, config, suffix] in batch:

    titlename = "Double Deep-Q-Networks {}".format(suffix)
    line_size = 80
    print(line_size * "=")

    # first of all join the names
    dir_man = DirectoryManager(run_folder, name, agent_name)

    # create the reward graph
    rewards = np.zeros((num_models, epochs))

    # create new graph and session, respectively
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as sess:

            # each model gets saved in its own list entry
            env_list = list()
            agent_list = list()
            feedback_list = list()
            fill_step_list = list()
            for m in range(num_models):
                with tf.variable_scope("model_{}".format(m)):

                    seed = seed + 1
                    tf.set_random_seed(seed)
                    [env, agent, memory, policy] = ddqn_general_ddqn_eps_config(name)
                    feedback, fill_step = build_general_configuration(env, agent, memory, policy)
                    env_list.append(env)
                    agent_list.append(agent)
                    feedback_list.append(feedback)
                    fill_step_list.append(fill_step)

            # init the graph
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # create commonly used dicts
            eval_dict = {agent.learn: False for agent in agent_list}
            train_dict = {agent.learn: True for agent in agent_list}

            # reset the environment
            env_res_op = [env.reset_graph() for env in env_list]
            sess.run(env_res_op)

            # obtain a grid for the q-function of the agent
            q_functions = agent_list[0].dqn.grid_graph([grid_granularity, grid_granularity])

            # Fill the replay memory
            print(line_size * "=")
            print("Filling Replay Memory...")
            print(line_size * "=")
            i = sess.run(fill_step_list)[0]
            while i > 0:
                print(i)
                i = sess.run(fill_step_list)[0]

            # create the plot object
            if plot_interactive or save_plot:
                rew_va_plt = RewardValueFunctionPlot(titlename, 3, env_list[0].env.observation_space)
            if plot_interactive:
                rew_va_plt.interactive()

            # start the training
            tr_start = time.time()
            print(line_size * "=")
            print("Training started at {}".format(datetime.datetime.now()))
            print(line_size * "=")

            # Repeat for the number of epochs
            for epoch in range(epochs):

                sess.run(env_res_op)
                episode_finished = False
                ep_start = time.time()

                # Repeat until the environment says it's done
                while not episode_finished:
                    feedback = sess.run(feedback_list, train_dict)
                    rewards[:, epoch] += [el[2] for el in feedback]

                    # Generally the output stuff
                    if render: env_list[0].render()

                    # finish episode if necessary
                    if feedback[0][1]:
                        ep_diff = round((time.time() - ep_start) * 1000, 2)
                        print("\tEpisode {} took {} ms and finished with mean reward of {}".format(epoch, ep_diff, np.mean(rewards[:, epoch])))
                        break

                if plot_interactive or (save_plot and epoch % save_epoch == 0):
                    q_values = sess.run(q_functions)
                    rew_va_plt.update(rewards[:, :epoch], q_values)

                    if save_plot and epoch % save_epoch == 0:
                        dir_man.save_plot(rew_va_plt, epoch)

                # save the rewards on each step
                dir_man.save_rewards(rewards[:, :epoch])

            tr_end = time.time()
            print(line_size * "=")
            print("Training took {} ms".format(tr_end - tr_start))

    if plot_interactive: rew_va_plt.show()