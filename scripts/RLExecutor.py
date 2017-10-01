import datetime
import time

import numpy as np
import tensorflow as tf

from plots.RewardValueFunctionPlot import RewardValueFunctionPlot
from utils.DirectoryManager import DirectoryManager

from scripts.configurations import ddqn_general_ddqn_eps_config
from scripts.configurations import ddqn_general_ddqn_greedy_config
from scripts.connector import build_general_configuration
from scripts import Workpackage


class RLExecutor:

    def __init__(self, run_folder, environments, agents):

        # store the environments and agents
        self.environments = environments
        self.agents = agents
        self.run_folder = run_folder

    # ---------------------------------------------------------------

    def execute(self, workpackage):
        assert isinstance(workpackage, Workpackage)

        env_results = list()
        agent_results = list()

        # repeat for all
        for env in self.environments:
            for agent in self.agents:

                # first of all join the names
                dir_man = DirectoryManager(self.run_folder, env.name, agent.name)

                # create the reward   if plot_interactive or save_plot:
                rew_va_plt = RewardValueFunctionPlot(titlename, 3, env_list[0].env.observation_space)
            if plot_interactive:
                rew_va_plt.interactive()graph
                rewards = np.zeros((num_models, epochs))

                # create new graph and session, respectively
                graph = tf.Graph()
                with graph.as_default():
                    with tf.Session(graph=graph) as sess:

                        # init the graph
                        init_op = tf.global_variables_initializer()
                        sess.run(init_op)

                        # reset the environment
                        env_res_op = [env.reset_graph() for env in env_list]
                        sess.run(env_res_op)

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
                    feedback = sess.run(feedback_list)
                    rewards[:, epoch] += [el[2] for el in feedback]

                    # Generally the output stuff
                    if render: [env.render() for env in env_list]

                    # finish episode if necessary
                    if feedback[0][1]:
                        ep_diff = round((time.time() - ep_start) * 1000, 2)
                        print("\tEpisode {} took {} ms and finished with new rewards {}".format(epoch, ep_diff, rewards[:, epoch]))
                        break

                if plot_interactive or (save_plot and epoch % save_epoch == 0):
                    q_values = sess.run(q_functions)
                    rew_va_plt.update(rewards[:, :epoch+1], q_values, plot_as_variance)

                    if save_plot and epoch % save_epoch == 0:
                        dir_man.save_plot(rew_va_plt, epoch)

                # save the rewards on each step
                dir_man.save_rewards(rewards[:, :epoch+1])

            tr_end = time.time()
            print(line_size * "=")
            print("Training took {} ms".format(tr_end - tr_start))

    if plot_interactive: rew_va_plt.show()