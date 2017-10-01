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
from scripts.connector import build_general_configuration


# the run settings
name = "MountainCar-v0"
run_folder = "./run/"

batch = [["ddqn_eps_base", ddqn_general_ddqn_eps_config, "Epsilon-Greedy"]]

# the settings for the framework
seed = 31
epochs = 2500
save_epoch = 100
save_plot = True
num_models = 1
plot_as_variance = False

# change display settings, note that these only apply
# to the first model created internally.
render = True
plot_interactive = True
grid_granularity = 500
tf.set_random_seed(seed)

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

            # create combined model
            [env, agent, memory, policy] = ddqn_general_ddqn_eps_config(name, num_models)
            feedback = build_general_configuration(env, agent, memory, policy)

            # init the graph
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # reset the environment
            env_res_op = env.reset_graph()
            sess.run(env_res_op)

            # obtain a grid for the q-function of the agent
            if save_plot or plot_interactive:
                q_functions = agent.dqn.grid_graph([grid_granularity, grid_granularity])

            # Fill the replay memory
            print(line_size * "=")
            print("Filling Replay Memory...")
            print(line_size * "=")
            vals = env.random_walk(memory.size)
            sess.run(memory.fill_graph(*vals))

            # create the plot object
            if plot_interactive or save_plot:
                rew_va_plt = RewardValueFunctionPlot(titlename, 3, env.observation_space())
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
                step_count = 0

                # Repeat until the environment says it's done
                while not episode_finished:
                    result = sess.run(feedback)
                    step_count += 1
                    rewards[:, epoch] += result[1]

                    # Generally the output stuff
                    if render: env.render(1)

                    # finish episode if necessary
                    if any(result[2]):
                        ep_diff = round((time.time() - ep_start) * 1000, 2)
                        print("\tEpisode {} took {} ms and finished with {} steps and rewards {}".format(epoch, ep_diff, step_count, rewards[:, epoch]))
                        break

                if plot_interactive or (save_plot and epoch % save_epoch == 0):
                    q_values = sess.run(q_functions)
                    rew_va_plt.update(rewards[:, :epoch+1], [q_value[0, :, :] for q_value in q_values], plot_as_variance)

                    if save_plot and epoch % save_epoch == 0:
                        dir_man.save_plot(rew_va_plt, epoch)

                # save the rewards on each step
                dir_man.save_rewards(rewards[:, :epoch+1])

            tr_end = time.time()
            print(line_size * "=")
            print("Training took {} ms".format(tr_end - tr_start))

    if plot_interactive: rew_va_plt.show()