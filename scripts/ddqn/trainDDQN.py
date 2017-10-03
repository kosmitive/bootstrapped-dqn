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

batch = [["ddqn_eps", ddqn_general_ddqn_eps_config, "Epsilon-Greedy", 13],
         ["ddqn_eps", ddqn_general_ddqn_eps_config, "Epsilon-Greedy", 25],
         ["ddqn_eps", ddqn_general_ddqn_eps_config, "Epsilon-Greedy", 38],
         ["ddqn_greedy", ddqn_general_ddqn_greedy_config, "Greedy", 13],
         ["ddqn_greedy", ddqn_general_ddqn_greedy_config, "Greedy", 25],
         ["ddqn_greedy", ddqn_general_ddqn_greedy_config, "Greedy", 38]]

# the settings for the framework
epochs = 5000
save_epoch = 10
save_plot = True
num_models = 1
plot_as_variance = False

# change display settings, note that these only apply
# to the first model created internally.
render = False
plot_interactive = False
grid_granularity = 250

# ---------------------------------------------------------------

for [agent_name, config, suffix, seed] in batch:

    # set the seeds for this batch
    tf.set_random_seed(seed)
    np.random.seed(seed)

    titlename = "Double Deep-Q-Networks {}".format(suffix)
    line_size = 80
    print(line_size * "=")

    # first of all join the names
    dir_man = DirectoryManager(run_folder, name, "{}_{}".format(agent_name, seed))

    # create the reward graph
    rewards = np.zeros((num_models, epochs))

    # create new graph and session, respectively
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as sess:

            # create combined model
            [env, agent, memory, policy, param] = config(name, num_models)
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
                t_q_functions = agent.target_dqn.grid_graph([grid_granularity, grid_granularity])

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

                obs_buffer = np.empty([51, 2], dtype=np.float32)
                obs_buffer[0, :] = sess.run(env_res_op)
                episode_finished = False
                ep_start = time.time()
                step_count = 0

                if epoch % 50 == 0:
                    sess.run(agent.target_dqn.copy_graph(agent.dqn))

                # Repeat until the environment says it's done
                while not episode_finished:
                    [cobs, result] = sess.run([env.current_observation_graph(), feedback])
                    step_count += 1
                    rewards[:, epoch] += result[1]
                    obs_buffer[step_count, :] = cobs

                    # Generally the output stuff
                    if render: env.render(1)

                    # finish episode if necessary
                    if all(result[2]):
                        ep_diff = round((time.time() - ep_start) * 1000, 2)
                        eps = sess.run([param])
                        print("\tEpisode {} took {} ms with {} steps and {} rewards using exploration of {}".format(epoch, ep_diff, step_count, rewards[:, epoch], eps))
                        break

                if plot_interactive or (save_plot and epoch % save_epoch == 0):
                    [q_values, t_q_values] = sess.run([q_functions, t_q_functions])
                    rew_va_plt.update(rewards[:, :epoch+1], [q_value[0, :, :] for q_value in q_values],
                                      [t_q_value[0, :, :] for t_q_value in t_q_values],
                                      plot_as_variance, obs_buffer)

                    if save_plot and epoch % save_epoch == 0:
                        dir_man.save_plot(rew_va_plt, epoch)

                # save the rewards on each step
                dir_man.save_rewards(rewards[:, :epoch+1])

            tr_end = time.time()
            print(line_size * "=")
            print("Training took {} ms".format(tr_end - tr_start))

    if plot_interactive: rew_va_plt.show()