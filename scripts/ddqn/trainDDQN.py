import datetime
import time

import numpy as np
import tensorflow as tf

from manager.DirectoryManager import DirectoryManager
from plots.RewardValueFunctionPlot import RewardValueFunctionPlot
from scripts.configurations import ddqn_general_ddqn_eps_config
from scripts.configurations import ddqn_general_ddqn_greedy_config
from scripts.connector import build_general_configuration

# the run settings
name = "MountainCar-v0"
run_folder = "./run/"

batch = [["ddqn_eps", ddqn_general_ddqn_eps_config, "Epsilon-Greedy", 15],
         ["ddqn_eps", ddqn_general_ddqn_eps_config, "Epsilon-Greedy", 25],
         ["ddqn_eps", ddqn_general_ddqn_eps_config, "Epsilon-Greedy", 38],
         ["ddqn_greedy", ddqn_general_ddqn_greedy_config, "Greedy", 13],
         ["ddqn_greedy", ddqn_general_ddqn_greedy_config, "Greedy", 25],
         ["ddqn_greedy", ddqn_general_ddqn_greedy_config, "Greedy", 38]]

# the settings for the framework
epochs = 500000
save_epoch = 50
save_plot = True
save_best = True
num_models = 1
plot_as_variance = False
num_cpu = 16
seed = 12
# change display settings, note that these only apply
# to the first model created internally.
render = True
plot_interactive = True
grid_granularity = 300
best_reward = -300
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
    rewards = np.zeros((epochs))

    # create new graph and session, respectively
    graph = tf.Graph()
    with graph.as_default():
        tf_config = tf.ConfigProto(log_device_placement=True)
        tf_config.gpu_options.allow_growth=True
        with tf.Session(graph=graph, config=tf_config) as sess:
            with tf.variable_scope("main"):
                tf.set_random_seed(seed)
                np.random.seed(seed + 1)

                # create combined model
                [env, agent, memory, policy, param], conf = config(name, epochs * 200)
                feedback, sample_mask = build_general_configuration(env, agent, memory, policy)

                # get copy graph
                copy = agent.copy_graph()
                is2d = env.observation_space().dim() == 2

                # init the graph
                init_op = tf.global_variables_initializer()
                sess.run(init_op)

                # reset the environment
                env_res_op = env.reset_graph()
                sess.run(env_res_op)

                # obtain a grid for the q-function of the agent
                if (save_best or save_plot or plot_interactive) and is2d:
                    agent.network.switch('dqn')
                    q_functions = agent.network.grid_graph([grid_granularity, grid_granularity], env.observation_space().IB, env.observation_space().IE)
                    agent.network.switch('target')
                    t_q_functions = agent.network.grid_graph([grid_granularity, grid_granularity], env.observation_space().IB, env.observation_space().IE)

                # Fill the replay memory
                print(line_size * "=")
                print("Filling Replay Memory...")
                print(line_size * "=")
                vals = env.random_walk(memory.size)
                sess.run(memory.fill_graph(*vals))

                # create the plot object
                if plot_interactive or save_plot or save_best:
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
                    res, _ = sess.run([env_res_op, sample_mask])

                    if is2d:
                        obs_buffer = np.empty([201, 2], dtype=np.float32)
                        obs_buffer[0, :] = res

                    episode_finished = False
                    ep_start = time.time()
                    step_count = 0
                    gs = sess.run(agent.global_step)

                    if (plot_interactive or (save_plot and epoch % save_epoch == 0) or (save_best and rewards[epoch] > best_reward)) and is2d:
                            [q_values, t_q_values] = sess.run([q_functions, t_q_functions])

                    # Repeat until the environment says it's done
                    while not episode_finished:
                        if gs % conf['target_offset'] == 0: sess.run(copy)
                        gs += 1
                        cobs, result = sess.run([env.current_observation_graph(), feedback])
                        step_count += 1
                        rewards[epoch] += result[1]

                        if is2d:
                            obs_buffer[step_count, :] = cobs

                        # Generally the output stuff
                        if render: env.render()

                        # finish episode if necessary
                        if result[2]:
                            ep_diff = round((time.time() - ep_start) * 1000, 2)
                            eps = sess.run([param])
                            print("\tEpisode {} took {} ms with {} steps and {} rewards using exploration of {}".format(epoch, ep_diff, step_count, rewards[epoch], eps))
                            break

                    if plot_interactive or (save_plot and epoch % save_epoch == 0) or (save_best and rewards[epoch] > best_reward):
                        if is2d:
                            rew_va_plt.update(rewards[:epoch+1], [q_value[:, :] for q_value in q_values],
                                             [t_q_value[:, :] for t_q_value in t_q_values],
                                              plot_as_variance, obs_buffer[:step_count+1])
                        else:
                            rew_va_plt.update(rewards[:epoch+1], None, None, plot_as_variance, None)

                        if save_best and rewards[epoch] > best_reward:
                            best_reward = rewards[epoch]
                            dir_man.save_plot(rew_va_plt, epoch, "best_rew_va.eps")

                        if save_plot and epoch % save_epoch == 0:
                            dir_man.save_plot(rew_va_plt, epoch)

                tr_end = time.time()
                print(line_size * "=")
                print("Training took {} ms".format(tr_end - tr_start))

                # save the rewards on each step
                dir_man.save_rewards(rewards)

    if plot_interactive: rew_va_plt.show()