import datetime
import time

import numpy as np
import tensorflow as tf

from manager.DirectoryManager import DirectoryManager
from plots.RewardValueFunctionPlot import RewardValueFunctionPlot
from scripts.configurations import ddqn_general_ddqn_eps_config
from scripts.configurations import ddqn_general_ddqn_greedy_config
from scripts.configurations import ddqn_general_ddqn_zoneout_config
from scripts.configurations import ddqn_general_ddqn_shakeout_config
from scripts.configurations import ddqn_general_ddqn_dropout_config
from scripts.connector import build_general_configuration

# the run settings
name = "MountainCar-v0"
run_folder = "./run/"

save_num = 1

batch = [["ddqn_shakeout", ddqn_general_ddqn_zoneout_config, "Shakeout", 15]]
         #["ddqn_dropout", ddqn_general_ddqn_dropout_config, "Dropout", 15],
         #["ddqn_zoneout", ddqn_general_ddqn_zoneout_config, "Zoneout", 15]]

# the settings for the framework
epochs = 2500

save_epoch = 100
save_plot = True
save_best = True
num_models = 5

plot_as_variance = True
num_cpu = 16
seed = 12

# change display settings, note that these only apply
# to the first model created internally.
render = False
plot_interactive = False
grid_granularity = 500
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
    dir_man = DirectoryManager(run_folder, name, "{}_{}_{}".format(agent_name, seed, save_num))

    # create the reward graph
    rewards = np.zeros((num_models, epochs))

    # create new graph and session, respectively
    graph = tf.Graph()
    with graph.as_default():
        tf_config = tf.ConfigProto(
            log_device_placement=True,
            inter_op_parallelism_threads=16,
            intra_op_parallelism_threads=16)

        tf_config.gpu_options.allow_growth=True
        with tf.Session(graph=graph, config=tf_config) as sess:
            with tf.variable_scope("main"):
                tf.set_random_seed(seed)
                np.random.seed(seed + 1)

                # create combined model
                [env, agents, memories, policies, params], conf = config(name, epochs * 200, num_models)
                feedback = build_general_configuration(env, agents, memories, policies)

                # get copy graph
                copy = [agents[k].copy_graph() for k in range(num_models)]
                is2d = env.observation_space().dim() == 2

                # init the graph
                init_op = tf.global_variables_initializer()
                sess.run(init_op)

                # reset the environment
                env_res_op = env.reset_graph()
                sess.run(env_res_op)

                # obtain a grid for the q-function of the agent
                if (save_best or save_plot or plot_interactive) and is2d:
                    with tf.variable_scope("0"):
                        agents[0].network.switch('dqn')
                        q_functions = agents[0].network.grid_graph([grid_granularity, grid_granularity], env.observation_space().IB, env.observation_space().IE)
                        agents[0].network.switch('target')
                        t_q_functions = agents[0].network.grid_graph([grid_granularity, grid_granularity], env.observation_space().IB, env.observation_space().IE)

                # Fill the replay memory
                print(line_size * "=")
                print("Filling Replay Memory...")
                print(line_size * "=")
                for memory in memories:
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
                dir_man.save_config(conf)

                # Repeat for the number of epochs
                for epoch in range(epochs):
                    res = sess.run(env_res_op)

                    if is2d:
                        obs_buffer = np.empty([201, 2], dtype=np.float32)
                        obs_buffer[0, :] = res[0, :]

                    episode_finished = False
                    ep_start = time.time()
                    step_count = 0
                    gs = sess.run([agents[k].global_step for k in range(num_models)])
                    pers_dones = num_models * [False]

                    if (plot_interactive or (save_plot and epoch % save_epoch == 0) or (save_best and np.mean(rewards, 0)[epoch] > best_reward)) and is2d:
                            [q_values, t_q_values] = sess.run([q_functions, t_q_functions])

                    # Repeat until the environment says it's done
                    while not episode_finished:

                        # copy the models which reached the target offset
                        copy_models = list()
                        for k in range(num_models):
                            if gs[k] % conf['target_offset'] == 0:
                                copy_models.append(copy[k])
                            gs[k] += 1
                        sess.run(copy_models)

                        cobs, result = sess.run([env.current_observation_graph(), feedback])
                        step_count += 1
                        for k in range(num_models):
                            if not pers_dones[k]: rewards[k, epoch] += result[1][k]
                            pers_dones[k] = result[2][k]

                        if is2d:
                            obs_buffer[step_count, :] = cobs[0, :]

                        # Generally the output stuff
                        if render: env.render(1)

                        # finish episode if necessary
                        if all(result[2]):
                            ep_diff = round((time.time() - ep_start) * 1000, 2)
                            eps = sess.run([params[0]])
                            print("\tEpisode {} took {} ms with {} steps and {} rewards using exploration of {}".format(epoch, ep_diff, step_count, np.mean(rewards, 0)[epoch], eps))
                            break

                    if plot_interactive or (save_plot and epoch % save_epoch == 0) or (save_best and np.mean(rewards, 0)[epoch] > best_reward):
                        if is2d:
                            rew_va_plt.update(rewards[:, :epoch+1], [q_value[:, :] for q_value in q_values],
                                             [t_q_value[:, :] for t_q_value in t_q_values],
                                              plot_as_variance, obs_buffer[:step_count+1])
                        else:
                            rew_va_plt.update(rewards[:, :epoch+1], None, None, plot_as_variance, None)

                        if save_best and np.mean(rewards, 0)[epoch] > best_reward:
                            best_reward = np.mean(rewards, 0)[epoch]
                            dir_man.save_plot(rew_va_plt, epoch, "best_rew_va.eps")
                            dir_man.save_q_func(q_values, epoch, "best_q_func.npy")

                        if save_plot and epoch % save_epoch == 0:
                            dir_man.save_plot(rew_va_plt, epoch)
                            dir_man.save_q_func(q_values, epoch)

                tr_end = time.time()
                print(line_size * "=")
                print("Training took {} ms".format(tr_end - tr_start))

                # save the rewards on each step
                dir_man.save_plot(rew_va_plt, epochs)
                dir_man.save_rewards(rewards)

    if plot_interactive: rew_va_plt.show()