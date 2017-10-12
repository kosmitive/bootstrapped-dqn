import datetime
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from manager.DirectoryManager import DirectoryManager
from plots.RewardValueFunctionPlot import RewardValueFunctionPlot
from scripts.configurations import ddqn_general_ddqn_eps_config
from scripts.configurations import ddqn_general_ddqn_greedy_config
from scripts.connector import build_general_configuration

# the run settings
name = "MountainCar-v0"
run_folder = "./run/"

batch = [["ddqn_eps", ddqn_general_ddqn_eps_config, "Epsilon-Greedy", 35],
         ["ddqn_greedy", ddqn_general_ddqn_greedy_config, "Greedy", 36]]

# the settings for the framework
epochs = 100

save_epoch = 100
save_plot = False
save_best = False
num_models = 5

plot_as_variance = False
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
    dir_man = DirectoryManager(run_folder, name, "{}_{}".format(agent_name, seed))

    # create the reward graph
    rewards = np.zeros((num_models, epochs))

    # create new graph and session, respectively
    graph = tf.Graph()
    with graph.as_default():
        tf_config = tf.ConfigProto(log_device_placement=True)
        #tf_config.intra_op_parallelism_threads = 8
        #tf_config.inter_op_parallelism_threads = 8
        #tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        tf_config.gpu_options.allow_growth=True
        with tf.Session(graph=graph, config=tf_config) as sess:
            #os.environ['LD_LIBRARY_PATH'] = "/opt/cuda"
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            with tf.variable_scope("main"):
                tf.set_random_seed(seed)
                np.random.seed(seed + 1)

                # create combined model
                [env, agents, memories, policies, params], conf = config(name, epochs * 200, num_models)
                feedback = build_general_configuration(env, agents, memories, policies)

                # get copy graph
                copy = [agents[k].network.copy_graph('dqn', 'target') for k in range(num_models)]
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
                        q_functions = agents[0].network.grid_graph([grid_granularity, grid_granularity], env.observation_space().intervals, scope='dqn')
                        t_q_functions = agents[0].network.grid_graph([grid_granularity, grid_granularity], env.observation_space().intervals, scope='dqn')

                # Do a random walk for each model
                print(line_size * "=")
                print("Filling Replay Memory...")
                print(line_size * "=")
                rand_walks = env.random_walk(memories[0].size)

                # create fill actions and fill up all memories
                fill_actions = list()
                for i in range(len(rand_walks)):
                    fill_actions.append(memories[i].fill_graph(*rand_walks[i]))
                sess.run(fill_actions)

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

                        sess.run(copy_models)

                        cobs, result = sess.run([env.current_observation_graph(), feedback], options=options, run_metadata=run_metadata)
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

                        if save_plot and epoch % save_epoch == 0:
                            dir_man.save_plot(rew_va_plt, epoch)

                tr_end = time.time()
                print(line_size * "=")
                print("Training took {} ms".format(tr_end - tr_start))

                # save the rewards on each step
                dir_man.save_plot(rew_va_plt, epochs)
                dir_man.save_rewards(rewards)

        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_01.json', 'w') as f:
            f.write(chrome_trace)

    if plot_interactive: rew_va_plt.show()