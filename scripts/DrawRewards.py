import matplotlib.pyplot as plt
import numpy as np
import os

# create base folders
base_folder = '/home/markus/Git/BT/Experiments/MountainCar-v0/'
infixes = ['shakeout', 'zoneout', 'dropout']
binfixes = ['Shakeout', 'Zoneout', 'Dropout']
merge_folders = ['ddqn_{}_15_1', 'ddqn_{}_15_2']
plot_rewards = False
plot_q_funcs = True
max_episode = 2500
save = True
colors = [["#215ab7", "#215ab786"], ["#b70000", "#b7000045"]]

out_folder = '/home/markus/Git/BT/Thesis/img/Evaluations/mountaincar'
file_name = 'mc_{}_15_1_2.eps'
q_func_name = 'value_func_mc_{}.eps'

if plot_rewards:
    for i in range(len(infixes)):
        infix = infixes[i]

        # create figure
        fig = plt.figure("rewards_{}".format(i))
        plt.clf()
        fig.set_size_inches(18.0, 5.3)

        for ri in range(len(merge_folders)):
            run = merge_folders[ri].format(infix)

            comp_path = os.path.join(base_folder, run, "rewards.npy")
            rewards = np.load(comp_path)[:, :max_episode]

            mean_rewards = np.mean(rewards, axis=0)
            var_rewards = np.var(rewards, axis=0)

            offset = 1.96 * np.sqrt(var_rewards / len(rewards))

            plt.plot(mean_rewards, label="M", color=colors[ri][0])
            #plt.fill_between(np.arange(len(mean_rewards)), mean_rewards - offset, mean_rewards + offset, facecolor=colors[ri][1])

            plt.xlabel("t")
            plt.tight_layout()

            save_path = os.path.join(out_folder, file_name.format(infix))
            if save: plt.savefig(save_path)
            else: plt.show()

if plot_q_funcs:
    for ri in range(len(merge_folders)):

        fig = plt.figure("q_funcs_{}".format(ri))
        plt.clf()
        fig.set_size_inches(len(infixes) * 5.6, 4.5)

        q_funcs = list()
        v_max = -100000000
        v_min = -v_max

        # get the style params
        vmargin = 0.1
        hmargin = 0.07
        bmargin = 0.15
        bar_width = 0.015
        width_cell = (1 - bar_width - (2 + len(infixes)) * hmargin) / len(infixes)

        for i in range(len(infixes)):
            run = merge_folders[ri].format(infixes[i])
            comp_path = os.path.join(base_folder, run, "q_funcs/q_2400.npy")
            q_func = np.load(comp_path)
            q_funcs.append(q_func)
            v_max = np.maximum(np.max(q_func), v_max)
            v_min = np.minimum(np.min(q_func), v_min)

        for i in range(len(infixes)):
            ax = fig.add_axes([(i + 1) * hmargin + i * width_cell, vmargin, width_cell, 1 - 2 * vmargin])

            value_function = np.max(q_funcs[i], axis=0)
            box = [-1.2, 0.6, -0.07, 0.07]
            vf = plt.imshow(value_function, interpolation='nearest', extent=box, aspect='auto', vmax=v_max, vmin=v_min)
            ax.set_title(binfixes[i])
            ax.set_ylabel("v")
            ax.set_xlabel("x")

        clb_ax = fig.add_axes([(len(infixes) + 1) * hmargin + len(infixes) * width_cell, bmargin, bar_width, 1 - 2 * bmargin])
        fig.colorbar(vf, cax=clb_ax)

        save_path = os.path.join(out_folder, q_func_name.format(merge_folders[ri].format("")))
        if save: plt.savefig(save_path)
        else: plt.show()