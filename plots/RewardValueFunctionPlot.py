import warnings
import matplotlib.cbook
import matplotlib.pyplot as plt
import numpy as np

from plots.Plot import Plot


warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


class RewardValueFunctionPlot(Plot):
    """A plot capable of displaying a graph along with the value function
    and the actions used in each state."""

    def __init__(self, title, N, state_space):

        # create both color fields
        self.act_cmap = plt.cm.get_cmap('plasma', N)
        self.title = title
        self.initialized = False
        self.N = N

        # placeholders so that IDE does not
        # complain
        self.im_value_func = None
        self.im_target_value_func = None
        self.im_actions = None
        self.plt_rewards = None
        self.box = [state_space.IB[0], state_space.IE[0], state_space.IB[1], state_space.IE[1]]

    def interactive(self):
        self.interactive = True
        plt.ion()
        plt.show()

    def show(self):
        """This method stops the interactive mode if activated
        and shows the plot."""
        if self.interactive: plt.ioff()
        self.interactive = False
        plt.show()

    def initialize_figure(self, show_v):
        """Initializes the figures if not happened already"""

        if self.initialized: return

        self.fig = plt.figure(self.title, figsize=(13,7))
        self.initialized = True
        margin = 0.08
        half_ratio = (1.0 - 3 * margin) / 2
        third_ratio = (1.0 - 4 * margin) / 3

        if show_v:

            # create figure and subplots accordingly
            self.im_value_func = self.fig.add_axes([margin, margin, third_ratio, half_ratio])
            self.im_value_func.set_title("V-Function")

            self.im_target_value_func = self.fig.add_axes([2 * margin + third_ratio, margin, third_ratio, half_ratio])
            self.im_target_value_func.set_title("Target V-Function")

            self.im_actions = self.fig.add_axes([3 * margin + 2 * third_ratio, margin, third_ratio, half_ratio])
            self.im_actions.set_title("Q-Greedy")

            self.plt_rewards = self.fig.add_axes([margin, 0.5 + margin / 2, 1 - 2 * margin, half_ratio])
            self.plt_rewards.set_title("Rewards")

        else:

            self.plt_rewards = self.fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
            self.plt_rewards.set_title("Rewards")

    def update(self, rewards, q_funcs, t_q_funcs, plot_as_variance, obs_buffer):
        """This can be used to update the plot. It will prove whether
        the plot was already initialized.

        Args:
            rewards: The rewards to plot
            q_funcs: A list of Q-Functions (Mapping to Discrete Actions)
            plot_as_variance: True if the reward signals shall be displayed as a distribution
        """

        # discard the plot
        if self.initialized:
            plt.clf()
            self.initialized = False

        # and reininitialize
        self.initialize_figure(q_funcs is not None)

        mean_reward = np.mean(rewards, axis=0)
        var_reward = np.var(rewards, axis=0)
        offset = 1.96 * np.sqrt(var_reward / len(rewards))

        self.plt_rewards.plot(mean_reward, label="M", color="#215ab7")
        self.plt_rewards.fill_between(np.arange(len(mean_reward)), mean_reward - offset, mean_reward + offset, facecolor="#215ab786")
        self.plt_rewards.set_xlabel("t")

        if q_funcs is not None:
            # determine best actions and value function
            stacked_q_funcs = np.stack(q_funcs, axis=2)
            best_actions = np.argmax(stacked_q_funcs, axis=2)
            value_function = np.max(stacked_q_funcs, axis=2)

            # print both plots
            ba = self.im_actions.imshow(best_actions, interpolation='nearest', cmap=self.act_cmap, vmin=-0.5, vmax=2.5, extent=self.box, aspect='auto')
            self.im_actions.set_xlabel("x")
            self.im_actions.yaxis.set_visible(False)
            ba_cbar = plt.colorbar(ba, ax=self.im_actions, ticks=[0, 1, 2])
            ba_cbar.set_ticklabels(['Left', 'Nothing', 'Right'])

            # fill the axes
            vf = self.im_value_func.imshow(value_function, interpolation='nearest', extent=self.box, aspect='auto')
            self.im_value_func.set_ylabel("v")
            self.im_value_func.set_xlabel("x")
            self.im_value_func.plot(obs_buffer[:,0], obs_buffer[:,1], color='black')
            plt.colorbar(vf, ax=self.im_value_func)

            # determine best actions and value function
            t_stacked_q_funcs = np.stack(t_q_funcs, axis=2)
            t_value_function = np.max(t_stacked_q_funcs, axis=2)

            # fill the axes
            vf2 = self.im_target_value_func.imshow(t_value_function, interpolation='nearest', extent=self.box, aspect='auto')
            self.im_target_value_func.set_ylabel("v")
            self.im_target_value_func.set_xlabel("x")
            plt.colorbar(vf2, ax=self.im_target_value_func)

        if self.interactive: plt.pause(0.01)

    def save(self, filename):
        self.fig.savefig(filename)

    def name(self):
        return "rew_val_plot"