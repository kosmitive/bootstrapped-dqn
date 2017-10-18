import configparser
import numpy as np
from os import makedirs, path
from plots.Plot import Plot


class DirectoryManager:

    def __init__(self, directory, environment, agent):

        self.root = path.join(directory, environment, agent)
        if not path.exists(self.root):
            makedirs(self.root)

    def save_plot(self, plot, epoch, name = None):
        assert isinstance(plot, Plot)

        if name is None:
            folder = path.join(self.root, plot.name())
            filename = path.join(folder, "plt_{}.eps".format(epoch))
        else:
            folder = self.root
            filename = path.join(folder, name)

        if not path.exists(folder):
            makedirs(folder)

        plot.save(filename)

    def save_q_func(self, q, epoch, name = None):

        if name is None:
            folder = path.join(self.root, "q_funcs")
            filename = path.join(folder, "q_{}.npy".format(epoch))
        else:
            folder = self.root
            filename = path.join(folder, name)

        if not path.exists(folder):
            makedirs(folder)

        np.save(filename, q)

    def save_rewards(self, rewards):

        filename = path.join(self.root, 'rewards.npy')
        np.save(filename, rewards)

    def save_config(self, config):
        parser = configparser.ConfigParser()
        parser.add_section('Settings')
        for key in config.keys():
            parser.set('Settings', key, str(config[key]))

        filename = path.join(self.root, 'config.ini')
        with open(filename, 'w') as f:
            parser.write(f)