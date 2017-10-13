# MIT License
#
# Copyright (c) 2017 Markus Semmler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

from os import makedirs, path
from plots.Plot import Plot


class DirManager:
    """A simple manager for the directory, e.g. it creates them if they don't
    exist and is capable of saving other objects to that folder.
    """

    def __init__(self, directory, environment, agent):
        """Specify thef folder hierarchically.

        Args:
            directory: The root directory to use
            environment: The name of the environment
            agent: The name of the agent
        """
        self.root = path.join(directory, environment, agent)
        if not path.exists(self.root):
            makedirs(self.root)

    def save_plot(self, plot, epoch, name = None):
        """This method takes a plot object and saves it
        in the folder.
        """
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

    def save_rewards(self, rewards):
        """Simply save rewards in specified folder"""
        filename = path.join(self.root, 'rewards.txt')
        np.savetxt(filename, rewards)
