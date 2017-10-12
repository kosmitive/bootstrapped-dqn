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

import tensorflow as tf


class Memory:
    """This is the memory interface, which can be utilized by Replay Memories,
    for a simple integration into the framework.
    """
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def reset_graph(self):
        """This should create a graph which resets the memory.

        Returns: A graph resetting the memory.
        """
        raise NotImplementedError()

    def sample_graph(self, sample_size):
        raise NotImplementedError()

    def store_and_sample_graph(self, current_state, next_state, action, reward, done):
        """This method inserts a new tuple into the replay memory.

        Args:
            current_state: The current_state in a binary encoded fashion.
            reward: The reward for the action taken
            action: The action that was taken for the reward
            next_state: The state after the action was executed.
            done: Whether the the episode was finished or not.
        """

        insert_count = self.store_graph(current_state, next_state, action, reward, done)

        # create a new variable scope
        with tf.variable_scope("replay_memory"):
            insert_op = tf.group(insert_count)
            with tf.control_dependencies([insert_op]):
                samples = self.sample_graph(self.sample_size)

            return samples

    def store_graph(self, current_state, next_state, action, reward, done):
        """This graph takes one experience tuple and stores it internally."""
        raise NotImplementedError()
