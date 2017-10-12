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
import tensorflow as tf
import extensions.tensorflow_extensions as tfe
import extensions.tensorflow_extensions as tfh


class NeuralNetwork:
    """A simple NeuralNetwork interface tightly coupled with tensorflow."""

    def __init__(self, config):
        """Simply creates some control variables for logging the variables.

        Args:
            config: The config to use for settings
        """

        # controls for scope logging
        self.vars = None
        self.log = {}
        self.conf = config

    # --- Graphs ---

    def copy_graph(self, from_scope, to_scope):
        """This copies the values from the passed network to the other.

        Args:
            from_scope: From which scope the weights come from
            to_scope: To which scope the weights go to
        """

        # create list for updates
        updates = list()
        def get_name(x): return x.name

        # iterate over
        for var, var_target in zip(
                sorted(self.log[from_scope], key=get_name),
                sorted(self.log[to_scope], key=get_name)):
            updates.append(var_target.assign(var))

        return tf.group(*updates)

    def eval_graph(self, x, scope, **kwargs):
        """This creates the graph it therefore receives the input tensor and
        of course the weights for each l.

        Args:
            x: The input to the graph. Usually a placeholder
            scope: The scope of this graph. It is used for creating multiple graph copies

        Returns:
            A fully constructed graph using the weights supplied.

        """
        raise NotImplementedError("Please implement evaluation graph")

    def grid_graph(self, grid_samples, intervals, scope):
        """This creates a graph which evaluates for each point of a dense grid
         the value of the neural network.

         Args:
            grid_samples: A list of values representing the number of values for this axis
            intervals: A list for the corresponding intervals
            scope: The scope for the evaluation network

        Returns:
            A list of created graphs
        """

        assert len(grid_samples) == len(intervals)

        # create akk ranges
        offset = 0.0001
        ranges = [tf.range(intervals[k][0], intervals[k][1],
                           (intervals[k][1] - intervals[k][0] - offset) / (grid_samples[k] - 1),
                           dtype=tf.float32)
                  for k in range(len(grid_samples))]

        # determine the length of all dimensions
        width = np.prod(grid_samples)
        elements = tf.meshgrid(*ranges)

        # reshape both array
        shaped_elements = [tf.reshape(sub_elements, [width]) for sub_elements in elements]

        # concat them
        all_elements = tf.stack(shaped_elements, axis=1)
        q_tensor = self.eval_graph(all_elements, scope)

        # split them up
        q_list = tf.unstack(q_tensor, axis=1)

        # resize them again
        return [tf.reshape(q, grid_samples) for q in q_list]

    def learn_graph(self, X, Y, scope, learning_rate, step_counter=None, clip_by_norm=3.0):
        """This creates the graph for learning, in detail it can be used to
        learn to max X to Y

        Args:
            X: The training graph itself
            Y: The actions which the network should output
            scope: The scope of this graph. It is used for creating multiple graph copies
            learning_rate: The learning rate to use
            step_counter: A counter used by the minimizer
            clip_by_norm: Clip the gradients to that value

        Returns:
            A fully constructed learning graph using the weights supplied.
        """

        # get huber loss from the error
        loss = tf.reduce_mean(tfh.huber_loss(1.0, X - Y))

        # Create a minimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss, var_list=self.log[scope])
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_by_norm), var)

        return optimizer.apply_gradients(gradients, global_step=step_counter)

    # --- Reused Funcs ---

    def _collect_scope_vars(self, scope, trainable=True):
        """This method collects all variables in the passed scope and saves
        it for faster reuse.

        Args:
            scope: The scope to collect the variables from
            trainable: True, if only trainable variables should be obtained
        """
        if scope not in self.log:
            self.vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES if trainable else tf.GraphKeys.VARIABLES,
                scope="{}/{}".format(tf.get_variable_scope().name, scope))

            self.log[scope] = self.vars

    def _eval_fc_network(self, x, structure, masks=None, regression_layer=True):
        """This method takes an input and a structure. And creates a connected neural
        network.

        Args:
            x: The input to the network.
            structure: The structure of the network.
            masks: The masks to use.
            regression_layer: Whether the last layer is a regression layer

        Returns:
            The created neural network.
        """

        # init vars of the creation algorithm
        Q = x
        for l in range(1, len(structure) - int(regression_layer)):
            with tf.variable_scope("layer_{}".format(l)):
                Q = tfe.eval_fc_layer(Q, structure[l - 1:l + 1],
                                      self.conf['activation_fn'], self.conf['layer_norm'],
                                      None if masks is None else masks[l - 1],
                                      self.conf['mask_type'])

        if regression_layer:
            Q = tfe.eval_fc_layer(Q, structure[-2:], layer_norm=self.conf['layer_norm'])

        return Q
