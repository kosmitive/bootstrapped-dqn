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

import extensions.python_extensions as pe
import tensorflow as tf

import utils.extensions.tensorflow_extensions as tfe
from nn.NeuralNetwork import NeuralNetwork


class FiniteEnsembleNetwork(NeuralNetwork):
    """This represents a abstract FiniteEnsembleNetwork. It can be used to
    introduce some kind of uncertainty.."""

    def __init__(self, K, D, H, config):
        """This constructs a new FiniteEnsembleNetwork. It takes the structure of the
        network and also the configuration.

        Args:
            K: The number of heads
            D: The number of heads per decision
            H: The number of heads for learning
            config:
                layer_norm: Activate Layer Normalization
                activation_fn: Define the activation_fn as lambda tf.Tensor: -> tf.Tensor
                mask_type: The mask type to use for regularization.
        """

        # define some initial value if the were not set
        self.K = K
        self.D = D
        self.H = H

        # set some default values
        pe.set_default_val(config, 'ensemble_ucb_rho', 0.3)
        pe.set_default_val(config, 'layer_norm', True)
        pe.set_default_val(config, 'activation_fn', tfe.leakyrelu(0.1))
        pe.set_default_val(config, 'mask_type', None)

        # obtain the spaces
        super().__init__(config)

        # create random value graphs
        self.rand_as = tf.random_uniform([D], dtype=tf.int64)
        self.rand_tr = tf.random_uniform([H], dtype=tf.int64)

        # create the vars for the indices
        self.heads_as = tf.Variable(self.rand_as, name="heads_as")
        self.heads_tr = tf.Variable(self.rand_tr, name="heads_tr")

    # --- Graphs ---

    def eval_graph(self, x, scope, **kwargs):
        """This evaluates the input

        Args:
            x: The input to the graph. Usually a placeholder
            scope: The scope of this graph. It is used for creating multiple graph copies
            kwargs:
                mode: Can take value "as" or "tr"

        Returns:
            Depending on the mode the mean action is received or the result tensor
        """
        au = 'ensemble_ucb'
        approx_ucb = au in kwargs and kwargs[au]

        # determine the important
        mode = 'mode'
        use_tr = mode in kwargs and kwargs[mode] is 'training'
        heads = self.heads_tr if use_tr else self.heads_as

        # get the result
        res = self.eval_heads_graph(x, scope)
        perm = [1, 0, 2]
        gat_res = tf.gather(tf.transpose(res, perm), heads)
        if use_tr: return tf.transpose(gat_res, perm)

        mean, var = tf.nn.moments(gat_res, [0])
        return mean + int(approx_ucb) * self.conf['ensemble_ucb_rho'] * var

    def eval_heads_graph(self, x, scope):
        """This creates the graph for one or multiple heads.

        Args:
            x: The input to the graph. Usually a placeholder
            scope: The scope of this graph. It is used for creating multiple graph copies

        Returns:
            A list of evaluated heads
        """
        raise NotImplementedError("Please implement a possibility to sample heads.")

    def sample_heads_as_graph(self):
        """This method creates an operation to sample new heads for action selection.

        Returns:
            The assignment operation
        """
        return tf.assign(self.heads_as, self.rand_as)

    def sample_heads_tr_graph(self):
        """This method creates an operation to sample new heads for training.

        Returns:
            The assignment operation
        """
        return tf.assign(self.heads_tr, self.rand_tr)