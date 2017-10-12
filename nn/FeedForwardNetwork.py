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
import extensions.tensorflow_extensions as tfe
import extensions.python_extensions as pe

from nn.NeuralNetwork import NeuralNetwork


class FeedForwardNetwork(NeuralNetwork):
    """This represents a general FeedForwardNetwork. It can be used in different
    types of contexts. In general it operates on batches of 1-dimensional
    data."""

    def __init__(self, structure, config):
        """This constructs a new FeedForwardNetwork. It takes the structure of the
        network and also the configuration.

        Args:
            structure: The structure for the hidden layers, a list of integers.
            config:
                layer_norm: Activate Layer Normalization
                activation_fn: Define the activation_fn as lambda tf.Tensor: -> tf.Tensor
                mask_type: The mask type to use for regularization.
        """

        # define some initial value if the were not set
        pe.set_default_val(config, 'layer_norm', True)
        pe.set_default_val(config, 'activation_fn', tfe.leakyrelu(0.1))
        pe.set_default_val(config, 'mask_type', None)

        # obtain the spaces
        self.structure = structure
        super().__init__(config)

    # --- Graphs ---

    def eval_graph(self, x, scope, **kwargs):
        """This creates the graph it therefore receives the input tensor and
        of course the weights for each l.

        Args:
            x: The input to the graph. Usually a placeholder
            scope: The scope of this graph. It is used for creating multiple graph copies

        Returns:
            A fully constructed graph using the weights supplied.

        """
        with tf.variable_scope(scope, reuse=(scope in self.log)):
            Q = self._eval_fc_network(x, self.structure, regression_layer=True)

        self._collect_scope_vars(scope)
        return Q

