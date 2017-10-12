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

from nn.FiniteEnsembleNetwork import FiniteEnsembleNetwork


class DuplicatedEnsembleNetwork(FiniteEnsembleNetwork):
    """This represents a general DuplicatedEnsembleNetwork. It can be used in different
    types of contexts."""

    def __init__(self, K, D, shared_structure, head_structure, config):
        """This constructs a new DuplicatedEnsembleNetwork. It takes the structure of the
        network and also the configuration.

        Args:
            shared_structure: The structure for the shared network
            head_structure: The structure for the head network
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
        self.shared_structure = shared_structure
        self.head_structure = head_structure
        super().__init__(K, D, K, config)

    # --- Graphs ---

    def eval_heads_graph(self, x, scope):
        """This creates the graph for one or multiple heads.

        Args:
            x: The input to the graph. Usually a placeholder
            scope: The scope of this graph. It is used for creating multiple graph copies

        Returns:
            A list of evaluated heads
        """
        with tf.variable_scope(scope, reuse=(scope in self.log)):
            shared_net = self._eval_fc_network(x, self.shared_structure, regression_layer=False)

            # sample all heads
            head_nets = list()
            for k in range(self.K):
                with tf.variable_scope("head_{}".format(k)):
                    head_nets.append(self._eval_fc_network(shared_net, self.head_structure, regression_layer=True))

            heads = tf.stack(head_nets, axis=1)
        self._collect_scope_vars(scope)
        return heads
