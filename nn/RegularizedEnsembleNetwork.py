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


class RegularizedEnsembleNetwork(FiniteEnsembleNetwork):
    """This represents a RegularizedEnsembleNetwork."""

    def __init__(self, K, D, H, structure, theta, config):
        """This constructs a new RegularizedEnsembleNetwork. The structure of the whole network
        is given however different masks are sampled such that

        Args:
            K: The number of heads
            D: The number of heads per decision
            H: The number of heads for learning
            structure: The structure for the hidden layers, a list of integers.
            theta: The probability of sampling a one.
            config:
                layer_norm: Activate Layer Normalization
                activation_fn: Define the activation_fn as lambda tf.Tensor: -> tf.Tensor
                mask_type: The mask type to use for regularization.
        """
        if 'mask_type' not in config or config['mask_type'] is None:
            raise RuntimeError("You have to specify a mask_type")

        # define some initial value if the were not set
        pe.set_default_val(config, 'layer_norm', True)
        pe.set_default_val(config, 'activation_fn', tfe.leakyrelu(0.1))
        pe.set_default_val(config, 'mask_type', None)

        # obtain the spaces
        self.structure = structure
        super().__init__(K, D, H, config)

        # sample the masks
        self.masks = list()
        for l in range(int(config['mask_type'] is 'zoneout'), len(structure)):
            random_mask = tf.distributions.Bernoulli(probs=theta)
            random_mask_var = tf.Variable(random_mask.sample([structure[l], K]), trainable=False)
            self.masks.append(random_mask_var)

    # --- Graphs ---

    def eval_heads_graph(self, x, scope):
        """This creates the graph for one or multiple heads.

        Args:
            x: The input to the graph. Usually a placeholder
            scope: The scope of this graph. It is used for creating multiple graph copies

        Returns:
            A list of evaluated heads
        """
        heads = list()
        with tf.variable_scope(scope, reuse=(scope in self.log)):
            for k in range(self.K):
                k_masks = [mask[:, k] for mask in self.masks]
                head = self._eval_fc_network(x, self.structure, regression_layer=True, masks=k_masks)
                heads.append(head)

            heads = tf.stack(heads, axis=1)
        self._collect_scope_vars(scope)
        return heads

