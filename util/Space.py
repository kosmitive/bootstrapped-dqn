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
from numpy.random import uniform, randint


class Space:
    """This class is a space. This is is both applicable for the
    continuous and discrete cases.
    """

    def __init__(self, dims):
        """Constructs a discrete space. Only the size has to
        be declared. Each dims element can either be defined
        by one or two numbers. The first corresponds to
        elements {0, ..., dims[i]-1}. Whereas the latter is
        interpreted like [dims[i][0], dims[i][1]].

        Args:
            dims: A list like described above.
        """

        self.dims = dims
        self.dims = [[dim] if isinstance(dim, int) else dim for dim in self.dims]
        self.D = len(self.dims)
        self.is_continuous = np.equal(self.D, 2)
        self.is_discrete = np.equal(self.D, 1)

        # define the bounds as well.
        l_bounds = [2 - len(dim) + (len(dim) - 1) * dim[0] for dim in self.dims]
        r_bounds = [dim[-1] for dim in self.dims]
        self.bounds = np.array(zip(l_bounds, r_bounds))

# --- Python ---

    def sample_element(self, shape=None):
        """Samples uniformly from the current space. Note that
        if shape is (X,Y,Z) the result will be (X,Y,Z,D)
        """

        sampling_lst = list()
        for d in range(self.D):
            f = uniform if self.is_continuous[d] else randint
            sampling_lst.append(f(*self.bounds[d], shape))

        return np.stack(sampling_lst, axis=len(shape) - 1)
