# Numpy as well as tensorflow will be used
# throughout hte application
import numpy as np
import tensorflow as tf

# import Memory for use as a replay memory
from src.util.Memory import Memory


# Constructs Deep NADE Model with Order Agnostic Training. Please refer to the papers available
# for more details on hpw this model works in detail.
class DeepNADEModel:

    # Constructs a DeepNADEModel using the passed parameters.
    #
    # - structure: Define the structure of the NADE Model as a list of integers
    #              where each integer specifies the numer of neurons in the specific layer
    #              NOTE: The number of neurons is implicitly controlled by the first integer in
    #                    this list.
    #
    # - alpha: Specify the learning rate, which should be used by the optimizer.
    # - num_orderings: For how much orderings should one sample be used.
    # - num_sampled_d: How much d should be sampled?
    # - memory_size: Here you can specify the size of the replay memory, which is useful in online settings.
    #
    def __init__(self, layers, num_orderings, num_sampled_d, minimizer, batch_size, memory_size=0):

        # fill in the values
        self.D = layers[0]
        structure = layers[1:]

        # save the properties of this model internally
        self.NO = num_orderings
        self.ND = num_sampled_d
        self.MS = memory_size

        # check if replay memory should be initialized for a replay memory.
        if memory_size != 0:
            self.M = Memory(memory_size, self.D)

        # define the variables and placeholders
        self.v = tf.placeholder(tf.float32, [self.D, None], name="v")
        self.alpha = tf.placeholder(tf.float32, [], name="alpha")

        # --------------- GRAPH ----------------------

        # Simply create the weights and the corresponding graph
        [weights, reg] = self.__produce_weights(layers)

        # create the training and the evaluation graph, using the same
        # shared weights
        aranged_d = tf.constant(np.arange(0, self.D), dtype=tf.int32)
        sampled_d = tf.slice(tf.random_shuffle(aranged_d), [0], [self.ND])

        # get the two graphs
        [M, _, _] = self.__produce_graph(aranged_d, weights)
        [_, pre_train_graph, mask_tensor_training] = self.__produce_graph(sampled_d, weights)

        # ---------------- EVALUATION -----------------

        # invert the passed arguments
        inv_v = tf.constant(1.0) - self.v
        M_inv = tf.constant(1.0) - M

        # expand v and inv_v
        exp_v = tf.expand_dims(tf.expand_dims(self.v, 0), 0)
        exp_inv_v = tf.expand_dims(tf.expand_dims(inv_v, 0), 0)
        exp_eye = tf.expand_dims(tf.expand_dims(tf.eye(self.D), 0), 3)

        # this gets the evaluation graph, if only one sample is supplied
        evaluation_sum = tf.multiply(M, exp_v) + tf.multiply(M_inv, exp_inv_v)
        diagonal_values = tf.reduce_sum(tf.multiply(exp_eye, evaluation_sum), axis=2)
        pd_probs = tf.reduce_prod(diagonal_values, axis=1)
        pred_pd_probs, pred_pd_var = tf.nn.moments(pd_probs, axes=[0])

        # set the evaluation model
        self.evaluation_model = pred_pd_probs

        # get log values
        log_iv_p_dist = tf.nn.softplus(pre_train_graph)
        log_p_dist = tf.nn.softplus(-pre_train_graph)

        # create training graph
        cD = tf.constant(self.D, dtype=tf.float32)
        normalization = cD / (cD - tf.cast(sampled_d, tf.float32) + tf.constant(1.0))

        # create cross entropy and afterwards the negative log likelihood graph
        cross_entropy = - tf.multiply(log_p_dist, exp_v) - tf.multiply(exp_inv_v, log_iv_p_dist)
        masked_hidden = tf.multiply(tf.constant(1.0) - mask_tensor_training, cross_entropy)
        reduced_sum = tf.reduce_mean(tf.reduce_sum(masked_hidden, axis=2), axis=2)
        self.nll = tf.reduce_mean(tf.reduce_mean(-normalization * reduced_sum, axis=1), axis=0)

        # additionally we try to minimize the variance of pd_probs as well
        self.minimizer = minimizer(self.alpha).minimize(self.nll)

        # ---------- SAVE & RESTORE ------------------

        self.saver = tf.train.Saver([weight for weight_pair in weights for weight in weight_pair])

        # ---------- INIT ------------------

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # This method creates weights for the passed layer structure of the network. It uses tensorflow
    # as a framework and simultaneously creates the matching regularization constant, which can afterwards
    # be used by the building algorithm for the graph itself.
    #
    # - layers: A list of integers, representing the number of neurons in each layer.
    #
    def __produce_weights(self, layers):

        # extract the pre size from the layers. And then
        # afterwards get a glimpse of the structure itself.
        pre_size = layers[0]
        structure = layers[1:]
        weights = list()
        reg = tf.constant(0.0)

        # iterate over all layers
        for l in range(len(structure)):
            # create new weights
            [W, b] = self.__init_weights(pre_size, structure[l])
            weights.append([W, b])
            pre_size = structure[l]

            # add regularization
            reg = reg + tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

        [W, b] = self.__init_weights(pre_size, self.D)
        weights.append([W, b])

        # add regularization
        reg = reg + tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        return [weights, reg]

    # this method produces a graph. It uses the number of sampled ds
    def __produce_graph(self, lst_d, weights):

        # create a mask tensor with the dimension of the permutation an the
        # number of sampled d's
        mask_tensor = self.__get_mask_tensor(self.NO, lst_d)

        # We have a 3 dimensional mask tensor, and a 2 dimension input value vector.
        # Hence we want to melt, them to a 4 dimensional tensor using the einsum
        v = tf.einsum('pde,ek->pdek', mask_tensor, self.v)

        # builder computational tree
        tree = v
        next_tree = v

        # iterate over all layers
        for [W, b] in weights:

            # create tree
            tree = tf.einsum('ie,pdek->pdik', W, next_tree) + b
            # next_tree = tf.maximum(tree, 0.1 * tree)
            next_tree = tf.nn.relu(tree)

        # create final distribution
        p_dist = tf.sigmoid(tree)

        return [p_dist, tree, tf.expand_dims(mask_tensor, 3)]

    # This method can be used to create a mask tensor. You pass a number of permutations.
    # The method is going to create a binary matrix M from (p x len(d) x e), where
    # sum(M[i, j, :]) == d[j] for each i, j.
    #
    # - p: The number of permutations per mask matrix
    # - d: A list of integers, representing the number of ones in each column
    #
    def __get_mask_tensor(self, p, d):

        # expand the d list, so that it is handled as a
        # column vector
        exp_d = tf.expand_dims(d, 0)

        # create range, but now, basically make a row
        # vector out of it.
        row_range = tf.range(0, self.D, 1)
        row_range = tf.expand_dims(row_range, 1)

        # create mask tensor
        boolean_mask = tf.greater(exp_d, row_range)
        mask_shape = [self.D, tf.size(d)]
        binary_mask = tf.where(boolean_mask, tf.ones(mask_shape), tf.zeros(mask_shape))

        # create for every permutation the corresponding matrices
        lst_mask_tensor = list()

        # shuffle the binary mask and add it in p different versions,
        # to the list.
        for k in range(p):
            shuffled = tf.random_shuffle(binary_mask)
            lst_mask_tensor.append(tf.transpose(shuffled))

        # create the final tensor of all masks
        train_mask_tensor = tf.stack(lst_mask_tensor, axis=0)

        # pass back the training mask tensor
        return train_mask_tensor

    # This method initializes the weights for one layer, including W and b.
    #
    # - in_size: Input size of layer.
    # - out_size: Output size of layer.
    #
    def __init_weights(self, in_size, out_size):

        # create the vectors for V and Wj
        W = self.__init_single_weight([out_size, in_size], "W")
        b = self.__init_single_weight([1, 1, out_size, 1], "b")

        # return all weights
        return [W, b]

    # This method initializes a single weight, given the sizes supplied.
    #
    # - size: The shape of the weight.
    # - name: The name of the weight.
    #
    @staticmethod
    def __init_single_weight(size, name):
        return tf.Variable(tf.random_normal(size, mean=0.0, stddev=0.005), name=name)

    # this method basically walks one step in the direction
    # for the
    def step(self, samples, num_steps=1, learning_rate=0.01):

        # when the size of the memory is zero use the original samples
        if self.MS == 0:

            # simply use the passed samples
            rand_samples = samples

        else:

            # length
            N = np.size(samples, 1)

            # add them to the memory
            for i in range(N):
                self.M.insert(samples[:, i])

            # sample a minibatch of same length than samples
            rand_samples = self.M.sample(10)

        # train
        for i in range(num_steps):

            # minimize
            self.sess.run(self.minimizer, feed_dict={self.v: rand_samples, self.alpha: learning_rate})

    # this method actually calculates the log likelihood
    def get_log_likelihood(self, samples):

        # simply retrieve the log likelihood for the passed data
        return self.sess.run(self.nll, feed_dict={ self.v: samples })

    # evaluates the model at sample
    def evaluate(self, samples):

        val = self.sess.run(self.evaluation_model, feed_dict={self.v: samples})
        return val

    # this saves the model
    def save_model(self, id, dir):
        self.saver.save(self.sess, dir + 'deep-nade-' + str(id))

    def restore_model(self, id, dir):
        self.saver.restore(self.sess, dir + 'deep-nade-' + str(id))
