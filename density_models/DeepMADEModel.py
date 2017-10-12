import numpy as np
import tensorflow as tf


# this class represents a deep made model built with tensorflow
class DeepMADEModel:

    # basic constructor for this model.
    # D - the size of the input to the made model
    # L - the number of hidden layers to use for this model
    def __init__(self, D, L):

        # save the parameters internally
        self.D = D
        self.K = D
        self.L = L

        # sample new masks
        self.create_masks()

        # create a placeholder for the x
        self.x = tf.placeholder(tf.float64, [self.D, None], name="x")
        self.init_weights()

        # now we want to build up the probability model
        hp = self.x
        for l in range(1, self.L + 1):
            hp = tf.sigmoid(tf.matmul(tf.add(self.b[l - 1], tf.multiply(self.W[l - 1], self.MW[l - 1])), hp))

        # calculate the probabilites
        xtop = tf.sigmoid(tf.add(self.c, tf.matmul(tf.multiply(self.V, self.MV), hp)))
        self.p = tf.exp(tf.reduce_sum((tf.add(tf.multiply(self.x, tf.log(xtop)),
                                       tf.multiply(tf.add(tf.ones([self.D], tf.float64), tf.negative(self.x)),
                                                   tf.log(tf.add(tf.ones([self.D], tf.float64), tf.negative(xtop)))))), axis=0))

        # build the negative log likelihood and use it inside the
        # minimizer
        self.nll = tf.reduce_mean(tf.reduce_sum(tf.negative(tf.add(tf.multiply(self.x, tf.log(xtop)), tf.multiply(tf.constant(1.0, tf.float64) - self.x,
                                                                                                   tf.log(tf.constant(1.0, tf.float64) - xtop)))), axis=0), axis=0)
        self.minimizer = tf.train.GradientDescentOptimizer(0.05).minimize(self.nll)

        # init the variables and the session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def create_masks(self):

        # generate the m
        m = self.gen_m_matrix()

        # now you can start to construct the masks
        # for each layer. Hence we need spae for
        # the mask tensor
        self.MW = [None] * self.L
        for l in range(self.L):
            # create the mask tensor for each layer
            self.MW[l] = self.oneOperator(m, l, l, 2 * [self.K])

        self.MV = self.oneOperator(m, 0, self.L, 2 * [self.K])

    # inits the weights
    def init_weights(self):

        # create the vectors for V and W

        self.c = self.init_single_weight([self.K], "c")
        self.V = self.init_single_weight([self.K, self.K], "V")
        self.W = list()
        self.b = list()
        for i in range(self.L):
            self.W.append(self.init_single_weight([self.K, self.K], "W" + str(i)))
            self.b.append(self.init_single_weight([self.K], "b" + str(i)))

        # return all weights
        return [self.W, self.V, self.b, self.c]

    # inits the weights
    def init_single_weight(self, size, name):
        return tf.Variable(tf.random_normal(size, mean=0.0, stddev=0.01, dtype=tf.float64), name=name)

    # this represents the numpty one operator
    def npOneOperator(self, m, ls, le, dim):

        # create the empty result
        result = np.empty(dim)

        # iterate over the whole matrix
        for r in range(dim[0]):
            for c in range(dim[1]):
                result[r, c] = int(m[ls, r] >= m[le, c])

        return result

    # this operator creates a mask matrix
    def oneOperator(self, m, ls, le , dim):

        result = self.npOneOperator(m, ls, le, dim)
        res = tf.Variable(result)
        tf.stop_gradient(res)
        return res

    # generates the m matrix
    def gen_m_matrix(self):

        # create the first row of this matrix
        m = np.empty([self.L + 1, self.K])
        m[0, :] = np.array([x for x in range(self.D)])
        np.random.shuffle(m[0, :])

        # now we want t sample the remaining values
        for l in range(1, self.L + 1):
            # define left and right boundary
            lb = np.min(m[l - 1, :])
            rb = self.D - 1

            # sample one number from the interval
            m[l, :] = np.random.randint(rb - lb) + lb

        return m

    # sample a new mask after each step
    def assign_mask(self):

        # generate random m matrix
        m = self.gen_m_matrix()

        self.TAW = [None] * self.L
        for l in range(self.L):

            # create the mask tensor for each layer
            self.sess.run(tf.assign(self.MW[l], self.npOneOperator(m, l, l, 2 * [self.K])))

        self.sess.run(tf.assign(self.MV, self.npOneOperator(m, 0, self.L, 2 * [self.K])))

    # this method basically walks one step in the direction
    # for the
    def step(self, samples):
        self.sess.run(self.minimizer, feed_dict={ self.x: samples })
        # self.assign_mask()

    # evaluates the model at sample
    def evaluate(self, x):

        # return the final value
        res = self.sess.run(self.p, feed_dict={self.x: x})
        # self.assign_mask()
        return res