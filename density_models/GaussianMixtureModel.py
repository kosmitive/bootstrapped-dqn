# import external packages
import matplotlib.pyplot as plt
import numpy as np

# import custom classes
from src.density_models.Gaussian import Gaussian


# This class represents a gaussian mixture model. The constructor can
# be used to create a random mixture model.
class GaussianMixtureModel:

    # This initializes the model,
    # numGauss -> Number of Gaussians
    # rangeMu  -> mu is from interval [-rangeMu, rangeMu]
    # minVar   -> Ensure this minimum variance
    # dim      -> dimension of each gaussian and hence the whole model
    # seed     -> seed, to reproduce a failure
    def __init__(self, numGauss, size, minVar, maxVar, minCov, maxCov, dim, seed):

        # seed the generator
        np.random.seed(seed)

        # save some fields
        self.dim = dim
        self.numGauss = numGauss
        self.size = size

        # initialize the normals
        self.normals = list()
        for k in range(numGauss):

            # create matrix, such that the entries are in (-1,1) except the diagonals. Afterwards
            # compute a positive definite matrix as well as the mean and initialize a gaussian
            # covL = np.ones((dim, dim)) - 2 * np.random.rand(dim, dim) + np.diag(minVar * np.ones((dim, 1)))
            covC = np.array([[3,1],[1,3]]) # np.dot(covL.transpose(), covL)
            covC = np.diag(minVar + np.random.rand(dim) * (maxVar - minVar)) \
                   + (minCov + (maxCov - minCov) * np.random.random_sample()) * (np.ones((dim, dim)) - np.eye(dim))
            mu = self.size * np.random.rand(dim, 1)
            self.normals.append(Gaussian(mu, covC))


    # This function can be used to generate a plot using the matplotlib
    def plot(self):
        [limitInf, limitSup] = self.getRange()

        # get the field
        numPoint = 1000
        [xv, yv, z] = self.getField(limitInf, limitSup, [numPoint, numPoint])

        plt.figure()
        plt.contourf(xv, yv, z, 30)
        plt.show()

    # can be used to obtain a stationary distribution
    def getField(self, limitInf, limitSup, numPoints):

        # create both lines
        x = np.linspace(limitInf[0], limitSup[0], numPoints[0])
        y = np.linspace(limitInf[1], limitSup[1], numPoints[1])

        # get meshgrid
        xv, yv = np.meshgrid(x, y)

        # calculate target
        z = self.eval(xv, yv)
        return [x, y, z]

    # can be used to obtain a stationary distribution
    def getNormalizedField(self, limitInf, limitSup, numPoints):
        [x, y, z] = self.getField(limitInf, limitSup, numPoints)
        nz = z / np.sum(z)
        return [x, y, nz]

    # This function can be used to obtain the range of this model
    def getRange(self):
        limitInf = np.finfo('d').max * np.ones(self.dim)
        limitSup = np.finfo('d').min * np.ones(self.dim)

        # iterate over all distributions
        for k in range(self.numGauss):
            [limInf, limSup] = self.normals[k].getLimits()
            limitInf = np.minimum(limitInf, limInf)
            limitSup = np.maximum(limitSup, limSup)

        return [[0, 0], [self.size, self.size]]

    # this function can be used to evaluate this model
    def eval(self, xv, yv):
        x = xv.flatten()
        y = yv.flatten()
        samples = np.vstack((x, y))
        z = np.zeros(np.size(samples, 1))
        for dist in self.normals:
            z += np.exp(dist.getLogProbas(samples))

        zv = np.reshape(z, xv.shape)
        return zv
