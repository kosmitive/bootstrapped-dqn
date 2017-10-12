import numpy as np

# This class represents a gaussian. It holds the necessary fields and some
# methods to access the properties of this object.
class Gaussian:

    # Constructor creates a basic covariance matrix.
    def __init__(self, mu, sigma):

        # save the mu internally
        self.mu = mu

        # compute the precision using cholesky
        # and invert the cholesky matrix L
        self.cholC = np.linalg.cholesky(sigma)
        self.cholP = np.linalg.solve(self.cholC, np.eye(np.size(mu)))
        self.precision = self.cholP.transpose().dot(self.cholP)
        self.covariance = sigma

    # This function is used to retrieve the limits of this particular
    # gaussian. Therefore the deviation is calculated.
    def getLimits(self):

        # get standard deviation
        sigma = np.sqrt(np.diag(self.covariance).transpose())

        # now calculate the limits
        limInf = self.mu - 2 * sigma
        limSup = self.mu + 2 * sigma

        # pass back the limits
        return [limInf, limSup]

    # delivers the dimension of this gaussian
    def getDim(self):
        return np.size(self.mu, 0)

    # this function can be used to evaluate for a given position the
    # probability
    def getLogProbas(self, samples):
        diff = samples - self.mu
        vecs = np.linalg.solve(self.cholC, diff)
        dots = np.einsum('ij,ij->j', vecs, vecs)
        res = -0.5 * (self.getDim() * np.log(2.0 * np.pi) + dots) - np.sum(np.log(np.diag(self.cholC)))
        return res