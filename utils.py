import numpy as np


def loaddataset():
    """
    Reads the dataset file and returns the dataset as an array
    :return: The dataset array.
    """
    datafile = "../UCIdata-exercise1/pima-indians-diabetes.data"
    namesfile = "../UCIdata-exercise1/pima-indians-diabetes.names"

    names = open(namesfile).readlines()
    for i in names: print(i)

    data = np.genfromtxt(datafile, delimiter=",")

    return data


def gaussian(X, mu, sigma, dim=None):
    """
    Calculates the gaussian probability for a given set.
    :param X: The variable vector in the gaussian calculation.
    :param mu: The mean of the wanted distribution.
    :param sigma: The (co)variance of the wanted distribution.
    :param dim: Indicator about whether we need the 1-dim or multi-dim gaussian case.
    :return: The resulting likelihood.
    """
    if dim != 1:
        x_len = len(X)
        denominator = (2 * np.pi) ** (x_len / 2) * np.sqrt(np.linalg.det(sigma))
        # exp = (-1/2)*np.dot(np.matmul(X-mu, np.linalg.inv(sigma)), X-mu)
        nominator = np.exp(-0.5 * np.dot(np.dot(np.transpose(X - mu), np.linalg.inv(sigma)), (X - mu)))
    else:
        z_score = (X - mu) / sigma
        denominator = sigma * np.sqrt(2 * np.pi)
        nominator = np.exp(-0.5 * np.dot(z_score.T, z_score))

    probability = (nominator / denominator)
    return probability if probability != 0 else 10 ** -100
    # TODO: hardcoded min because log(0) threw warning and inf result.


def question_a(X):
    """
    Covariance matrix is diagonal, with elements being equal, while the distribution is gaussian.

    :param X: The datapoints that belong to a specific class
    :return: mean vector and covariance matrix
    """
    mus = np.mean(X, axis=0)

    diff = (X - mus)
    su = np.sum(np.dot(diff.T, diff)) / (8 * X.shape[0])

    sigmas = su * np.eye(X.shape[1])

    return mus, sigmas


def question_b(X):
    """
    Covariance matrix is non-diagonal, while the distribution is gaussian.

    :param X: The datapoints that belong to a specific class
    :return: mean vector and covariance matrix
    """

    mus = np.mean(X, axis=0)

    diff = (X - mus)
    sigmas = np.dot(diff.T, diff) / X.shape[0]

    return mus, sigmas


def question_c(X):
    """
    Features are i.i.d. with gaussian marginal distributions.

    :param X: The datapoints that belong to a specific class
    :return: mean vector and covariance matrix
    """

    mus = np.mean(X, axis=0)

    diff = (X - mus)

    sigmas = np.eye(X.shape[1])
    for i in range(mus.shape[0]):
        sigmas[i, i] = np.sum(np.square(diff[:, i])) / X.shape[0]

    return mus, sigmas


def question_d_multidimensional(X, x_i):
    """
    Features are i.i.d. with unknown marginal distributions.
    Window kernels are standard gaussians.
    This is the multi-dimensional case of PWs.
    We can use integration to get a mean and variance from the pdf if needed.

    :param X: The datapoints that belong to a specific class.
    :param x_i: The variable we want to derive the likelihood for.
    :return: the result of the pdf for some sample x_i.
    """
    # TODO: remove this function?
    h = np.sqrt(X.shape[0])
    kernel_mu = np.array([0 for _ in range(X.shape[1])])
    kernel_sigma = np.eye(X.shape[1])

    sum = 0
    for i, x in enumerate(X):  # for each datapoint in X
        x_tmp = (x_i - x) / h
        kernel = gaussian(x_tmp, kernel_mu, kernel_sigma)
        sum += kernel

    denominator = (h ** X.shape[1]) * X.shape[0]
    return sum / denominator


def question_d(X, x_i):
    """
    Features are i.i.d. with unknown marginal distributions.
    Window kernels are standard gaussians.

    :param X: The datapoints that belong to a specific class.
    :param x_i: The variable we want to derive the likelihood for.
    :return: the iid result vector of the pdf for some sample x_i.
    """

    h = np.sqrt(X.shape[0])
    kernel_mu = 0
    kernel_sigma = 1

    denominator = h * X.shape[0]

    marginals = []
    for j in range(X.shape[1]):
        summ = 0
        for i, x in enumerate(X[:, j]):  # for each datapoint in X
            x_tmp = (x_i - x) / h
            kernel = gaussian(x_tmp, kernel_mu, kernel_sigma, dim=1)
            summ += kernel

        marginals.append(summ / denominator)

    return marginals
