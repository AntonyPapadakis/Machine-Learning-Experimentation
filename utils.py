import numpy as np


def loaddataset():
    datafile = "../UCIdata-exercise1/pima-indians-diabetes.data"
    namesfile = "../UCIdata-exercise1/pima-indians-diabetes.names"

    names = open(namesfile).readlines()
    for i in names: print(i)

    data = np.genfromtxt(datafile, delimiter=",")

    return data


def gaussian(X, mu, sigma, dim=None):
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
    return probability if probability != 0 else 10 ** -100  # TODO: hardcoded limit because log(0) threw warning and inf result.


def question_a(X):
    '''
    Covariance matrix is diagonal, with elements being equal, while the distribution is gaussian.

    :param X: The datapoints that belong to a specific class
    :return: mean vector and covariance matrix
    '''
    mus = np.mean(X, axis=0)

    diff = (X - mus)
    su = np.sum(np.dot(diff.T, diff)) / (8 * X.shape[0])

    sigmas = su * np.eye(X.shape[1])

    return mus, sigmas


def question_b(X):
    '''
    Covariance matrix is non-diagonal, while the distribution is gaussian.

    :param X: The datapoints that belong to a specific class
    :return: mean vector and covariance matrix
    '''
    # TODO: how? isn't using np.cov wrong?
    # TODO: is it copy paste this? https://stats.stackexchange.com/questions/351549/maximum-likelihood-estimators-multivariate-gaussian
    # TODO: Also for the Sigma, the summation doesn't break the 8x8 shape?
    # matrices SHOULD BE summed with each other and keep dimensions.
    mus = np.mean(X, axis=0)

    diff = (X - mus)
    sigmas = np.cov(X, rowvar=False)  # TODO change
    # sigmas = np.sum(np.dot(diff.T, diff)) / X.shape[0]

    return mus, sigmas


def question_c(X):
    '''
    Features are i.i.d. with gaussian marginal distributions.

    :param X: The datapoints that belong to a specific class
    :return: mean vector and covariance matrix
    '''
    mus = np.mean(X, axis=0)

    diff = (X - mus)

    sigmas = np.eye(X.shape[1])
    for i in range(mus.shape[0]):
        sigmas[i, i] = np.sum(np.square(diff[:, i])) / X.shape[0]

    return mus, sigmas


def question_d_multi(X, x_i):
    '''
    Features are i.i.d. with unknown marginal distributions.
    Window kernels are standard gaussians.

    :param X: The datapoints that belong to a specific class
    :return: the result of the pdf for some sample x_i
    '''

    h = np.sqrt(X.shape[0])
    kernel_mu = np.array([0 for j in range(X.shape[1])])
    kernel_sigma = np.eye(X.shape[1])

    # x_i = 0  # TODO: it probably needs test values / new samples, prob make it an argument
    sum = 0
    for i, x in enumerate(X):  # for each datapoint in X
        x_tmp = (x_i - x) / h
        kernel = gaussian(x_tmp, kernel_mu, kernel_sigma)
        sum += kernel

    denominator = (h ** X.shape[1]) * X.shape[0]
    # TODO: This is the multi-dimensional case of PWs

    # TODO: We can use integration to get a mean and variance from the pdf if needed
    return sum / denominator


def question_d(X, x_i):
    '''
    Features are i.i.d. with unknown marginal distributions.
    Window kernels are standard gaussians.

    :param X: The datapoints that belong to a specific class
    :return: the result of the pdf for some sample x_i
    '''

    h = np.sqrt(X.shape[0])
    kernel_mu = 0
    kernel_sigma = 1

    denominator = h * X.shape[0]

    # x_i = 0  # TODO: it probably needs test values / new samples, prob make it an argument
    marginals = []
    for j in range(X.shape[1]):
        sum = 0
        for i, x in enumerate(X[:, j]):  # for each datapoint in X
            x_tmp = (x_i - x) / h
            kernel = gaussian(x_tmp, kernel_mu, kernel_sigma, dim=1)
            sum += kernel

        marginals.append(sum / denominator)

    return marginals
