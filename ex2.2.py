import numpy as np
from utils import *


def gof(X, mus=None, sigmas=None, aic=False):
    k = X.shape[1]  # TODO: mike said +1 for sigma for a, +8 for sigma for c etc.
    avg = 0

    if aic:
        penalty = 2
    else:  # for bic
        penalty = np.log(X.shape[0])

    for i, x in enumerate(X):

        if mus is not None:  # TODO: since we obtain these values with max likelihood this is correct?
            log_theta = np.log(gaussian(x, mus, sigmas))
        else:  # TODO: since we DIDN'T obtain these values with max likelihood, we need to implement Max Likelihood PROB NOT
            # on derived pdf?
            y = question_d(X, x)
            f = np.prod(y)
            t = np.log(y)
            log_theta = np.sum(t)
            # TODO: Instead of calculating the log product, we should the sum of logs to avoid multiplying to 0.
            # Problem: result is negative and takes the metric way up.

        if log_theta == 0:
            print("dsfsd")

        avg += -2 * log_theta + k * penalty

    return avg / X.shape[0]  # TODO: multiply probabilities and take one akaike for dataset, instead of average


def main():
    print("hello")
    data = loaddataset()
    X = data[:, :-1]
    y = data[:, -1:]

    for i in range(len(np.unique(y))):
        # Split set based on each label y.
        subset = np.array([X[j] for j in range(X.shape[0]) if y[j] == i])

        # Get pdf parameters for each class
        mus_a, sigmas_a = question_a(subset)
        mus_b, sigmas_b = question_b(subset)
        mus_c, sigmas_c = question_c(subset)
        # mus_d, sigmas_d = question_d(subset) # TODO: prob delete from here? it doesn't give any params

        # -------- Goodness of fits
        print("Class y={}, assumption a, average results: AIC={} and BIC={}".
                                        format(i, gof(subset, mus_a, sigmas_a, aic=True),gof(subset, mus_a, sigmas_a)))
        print("Class y={}, assumption b, average results: AIC={} and BIC={}".
                                        format(i, gof(subset, mus_b, sigmas_b, aic=True), gof(subset, mus_b, sigmas_b)))
        print("Class y={}, assumption c, average results: AIC={} and BIC={}".
                                        format(i, gof(subset, mus_c, sigmas_c, aic=True), gof(subset, mus_c, sigmas_c)))
        print("Class y={}, assumption d, average results: AIC={} and BIC={}".
                                        format(i, gof(subset, aic=True), gof(subset)))

        print("test")

    print("hello")


if __name__ == "__main__":
    main()
