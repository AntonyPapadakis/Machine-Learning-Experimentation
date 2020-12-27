import numpy as np
from utils import *


def gof(X, mus=None, sigmas=None, aic=False):
    k = X.shape[1]  # TODO: +1 for sigma for a, +8 for sigma for c etc or not?

    if aic:
        penalty = 2
    else:  # for bic
        penalty = np.log(X.shape[0])

    log_theta = 0

    for i, x in enumerate(X):

        # Instead of calculating the log product, we do the sum of logs to avoid multiplying to 0.
        if mus is not None:
            p_i = gaussian(x, mus, sigmas)
            log_pi = np.log(p_i)
        else:
            p_i = question_d(X, x)
            log_pi = np.sum(np.log(p_i))
            # TODO Problem: the metric gets 10 times larger

        log_theta += log_pi



        if log_theta == 0:
            print("dsfsd")

    gof = -2 * log_theta + k * penalty

    return round(gof, 2)


def gof1(X, mus=None, sigmas=None, aic=False):
    #TODO: remove this, it's just the parzen for d-space not 1-d.
    k = X.shape[1]

    if aic:
        penalty = 2
    else:  # for bic
        penalty = np.log(X.shape[0])

    log_theta = 0

    for i, x in enumerate(X):

        # Instead of calculating the log product, we do the sum of logs to avoid multiplying to 0.
        if mus is not None:
            p_i = gaussian(x, mus, sigmas)
            log_pi = np.log(p_i)
        else:
            p_i = question_d_multidimensional(X, x)
            log_pi = np.log(p_i)

        log_theta += log_pi

        if log_theta == 0:
            print("dsfsd")

    gof = -2 * log_theta + k * penalty

    return round(gof, 2)

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

        # -------- Goodness of fits
        print("Class y={}, assumption a, average results: AIC={} and BIC={}".
                                        format(i, gof(subset, mus_a, sigmas_a, aic=True),gof(subset, mus_a, sigmas_a)))
        print("Class y={}, assumption b, average results: AIC={} and BIC={}".
                                        format(i, gof(subset, mus_b, sigmas_b, aic=True), gof(subset, mus_b, sigmas_b)))
        print("Class y={}, assumption c, average results: AIC={} and BIC={}".
                                        format(i, gof(subset, mus_c, sigmas_c, aic=True), gof(subset, mus_c, sigmas_c)))
        print("Class y={}, assumption d, average results: AIC={} and BIC={}".
                                        format(i, gof(subset, aic=True), gof(subset)))

        print("----")

        print("Class y={}, assumption d_multi, average results: AIC={} and BIC={}".
                                        format(i, gof1(subset, aic=True), gof1(subset)))
        print("test")

    print("hello")


if __name__ == "__main__":
    main()
