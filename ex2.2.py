from utils import *


def plotGoodness(aics, bics, num_of_models):
    import matplotlib.pyplot as plt
    N = num_of_models
    aic_class_0 = aics[:num_of_models]
    aic_class_1 = aics[num_of_models:]
    bic_class_0 = bics[:num_of_models]
    bic_class_1 = bics[num_of_models:]
    ind = np.arange(N)  # the x locations for the groups
    width = 0.20  # the width of the bars: can also be len(x) sequence

    fig = plt.figure(figsize=(15, 10))
    fig.clf()

    # Add axes and data
    bars = fig.add_subplot()

    p1 = bars.bar(ind, aic_class_0, width, zorder=2)
    p2 = bars.bar(ind, aic_class_1, width, bottom=aic_class_0, zorder=2)

    bars.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, zorder=0)

    bars.set_ylabel('Scores')
    bars.set_title('AIC and BIC per method')
    bars.set_xticks(ind)
    bars.set_xticklabels(('quest_a', 'quest_b', 'quest_c', 'quest_d'))
    plt.xticks(rotation=25)
    bars.legend((p1[0], p2[0]), ('AIC - 0', 'AIC - 1'))

    fig.show()


def gof(X, k, mus=None, sigmas=None, aic=False):
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
            log_pi = question_d(X, x)

        log_theta += log_pi

    gof = -2 * log_theta + k * penalty

    return round(gof, 2)


def main():
    data = loaddataset()
    X = data[:, :-1]
    y = data[:, -1:]

    aics = []
    bics = []

    for i in range(len(np.unique(y))):
        # Split set based on each label y.
        subset = np.array([X[j] for j in range(X.shape[0]) if y[j] == i])

        # Get pdf parameters for each class
        mus_a, sigmas_a = question_a(subset)
        mus_b, sigmas_b = question_b(subset)
        mus_c, sigmas_c = question_c(subset)

        # -------- Goodness of fits
        akaike_param_number = mus_a.shape[0] + 1  # mean's params plus the variance param
        aic = gof(subset, akaike_param_number, mus_a, sigmas_a, aic=True)
        bic = gof(subset, akaike_param_number, mus_a, sigmas_a)
        aics.append(aic)
        bics.append(bic)
        print("Class y={}, assumption a, average results (k={}): AIC={} and BIC={}".format(i, akaike_param_number, aic,
                                                                                           bic))

        m = mus_b.shape[0]
        akaike_param_number = m + m * (m + 1) / 2  # mean's params plus the covariance matrix's params (symmetric)
        aic = gof(subset, akaike_param_number, mus_b, sigmas_b, aic=True)
        bic = gof(subset, akaike_param_number, mus_b, sigmas_b)
        aics.append(aic)
        bics.append(bic)
        print("Class y={}, assumption b, average results (k={}): AIC={} and BIC={}".format(i, akaike_param_number, aic,
                                                                                           bic))

        akaike_param_number = mus_c.shape[0] + mus_c.shape[0]  # mean's params plus the variance diagonal
        aic = gof(subset, akaike_param_number, mus_c, sigmas_c, aic=True)
        bic = gof(subset, akaike_param_number, mus_c, sigmas_c)
        aics.append(aic)
        bics.append(bic)
        print("Class y={}, assumption c, average results (k={}): AIC={} and BIC={}".format(i, akaike_param_number, aic,
                                                                                           bic))

        akaike_param_number = subset.shape[1] + subset.shape[1] + 1  # mean and variance diagonal  for kernel plus
        # hypercube size
        aic = gof(subset, akaike_param_number, aic=True)
        bic = gof(subset, akaike_param_number)
        aics.append(aic)
        bics.append(bic)
        print("Class y={}, assumption d, average results (k={}): AIC={} and BIC={}".format(i, akaike_param_number, aic,
                                                                                           bic))

    plotGoodness(aics, bics, num_of_models=4)


if __name__ == "__main__":
    main()
