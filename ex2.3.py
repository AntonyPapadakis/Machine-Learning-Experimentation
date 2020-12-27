import numpy as np
from utils import *


def cross_val_split(dataset, folds):
    """
    Splits the dataset into folds number of subsets of almost equal size after randomly shuffling it, for cross
    validation.

    :param dataset: The dataset to be splitted.
    :param folds: The number of folds to be created.
    :return: The dataset in cuts.
    """

    np.random.shuffle(dataset)
    splits = np.array_split(dataset, folds)

    return splits


def train_folds_merge(folds, test_id):
    """
    Glues together the folds of training splits into a unified train set.

    :param folds: The folds produced from the dataset segmentation for crossvalidation.
    :param test_id: The one fold that should be excluded to play the role of the test set.
    :return: The unified train set.
    """
    train_idxs = list(range(0, len(folds)))
    train_idxs.remove(test_id)
    list_of_splits = []

    for j in train_idxs:
        list_of_splits.append(folds[j])

    return np.vstack(list_of_splits)


def predictClass(x, mus, sigmas, X_train, number_of_classes, class_probabilities):
    """
    For every model, it calculates the likelihood for each class, and picks the class with max likelihood.

    :param x: The datapoint we want to derive the class for.
    :param mus: A list with the mean vector for each method. First three are for first class, next three for
    second class, etc.
    :param sigmas: A list with the covariance matrix for each method. Same as mus.
    :param X_train: The train set - needed for Parzen Windows method.
    :param number_of_classes: The number of different classes in the dataset.
    :param class_probabilities: An array with the probability of each class.
    :return: A vector with the predicted classes by each model.
    """

    predictions = []

    # For the parametric methods
    number_of_models = int(len(mus) / 2)
    for i in range(0, number_of_models):
        method_likelihoods = []
        for j in range(number_of_classes):
            index = i + j * number_of_models  # the index will "jump" over the other methds in the lists.

            prob = gaussian(x, mus[index], sigmas[index]) * class_probabilities[j]  # The beyes classifier rule
            method_likelihoods.append(prob)
        predictions.append(np.argmax(method_likelihoods))

    # For the non-parametric method
    method_likelihoods = []
    for j in range(number_of_classes):
        sumlog_pi = question_d(X_train, x)
        p_i = sumlog_pi * class_probabilities[j]  # The beyes classifier rule
        method_likelihoods.append(p_i)

    predictions.append(np.argmax(method_likelihoods))

    return predictions


def main():
    print("hello")
    data = loaddataset()
    folds = 5

    data_splits = cross_val_split(data, folds)
    accuracies = np.zeros((4, folds), dtype=int)  # array with one row for each model and one column for each fold.
    # Holds the counts of accurates for each model, which turns in accuracies before moving on to the next fold.

    # For each fold:
    for t, test_set in enumerate(data_splits):
        train_set = train_folds_merge(data_splits, t)

        X_train = train_set[:, :-1]
        y_train = train_set[:, -1:]
        X_test = test_set[:, :-1]
        y_test = test_set[:, -1:]

        number_of_classes = len(np.unique(y_train))
        class_probabilities = np.zeros(number_of_classes)  # array with the probability for each class to exist

        # Lists with model parameters, utilized in predictClass().
        mus = []
        sigmas = []

        # Training
        for i in range(number_of_classes):
            # Split set based on each label y.
            subset = np.array([X_train[j] for j in range(X_train.shape[0]) if y_train[j] == i])

            # The class probabitlity for this class and fold.
            class_probabilities[i] = subset.shape[0] / X_train.shape[0]

            # Get pdf parameters for each class
            mus_a, sigmas_a = question_a(subset)
            mus.append(mus_a)
            sigmas.append(sigmas_a)

            mus_b, sigmas_b = question_b(subset)
            mus.append(mus_b)
            sigmas.append(sigmas_b)

            mus_c, sigmas_c = question_c(subset)
            mus.append(mus_c)
            sigmas.append(sigmas_c)

            # Question c: Training for last distribution model takes place right before testing,
            # because there aren't any parameters to be learned beforehand.

        # Testing:
        for i in range(X_test.shape[0]):
            # Take the predictions from all methods
            preds = predictClass(X_test[i], mus, sigmas, X_train, number_of_classes, class_probabilities)

            # Evaluate these predictions
            target = y_test[i]
            for j, pred in enumerate(preds):
                if pred == target:
                    accuracies[j, t] += 1

        # Lastly, the accuracies for this fold are calculated in place.
        accuracies[:, t] = accuracies[:, t] * 100 / X_test.shape[0]

    accurates_counts_avg = accuracies.mean(axis=1)  # We average the accuracies of the k-fold crossvalidation

    print("Average Accuracy over {}-fold cross validation:\n".format(folds))
    print("Assumption A: {}%".format(round(accurates_counts_avg[0], 1)))
    print("Assumption B: {}%".format(round(accurates_counts_avg[1], 1)))
    print("Assumption C: {}%".format(round(accurates_counts_avg[2], 1)))
    print("Assumption D: {}%".format(round(accurates_counts_avg[3], 1)))


if __name__ == "__main__":
    main()
