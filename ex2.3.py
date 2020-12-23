import numpy as np
from utils import *

# First three for each are for first class, next three for second class, etc.
mus = []
sigmas = []


def cross_val_split(dataset, folds):
    # split the dataset into folds number of subsets of almost equal size after randomly shuffling it
    cv_dataset = dataset.copy()
    np.random.shuffle(cv_dataset)
    splits = np.array_split(cv_dataset, folds)
    return splits


def train_folds_merge(folds, test_id):
    train_idxs = list(range(0, len(folds)))
    train_idxs.remove(test_id)
    list_of_splits = []
    for j in train_idxs:
        list_of_splits.append(folds[j])

    return np.vstack(list_of_splits)


def predictClass_Parametrics(x, number_of_classes):
    likelihoods = []

    for i in range(0, (len(mus) / 2)):
        method_likelihoods = []
        for j in range(number_of_classes):
            index = i + j * (len(
                mus) / 2)  # this formula makes sure the index will "jump" over the other methds in the lists.
            method_likelihoods.append(gaussian(x, mus[index], sigmas[index]))
        likelihoods.append(np.argmax(method_likelihoods))
    return likelihoods


def main():
    print("hello")
    data = loaddataset()
    folds = 5

    data_splits = cross_val_split(data, folds)
    acc = np.zeros(folds)
    sums = np.zeros(folds)

    for t, test_set in enumerate(data_splits):
        train_set = train_folds_merge(data_splits, t)

        X_train = train_set[:, :-1]
        y_train = train_set[:, -1:]
        X_test = test_set[:, :-1]
        y_test = test_set[:, -1:]

        number_of_classes = len(np.unique(y_train))

        # Training
        for i in range(number_of_classes):
            # Split set based on each label y.
            subset = np.array([X_train[j] for j in range(X_train.shape[0]) if y_train[j] == i])

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

            # Training for last distribution model takes place right before testing,
            # because there aren't any parameters to be learned beforehand.

        # Testing:
        accurates_counts = np.zeros((4, 5), dtype=int)  # matrix to hold the temp predictions for each pdf model

        for i in range(X_test.shape[1]):
            # Take the predictions from all methods

            # For the parametric methods
            preds = predictClass_Parametrics(X_test[i], number_of_classes)

            # For the non-parametric method
            method_likelihoods = []
            for j in range(number_of_classes):
                method_likelihoods.append(question_d_multi(X_train, X_test[i]))  # TODO: Change into marginal method
            preds.append(np.argmax(method_likelihoods))

            # Evaluate these predictions
            for j, p in enumerate(preds):
                if p == y_test[i]:
                    accurates_counts[j][t] += 1

    # TODO: Average accurates_counts over the fold scores

    # TODO: Print the accuracy over X_test size

    print("hello")


if __name__ == "__main__":
    main()
