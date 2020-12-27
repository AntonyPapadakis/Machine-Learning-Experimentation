import numpy as np
from numpy import genfromtxt


def load_dataset(fname):
    # loads a dataset assuming the last column is the labels
    # return the number of labels, the number of feats and the labels as strings
    my_data = genfromtxt(fname, delimiter=',', dtype='S', encoding=None)
    num_feats = my_data.shape[1]-1
    my_data_feats = np.array(my_data[:, :num_feats], dtype='float32')
    labels, indexed_data = np.unique(my_data[:, -1], return_inverse=True)
    my_data_labels = np.array(indexed_data)
    dataset = np.hstack((my_data_feats, my_data_labels[:, None]))
    return [dataset, labels.shape[0], my_data_feats.shape[1], labels]

def transform_labels(dataset, lb):
    # in order to transform the problem to a binary classification
    # change lb label to 1 and all others to -1
    # return a copy of the original dataset with the transformed labels
    t_dataset = dataset.copy()
    transf = np.where(t_dataset[:, -1].astype(int) == lb, 1, -1)
    t_dataset[:, -1] = transf[:]
    return t_dataset

def perceptron_classifier(patterns, targets, lr):
    # implementation of a single neuron perceptron as described in the lecture notes
    # try to converge for 10k epochs
    # return the weights and bias(w0) after 10k epochs or if converged
    epochs = 10000
    aug_patterns = np.hstack((patterns, -1 * np.ones((patterns.shape[0], 1))))
    t_aug_patterns = aug_patterns[:, :] * targets[:, None]
    weights = np.zeros(t_aug_patterns.shape[1])
    num_patterns = patterns.shape[0]
    for j in range(0, epochs):
        errors = 0
        for i in range(t_aug_patterns.shape[0]):
            y = np.dot(weights, t_aug_patterns[i, :])
            if y <= 0:
                errors += 1
                weights = weights + (lr * t_aug_patterns[i, :])
        acc = (num_patterns - errors) / num_patterns
        if errors == 0:
            print("Converged at epoch %d" % (j + 1))
            break
        if errors != 0 and j >= epochs -1:
            print("Not converged")
    print('Epoch %d: Accuracy = %f, errors = %d' % (j + 1, acc, errors))
    return weights[:-1], weights[-1]

# load dataset
dataset, num_classes, num_feats, labels_str = load_dataset('iris.data')

# with lb we choose which class is 1 and all others are considered -1
# for example if lb = 0 Iris-setosa will be labeled with 1 and the others with -1
lb_list = [0, 1, 2]
lr = 1
for lb in lb_list:
    print('For labeling-> %s = 1 Others = -1' % (labels_str[lb].decode("utf-8")))
    tr_dataset = transform_labels(dataset, lb)

    data = tr_dataset[:, :num_feats]
    targets = tr_dataset[:, -1].astype(int)

    w, bias = perceptron_classifier(data, targets, lr)
    print('weights: ', w)
    print('bias(w0): ', bias)

