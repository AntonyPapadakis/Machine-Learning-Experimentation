import numpy as np
from numpy import genfromtxt


def load_dataset(fname):
    # loads a dataset assuming the last column is the labela
    # return the number of labels as well as the number of feats
    my_data = genfromtxt(fname, delimiter=',', dtype='S', encoding=None)
    num_feats = my_data.shape[1]-1
    my_data_feats = np.array(my_data[:, :num_feats], dtype='float32')
    labels, indexed_data = np.unique(my_data[:, -1], return_inverse=True)
    my_data_labels = np.array(indexed_data)
    dataset = np.hstack((my_data_feats, my_data_labels[:, None]))
    return [dataset, labels.shape[0], my_data_feats.shape[1]]


def knn_classify(point, data, target, k, num_classes):
    dists = np.zeros(data.shape[0])
    # calculate distance of point to every other point in the training dataset
    for i, d in enumerate(data):
        dists[i] = np.sqrt(np.sum((d-point)**2))
        # dists[i] = np.sum((d-point)**2) squared # euclidian
        # dists[i] = np.sum(np.absolute(d-point)) # mahalanobis
    assert np.count_nonzero(dists==0) == 0 # check that it is not exactly matching a point for debugging
    # get the indices of the top k nearest neighbours
    idxs = np.argsort(dists)[:k]
    # perform a simple majority voting to determine the predicted class
    l_counts = np.zeros(num_classes)
    for i in idxs:
        lb = target.item(i)
        l_counts[lb] += 1
    pred_class = np.argmax(l_counts)
    return [point, pred_class]


def cross_val_split(dataset, folds):
    # split the dataset into folds number of subsets of almost equal size after randomly shuffling it
    cv_dataset = dataset.copy()
    np.random.shuffle(cv_dataset)
    splits = np.array_split(cv_dataset, folds)
    return splits


dataset, num_classes, num_feats = load_dataset('iris.data')

# remove duplicate rows
a = dataset
dataset, count = np.unique(a, axis=0, return_counts=True)


kmax= 20
# perform cross validation for multiple k
mean_accs = np.zeros(kmax + 1)
# set seed
np.random.seed(0)
for k in range(1, kmax + 1):
    print("Running for k = %d" % k)
    folds = 5
    data_splits = cross_val_split(dataset, folds)
    acc = np.zeros(folds)
    sums = np.zeros(folds)
    # create the test/train data by keeping 1 fold as test
    # and all the others combined as train
    # in the and average the results over all different combinations
    for i in range(0, folds):
        test_set = data_splits[i]
        train_idxs = list(range(0, folds))
        train_idxs.remove(i)
        list_of_splits = []
        # print(train_idxs)
        # print(i)
        for j in train_idxs:
            list_of_splits.append(data_splits[j])
        train_set = np.vstack(list_of_splits)

        # knn for test set
        test_data = test_set[:, :num_feats]
        test_target = test_set[:, -1].astype(int)
        train_data = train_set[:, :num_feats]
        train_target = train_set[:, -1].astype(int)
        pred = np.zeros(test_target.shape, dtype=int)
        for n, p in enumerate(test_data):
            point, pred_class = knn_classify(p, train_data, train_target,  k, num_classes)
            pred[n] = pred_class
        # calculate accuracy for the current split
        sums[i] = np.sum(np.where(pred == test_target, 1, 0))
        acc[i] = sums[i] / pred.shape[0]
    # store the mean accuracy of cross validation sets for each k
    mean_accs[k] = np.mean(acc)
print (mean_accs)

