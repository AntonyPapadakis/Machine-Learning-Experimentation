import numpy as np
import pandas as pd


def loaddataset():
    datafile = "../UCIdata-exercise1/pima-indians-diabetes.data"
    namesfile = "../UCIdata-exercise1/pima-indians-diabetes.names"

    names = open(namesfile).readlines()
    for i in names: print(i)

    data = np.genfromtxt(datafile, delimiter=",")
    features = data[:, :-1]
    category = data[:, -1:]

    return features, category


def main():
    print("hello")
    features, category = loaddataset()


if __name__ == "__main__":
    main()
