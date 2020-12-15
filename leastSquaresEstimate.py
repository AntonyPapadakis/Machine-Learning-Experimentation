import numpy as np
import math

'''
Getting the theta and Y prediction using the least squares
method and linear regression
'''
def getY(N,X,Y):

    extend = np.ones((2, N))

    extend[0,1:N] = X[1:N]
    X=extend
    X=X.T
    XT = X.T
    inv = np.linalg.inv(np.dot(XT, X))

    temp = np.dot(inv, XT)

    thetaPrediction = np.dot(temp, Y)


    return thetaPrediction