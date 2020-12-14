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
    XT = X.T
    inv = np.linalg.inv(np.dot(XT, X))

    extend = np.ones((2, N))
    extend[0,1:N] = Y[0,1:N]
    Y = extend
    thetaPrediction = np.dot(np.dot(inv, XT), Y)



    return thetaPrediction