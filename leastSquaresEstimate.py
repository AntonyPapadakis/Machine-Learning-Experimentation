import numpy as np
import math

'''
Getting the theta estimates using the least squares
method and linear regression
'''
def getY(N,X,Y):

    thetaPrediction = np.zeros((1,N))
    XT = np.transpose(X)

    #thetaPrediction=np.linalg.inv(XT*X)*XT*Y
    thetaPrediction=(XT*X)*XT*Y

    return thetaPrediction