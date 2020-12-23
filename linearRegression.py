import numpy as np
import math

'''
Getting the yn's 
'''
def getY(N,X,noise,thetaT):

        y = np.zeros((1, N))
        for n in range(0, N):
            for i in range(0, 6):
                y[0,n] += thetaT[0, i]*math.pow(X[n], i)
            y[0,n] += noise[0,n]
        return y


def getYNonoise(N,X,thetaT):

    y = np.zeros((1, N))
    for n in range(0, N):
        for i in range(0, 6):
            y[0,n] += thetaT[0, i]*math.pow(X[n], i)
        y[0,n] += 0
    return y