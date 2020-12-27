import numpy as np

'''
Getting the theta and Y prediction using the least squares
method and linear regression
'''
def getY(N,Fx,Y,degree,lam=0):

    #for ridge regression
    lI = lam*np.identity(degree)


    #extend = np.ones((N,degree))

    FxT = Fx.T

    inv = np.linalg.inv(np.dot(FxT,Fx) + lI)

    temp = np.dot(inv, FxT)

    thetaPrediction = np.dot(temp, Y)


    return (thetaPrediction, Fx)