import numpy as np

'''
Getting the theta and Y prediction using the least squares
method and linear regression
'''
def getY(N,Fx,Y):

    extend = np.ones((N,6))


    extend[0:N,0:5] = Fx[0:N,0:5]
    Fx=extend
    FxT = Fx.T

    inv = np.linalg.pinv(np.dot(FxT,Fx))

    temp = np.dot(inv, FxT)

    thetaPrediction = np.dot(temp, Y)


    return (thetaPrediction, Fx)