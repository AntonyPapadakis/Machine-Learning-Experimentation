import numpy as np

'''
Getting the theta and Y prediction using the least squares
method and linear regression
'''
def getY(N,Fx,Y,degree):

    extend = np.ones((N,degree))


    extend[0:N,0:degree] = Fx[0:N,0:degree]
    Fx=extend
    FxT = Fx.T

    inv = np.linalg.pinv(np.dot(FxT,Fx))

    temp = np.dot(inv, FxT)

    thetaPrediction = np.dot(temp, Y)


    return (thetaPrediction, Fx)