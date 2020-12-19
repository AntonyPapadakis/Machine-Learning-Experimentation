import numpy as np
import equidistantPoints as p
import gaussianNoise as gn
import linearRegression as lr
import leastSquaresEstimate as ls
import meanSquareError as m
from random import random
from random import seed
import matplotlib.pyplot as plt
import math


N=20
variance=0.1
mean=0

#initializing weight vector theta
theta = np.zeros((6,1))
thetaValues = [0.2, -1, 0.9, 0.7,  0, -0.2]

for i in range(0, 6):
    theta[i] = thetaValues[i]

thetaTransposed = np.transpose(theta)

#N=20 equidistant points in the interval [0,2]
N=20
start=0
end=2

X, Fx = p.getPoints(N,start,end,3)

#noise
variance=0.1
mean=0
noise = gn.getNoise(N,mean,variance)

#get the yn's
Y = lr.getY(N, X, noise, thetaTransposed).T

#there are more data points than there are parameters to be determined
thetaPredicted, Fx = ls.getY(N,Fx,Y,3)

#predicted y values
Y_pred = np.dot(thetaPredicted.T,Fx.T)

#mean square error over training set
MSE = m.MSE(Y_pred,Y,N)

print("the MSE is:", MSE)
#blue for actual values red for predicted
plt.figure(num=1)
plt.title("training set")
plt.plot(X,Y,'bo',X,Y_pred[0],'ro')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
