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
from scipy.stats import norm


#N=20 equidistant points in the interval [0,2]
N=20
start=0
end=2
variance=0.1
mean=0


#initializing weight vector theta
theta = np.zeros((6,1))
thetaValues = [0.2, -1, 0.9, 0.7,  0, -0.2]

for i in range(0, 6):
    theta[i] = thetaValues[i]

thetaTransposed = np.transpose(theta)

#gaussian noise
noise = gn.getNoise(N,mean,variance)

X_true, Fx = p.getPoints(N,start,end,6)
#get the yn's
Y_true = lr.getY(N, X_true, noise, thetaTransposed).T

real_mean = np.mean(Y_true)
real_variance = np.var(Y_true)

plt.figure(num=1)
plt.title("real curve")

plt.plot(X_true, norm.pdf(Y_true), 'b-', lw=5, alpha=0.95, label='norm pdf')

plt.xlabel('X')
plt.ylabel('probability')
plt.show()

for degree in {3,11,6}: #3 for 2nd degree and 11 for 10th degree polynomial

    plt.figure(num=degree)
    plt.title("experiments' curves "+ str(degree-1) +" degree polynomial")

    plt.plot(X_true, norm.pdf(Y_true), 'b-', lw=5, alpha=0.6, label='norm pdf')

    l_means = [] #list of means
    l_variances = [] #list of variance
    points = np.zeros((100,20))

    for i in range(0,100):

        #gaussian noise
        noise = gn.getNoise(N,mean,variance)

        X, Fx = p.getPoints(N,start,end,degree)
        #get the yn's
        Y = lr.getY(N, X, noise, thetaTransposed).T

        #there are more data points than there are parameters to be determined
        thetaPredicted, Fx = ls.getY(N,Fx,Y,degree)

        #predicted y values
        Y_pred = np.dot(thetaPredicted.T,Fx.T)

        points[i, 0:20] = Y_pred[0:20]
        l_means.append(np.mean(Y_pred))
        l_variances.append(np.var(Y_pred))

        #curves
        plt.plot(X, norm.pdf(Y), 'r-', lw=5, alpha=0.6, label='pdf experiments')


        #mean square error over training set
        MSE = m.MSE(Y_pred,Y,N)


    plt.xlabel('X')
    plt.ylabel('probability')
    plt.show()

    #green for actual values red for predicted
    plt.figure(num=degree-1)
    plt.title("means/variances " + str(degree-1) + " degree polynomial")

    for j in range(0,100):
        plt.plot(l_means[j], l_variances[j], 'ro')

    plt.plot(real_mean, real_variance, 'go')
    plt.xlabel('mean')
    plt.ylabel('variance')
    plt.show()


