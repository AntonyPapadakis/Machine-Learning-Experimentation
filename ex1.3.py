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

X, Fx20 = p.getPoints(N,start,end,6)
#get the yn's
Y = lr.getY(N, X, noise, thetaTransposed).T

real_mean = np.mean(Y)
real_variance = np.var(Y)

'''
----------------------------------------
#create the test set
----------------------------------------
'''
N=10000
# seed random number generator
seed(1)
# generate 1000 random numbers between 0-2
upper_bound = 2
lower_bound = 0
test_X=[]

for _ in range(N):

    if lower_bound != 0:
        value = random()*(upper_bound-lower_bound)*lower_bound
    else:
        value = random()*upper_bound

    test_X.append(value)

test_X = np.array(test_X, dtype=float)

Fx1000 = np.ones((N,6))
for i in range(6):
    for j in range(N):
        Fx1000[j,i] = math.pow(test_X[j], i)

#noise
noise = gn.getNoise(N,mean,variance)

#get the yn's
Y_test = lr.getY(N, test_X, noise, thetaTransposed).T


#ridge regression
lam = {0, 0.5, 0.002, 0.005, 50, 1, 2, 10, math.pow(variance,2) / math.pow(theta[0],2) }

for l in lam:

    '''
    first step of the experiment using N=20
    '''
    N=20

    #there are more data points than there are parameters to be determined
    thetaPredicted, Fx = ls.getY(N, Fx20, Y, 6, l)

    #predicted y values
    Y_pred = np.dot(thetaPredicted.T,Fx.T)

    #mean square error over training set
    MSE = m.MSE(Y_pred, Y, N)

    print("the MSE for the training set N=20 is:", MSE,"for lamda=",l)
    #blue for actual values red for predicted
    plt.figure(num=1)
    plt.title("training set ridge regression for lamda="+str(l))
    plt.plot(X,Y,'bo',X,Y_pred[0],'ro')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


    '''
    second step of the experiment using N=1000
    '''
    N=10000


    #theta predicted using the least squares method
    thetaPredicted, Fx = ls.getY(N, Fx1000, Y_test, 6, l)

    #predicted y values over the testing set
    Y_pred_test = np.dot(thetaPredicted.T,Fx.T)

    #mean square error over test set
    MSE_test = m.MSE(Y_pred_test,Y_test,N)

    print("the MSE for the test set N=1000 is:", MSE_test," for lamda=",l)

    #blue for actual values red for predicted
    plt.figure(num=3)
    plt.title("test set ridge regression for lamda="+str(l))
    plt.plot(test_X,Y_test,'bo',test_X,Y_pred_test[0],'ro')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()