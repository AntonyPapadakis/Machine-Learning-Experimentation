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
variance=0.1 #NOISE variance
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
N=1000
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
lam = {0, 0.5, 50, 2, 10, math.pow(variance,2) / math.pow(theta[0],2) }


print(math.pow(variance,2) / math.pow(theta[0],2))

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)
fig.set_size_inches(25, 25)
fig.set_dpi(200)

fig.suptitle('N=20 plots')

fig2, (ax21, ax22, ax23, ax24, ax25, ax26) = plt.subplots(6, 1)
fig2.set_size_inches(25, 25)
fig2.set_dpi(200)

fig2.suptitle('N=1000 plots')

l_ax1 = [ax1, ax2, ax3, ax4, ax5, ax6]
l_ax2 = [ax21, ax22, ax23, ax24, ax25, ax26]

for l,ax1,ax2 in zip(lam,l_ax1,l_ax2):

    '''
    first step of the experiment using N=20
    '''
    N=20

    #there are more data points than there are parameters to be determined
    thetaPredicted, Fx = ls.getY(N, Fx20, Y, 6, l)

    #predicted y values
    Y_pred = np.dot(thetaPredicted.T,Fx.T)

    #true values without noise
    Y1 = lr.getYNonoise(N, X, thetaTransposed)


    #mean square error over training set
    MSE = m.MSE(Y_pred, Y, N)

    print("the MSE for the training set N=20 is:", MSE,"for lamda=",l)
    #blue for actual values red for predicted
    ax1.set_title("training set ridge regression for lamda="+str(l))
    ax1.plot(X,Y1[0],'bo',X,Y_pred[0],'ro')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    '''
    second step of the experiment using N=1000
    '''
    N=1000

    #predicted y values over the testing set
    Y_pred_test = np.dot(thetaPredicted.T,Fx1000.T)

    #true values without noise
    Y1 = lr.getYNonoise(N, test_X, thetaTransposed)

    #mean square error over test set
    MSE_test = m.MSE(Y_pred_test,Y_test,N)

    print("the MSE for the test set N=1000 is:", MSE_test," for lamda=",l)

    #blue for actual values red for predicted
    ax2.set_title("test set ridge regression for lamda="+str(l))
    ax2.plot(test_X,Y1[0],'bo',test_X,Y_pred_test[0],'ro')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

plt.show()
