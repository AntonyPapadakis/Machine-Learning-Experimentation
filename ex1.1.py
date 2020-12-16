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

'''
-------------------------------------------------------------------------------------------
Exercise 1:
-------------------------------------------------------------------------------------------
generalised regression model
y = theta0 + theta1 * x + theta2 * x^2 + theta3 * x^3 + theta5 * x^5 + noise

where noise corresponds to white Gaussian noise

and the components of the weight vector theta assume the values:
theta0 = 0.2
theta1 = -1
theta2 = 0.9
theta3 = 0.7
theta4 =  0
theta5 = -0.2

so:
theta = transposed [0.2 -1 0.9 0.7 -0.2]

In all cases of exercise 1 we consider N equidistant points x1,x2, ... xn n=1, 2, ... , N
all the points are in the interval [0,2]

our training set consists of
yn = theta0 + theta1 * xn + theta2 * xn^2 + theta3 * xn^3 + theta5 * xn^5 + noise_n n=1, 2, ... ,N

noise_n are noise samples coming from a Gaussian distribution with 
mean=0, variance= sigma_n^2

--------------------------------------------------------------------------------------------
'''
'''
--------------------------------------------------------------------------------------------
For question 1.1:

Consider N=20, sigma_n^2=0.1
and
y = theta0 + theta1 * x + theta2 * x^2 + theta3 * x^3 + theta5 * x^5 + noise

1)Use Least Squares method to estimate the parameter vector
2)Calculate MSE of y over
a)the training set 
b)a test set of 1000 points in the interval [0,2]
--------------------------------------------------------------------------------------------
'''

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

X, Fx = p.getPoints(N,start,end,6)

#noise
variance=0.1
mean=0
noise = gn.getNoise(N,mean,variance)

#get the yn's
Y = lr.getY(N, X, noise, thetaTransposed).T

#there are more data points than there are parameters to be determined
thetaPredicted, Fx = ls.getY(N,Fx,Y,6)

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
for _ in range(1000):

    if lower_bound != 0:
        value = random()*(upper_bound-lower_bound)*lower_bound
    else:
        value = random()*upper_bound

    test_X.append(value)


test_X = np.array(test_X, dtype=float)
Fx = np.ones((N,6))
for i in range(6):
    for j in range(20):
        Fx[j,i] = math.pow(test_X[j], i)

#noise
noise = gn.getNoise(N,mean,variance)

#get the yn's
Y_test = lr.getY(N, test_X, noise, thetaTransposed).T

#theta predicted using the least squares method
thetaPredicted, Fx = ls.getY(N,Fx,Y_test,6)

#predicted y values over the testing set
Y_pred_test = np.dot(thetaPredicted.T,Fx.T)

#mean square error over test set
MSE_test = m.MSE(Y_pred_test,Y_test,N)

print("the MSE for the test set is:", MSE_test)

#blue for actual values red for predicted
plt.figure(num=3)
plt.title("test set 1 ")
plt.plot(test_X,Y_test,'bo',test_X,Y_pred_test[0],'ro')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

'''
plt.figure(num=4)
plt.title("test set 2 ")
plt.plot(test_X,Y_test,'bo',test_X,Y_pred_test[1],'ro')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
'''