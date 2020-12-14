import numpy as np
import equidistantPoints as p
import gaussianNoise as gn
import linearRegression as lr
import leastSquaresEstimate as ls
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
#print(theta)

thetaTransposed = np.transpose(theta)
#print(thetaTransposed)

#N=20 equidistant points in the interval [0,2]
N=20
start=0
end=2

points = p.getPoints(N,start,end)
#print(points)

#noise
variance=0.1
mean=0
noise = gn.getNoise(N,mean,variance)
#print(noise)

#get the yn's
Y = lr.getY(N, points, noise, thetaTransposed)
#print(Y)

#there are more data points than there are parameters to be determined
thetaPredicted = ls.getY(N,points,Y)
#print(thetaPredicted)



#mean square error over training set
msq_error = 0
for i in range(0, N):
   print(thetaPredicted[i])
   print(Y)
   sub = np.subtract(Y,np.dot(thetaPredicted[i].T,points[i]))

   for i in range(0, N):
       msq_error += math.pow(sub[0,i], 2)

   msq_error = msq_error/N

msq_error = msq_error/N
print(msq_error)