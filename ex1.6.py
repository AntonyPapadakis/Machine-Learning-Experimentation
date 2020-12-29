import numpy as np
import equidistantPoints as p
import gaussianNoise as gn
import linearRegression as lr
import meanSquareError as m
import matplotlib.pyplot as plt
import math

'''
Expectation-Maximization

training set with N=500 and
noise variance = 0.05
noise mean =0

'''

#initializing weight vector theta
theta = np.zeros((6,1))
thetaValues = [0.2, -1, 0.9, 0.7,  0, -0.2]

for i in range(0, 6):
    theta[i] = thetaValues[i]

thetaTransposed = np.transpose(theta)

#number of points
N=500
start=0
end=2
X, Fx = p.getPoints(N,start,end,6)


#noise
variance=0.05
mean=0
noise = gn.getNoise(N,mean,variance)

#get the yn's
Y = lr.getY(N, X, noise, thetaTransposed).T

#EM algorithm

'''
initialization

a = 1/variance_theta
b = 1/variance_noise
'''

a=1 #a0
b=1 #b0
a_1=0.8
b_1=18.1
I = np.identity(6)
K=1

v_noise0=1
variance_noise=1.5
keep_var_noise=[]
keep_var_noise.append(v_noise0)

epsilon = 0.00001
Cov_thetaY=0
mean_thetaY=0



iterations=0 #EM iterations
while np.abs(variance_noise-v_noise0)>epsilon:

    Cov_thetaY = np.linalg.inv(np.dot(a,I) + np.dot(b,np.dot(Fx.T, Fx)))
    mean_thetaY = np.dot(b,np.dot(Y.T,np.dot(Cov_thetaY,Fx.T).T))

    if(iterations>0):
        keep_var_noise.append(variance_noise)
    v_noise0=variance_noise

    a=a_1
    b=b_1

    a_1 = K/(math.pow(np.linalg.norm(mean_thetaY),2)+np.trace(Cov_thetaY))

    c=math.pow(np.linalg.norm(Y-np.dot(Fx,mean_thetaY.T)),2)
    b_1 = N/(c+np.trace(np.dot(Fx,np.dot(Cov_thetaY,Fx.T))))


    variance_theta = 1/a_1
    variance_noise = 1/b_1
    iterations+=1
    print(variance_noise)


variance_noise = v_noise0

'''
----------------------------------------------------
ploting variance noise change through EM iterations
-----------------------------------------------------
'''

#plot for noise variance through EM iterations
plt.figure(num=1)
plt.title("noise variance vs EM iterations")
plt.plot(range(0,iterations),keep_var_noise,'r-')
plt.xlabel('iterations')
plt.ylabel('noise variance')
plt.show()



'''
-----------------------------------------
time for the test set
N=20 in the interval [0,2]
-----------------------------------------
'''

#points
N=20
start=0
end=2
X, Fx = p.getPoints(N,start,end,6)
Y_true= lr.getYNonoise(N, X, thetaTransposed)

#predictions
Y_pred = np.dot(mean_thetaY,Fx.T)
variance_Y = []

for i in range(N):
    #for the error bars
    variance_Y.append(variance_noise + np.dot(Fx[i],Cov_thetaY) @ Fx[i].T)



errors = np.abs(Y_true-Y_pred)
print("errors:",errors)

#mean square error over test set
MSE_test = m.MSE(Y_pred,Y_true,N)

print("the MSE for the test set is:", MSE_test)

#blue for true values red for predicted
plt.figure(num=2)
plt.title("true(red) vs predictions and their errorbars(variance of y predictions)")

plt.plot(X,Y_true.T,'r-',X,Y_pred.T,'bo')
plt.errorbar(X,Y_pred.T, yerr=variance_Y ,fmt='none', color='b')

plt.xlabel('X')
plt.ylabel('Y')
plt.show()

