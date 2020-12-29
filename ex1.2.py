import numpy as np
import equidistantPoints as p
import gaussianNoise as gn
import linearRegression as lr
import leastSquaresEstimate as ls
import meanSquareError as m
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

X, Fx = p.getPoints(N,start,end,6)
#get the yn's
Y_true = lr.getYNonoise(N, X, thetaTransposed).T

real_mean = np.mean(Y_true)
real_variance = np.var(Y_true)

plt.figure(num=1)
plt.title("X vs Y")

plt.plot(X, Y_true, 'b-', lw=5, alpha=0.95, label='norm pdf')

plt.xlabel('X')
plt.ylabel('Y')
plt.show()

x = np.linspace(real_mean - 3*math.sqrt(real_variance), real_mean + 3*math.sqrt(real_variance), 100)



for degree in {11,3}: #3 for 2nd degree and 11 for 10th degree polynomial

    plt.figure(num=degree)
    plt.title("experiments' X vs Y "+ str(degree-1) +" degree polynomial")

    l_means = [] #list of means
    l_variances = [] #list of variance
    points = np.zeros((100,N))

    for i in range(0,100):

        #gaussian noise
        noise = gn.getNoise(N,mean,variance)

        Fx = p.getF(N,X,degree)
        #get the yn's
        Y = lr.getY(N, X, noise, thetaTransposed).T

        #there are more data points than there are parameters to be determined
        thetaPredicted, Fx = ls.getY(N,Fx,Y,degree)

        #predicted y values
        Y_pred = np.dot(thetaPredicted.T,Fx.T)

        points[i, 0:N] = Y_pred[0:N]
        l_means.append(np.mean(Y_pred))
        l_variances.append(np.var(Y_pred))

        #curves


        #mean square error over training set
        MSE = m.MSE(Y_pred,Y,N)


    #variance and mean of each y_pred point
    variance_ypred = np.zeros((1,N))
    mean_ypred = np.zeros((1,N))
    for i in range(N):
        variance_ypred[0,i] = np.var(points[:,i])
        mean_ypred[0,i] = np.mean(points[:,i])



    #true curve - blue

    plt.plot(X, Y_true, 'b-', lw=5, alpha=0.6, label='norm pdf')
    plt.xlabel('X')
    plt.ylabel('Y')

    #predicted curve - red

    plt.plot(X,mean_ypred[0],'r-')
    plt.errorbar(X,mean_ypred[0], yerr=variance_ypred[0] ,fmt='none', color='r')

    plt.show()

    #green for actual values red for predicted
    plt.figure(num=degree-1)
    plt.title("Curves " + str(degree-1) + " degree polynomial")


    print(degree-1," polynomial case")
    print("predicted values variance",variance_ypred)
    print("predicted values means",mean_ypred)
    print("MSE ", MSE)
    print("real variance", real_variance)
    print("real mean", real_mean)
    print("mean of means: ",sum(l_means)/len(l_means))
    print("mean of variance: ", sum(l_variances)/len(l_variances))

    for j in range(0,100):
        plt.plot(x,norm.pdf(x,l_means[j], l_variances[j]), 'ro')

    plt.plot(x,norm.pdf(x,real_mean, real_variance), 'go')
    plt.xlabel('X')
    plt.ylabel('pdfunction value')
    plt.show()


