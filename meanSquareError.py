import numpy as np
import math

'''
Y is the column vector containing the set of yns and
Y_pred is the predicted ys represented in a l x N array 
where N is the number of points in the training set
'''
def MSE(Y_pred,Y,N):

    temp = []
    for i in range(0, len(Y_pred)):
        sub = np.subtract(Y.T, Y_pred[i])
        msq_error=0
        for i in range(0, N):
            sq_error = math.pow(sub[0,i], 2)
            print("the square error for n = ",i,"is: ",sq_error)
            msq_error += sq_error
        msq_error = msq_error/N
        temp.append(msq_error)

    MSE = min(temp)
    return MSE