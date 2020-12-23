import numpy as np
import random as r
import math

'''
Getting gaussian noise
'''
def getNoise(N,mean,variance):

    #get std
    std = math.sqrt(variance)

    #initialize f
    f = np.zeros((1, N))

    #get the random x's
    for i in range(0, N):
        #x[0,i] = r.uniform(-std, std)
        #f[0,i] = (1/(std*math.sqrt(2*math.pi)) )* math.exp(-1/2 * math.pow((x[0,i]-mean)/std, 2))
        f[0,i]= np.random.normal(mean,std,1)


    return f