import numpy as np
import random as r
import math

'''
Getting gaussian noise
'''
def getNoise(N,mean,variance):

    #get std
    std = math.sqrt(variance)

    #initialize x
    x = np.zeros((1, N))
    f = np.zeros((1, N))

    #get the random x's
    for i in range(0, N):
        x[0,i] = r.uniform(0.0, std)
        f[0,i] = 1/(std*math.sqrt(2*math.pi)) * math.exp(-1/2 * math.pow((x[0,i]-mean)/std, 2))

    return f