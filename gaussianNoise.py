import numpy as np
import random as r
import math


def getNoise(N,mean,variance):
    """Getting gaussian noise"""

    #get std
    std = math.sqrt(variance)

    #initialize f
    f = np.zeros((1, N))

    #get the random x's
    for i in range(0, N):

        f[0,i]= np.random.normal(mean,std,1)


    return f