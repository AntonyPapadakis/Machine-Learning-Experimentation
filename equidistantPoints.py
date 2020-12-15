import numpy as np
import math


'''
Getting N equidistant points
'''
def getPoints(N,start,end):

    maxDistance = end - start

    #since our N points are equidistant
    pointDistance = maxDistance / N

    points = np.arange(start,end,pointDistance)

    Fx = np.ones((N,5))
    for i in range(5):
        for j in range(20):
            Fx[j,i] = math.pow(points[j],i)

    return (points, Fx)