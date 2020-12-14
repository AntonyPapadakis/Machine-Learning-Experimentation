import numpy as np


'''
Getting N equidistant points
'''
def getPoints(N,start,end):

    maxDistance = end - start

    #since our N points are equidistant
    pointDistance = maxDistance / N

    points = np.arange(start,end,pointDistance)

    return points