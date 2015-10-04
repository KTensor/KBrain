import numpy as np
import random as random

#KEVIN WANG
#github:xorkevin

def sigmoid(x, deriv = False):
    if deriv == True:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

def lninv(x):
    return 1/(np.log(x+2))

    
class Unit:
    def __init__(self, aSize):
        self._size = aSize

    def setWeights(self, zX, zH, rX, rH):
        '''parameters: update input, update history, reset input, reset history; each should be an input length vector, history length vector'''
        pass

    def step(self, input):
        '''intput should be a ...'''
        z =
