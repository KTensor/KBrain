import numpy as np
import random as random

#KEVIN WANG
#github:xorkevin

def sigmoid(x, deriv = False):
    if deriv == True:
        return 4*x*(1-x)
    else:
        return 1/(1+np.exp(-4*x))

def sigmoid2(x, deriv = False):
    if deriv == True:
        return 4*x*(1-x/2)
    else:
        return 2/1+np.exp(-4*x) - 1

def lninv(x):
    return 1/(np.log(x+2))


class Unit:
    def __init__(self, aSize):
        self._size = aSize

    def setWeights(self, zX, zH, rX, rH, hX, hH):
        '''parameters: update input, update history, reset input, reset history; each should be an input length vector, history length vector'''
        self._zX = None
        self._zH = None
        self._rX = None
        self._rH = None
        self._hX = None
        self._hH = None

    def setInitStep(self, firstInput):
        '''input should be a ...'''
        self._h = firstInput

    def step(self, input):
        '''intput should be a ...'''
        z = sigmoid(self._zX.multiply(input) + self._zH.multiply(self._h))
        r = sigmoid(self._rX.multiply(input) + self._rH.multiply(self._h))
        deltaH = sigmoid2(self._hX.multiply(input) + r.multiply(self._hH.multiply(self._h)))
        self._h = z.multiply(self._h) + (1-z).multiply(deltaH)
