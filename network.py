import numpy as numpy;

def sigmoid(x, deriv = False):
    if(deriv == True):
        return x*(1-x)
    else:
        return 1/(1+numpy.exp(-x))

class Neuron:
    def __init__(self):

class Layer:
    def __init__(self):

class Network:
    def __init__(self):
