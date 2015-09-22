import numpy as np;

def sigmoid(x, deriv = False):
    if(deriv == True):
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self, aWeights):
        '''weights should be a (L+1) vector'''
        self._weights = aWeights

    def out(self, input):
        '''input should be a (L+1) vector, should have a -1 as last value for threshold weight'''
        return sigmoid(np.dot(input, self._weights)[0][0])


class Layer:
    def __init__(self, numNeurons = 1):
        self._neurons = []
        self._neurLength = numNeurons
        self._nextLayer = None

    def connect(self, layer):
        '''A -> B : B.connect(A)'''
        weightLength = layer._neurLength + 1
        for i in range(0, self._neurLength):
            self._neurons.append(Neuron(2*np.random.random((weightLength)) - 1))
        layer._nextLayer = self


class Network:
    def __init__(self, inp, hid, out):
        self._input = Layer(inp)
        self._output = Layer(out)
        self._hidden = []
        for i in hid:
            self._hidden.append(Layer(i))

    def out(self):
        None

    def train(self):
        None
'''
To do:
- input -> output
- training
- genetic algorithm 
'''
