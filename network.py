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

    def out(self, aInput):
        '''input should be a (L+1) vector, should have a -1 as last value for threshold weight'''
        return sigmoid(np.dot(aInput, self._weights))


class Layer:
    def __init__(self, numNeurons = 1):
        self._neurons = []
        self._neurLength = numNeurons
        self._nextLayer = None
        self._prevLayer = None

    def connect(self, layer):
        '''A -> B : B.connect(A)'''
        self._neurons = []
        weightLength = layer._neurLength + 1
        for i in range(0, self._neurLength):
            self._neurons.append(Neuron(2*np.random.random(weightLength) - 1))
        layer._nextLayer = self
        self._prevLayer = layer

    def out(self, aInput):
        '''input should be a (L) vector'''
        inp = aInput.append(-1)
        arr = np.array([])
        for(neuron in self._neurons):
            arr.append(neuron.out(inp))
        return arr


class Input(Layer):
    def __init__(self, length):
        super().__init__(self, length)
        self._inputs = np.array([])

    def out(self, aInput):
        self._inputs = aInput
        return self._inputs


class Network:
    def __init__(self, inp, hid, out):
        self._input = Input(inp)
        self._output = Layer(out)
        self._hidden = []
        for i in hid:
            self._hidden.append(Layer(i))

    def out(self):
        pass

    def train(self):
        pass
'''
To do:
- training
- genetic algorithm
'''
