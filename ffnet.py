import numpy as np
from ast import literal_eval as ast_literal_eval
import sys
import random as random

#KEVIN WANG
#github:xorkevin

def sigmoid(x, deriv = False):
    if deriv == True:
        return (1 - x*x)*1.2
    else:
        return 1.2*np.tanh(x)

def sig(x, deriv = False):
    if deriv == True:
        if x > 2:
            return 0.001359
        elif x < -2:
            return 0.001359
        return 4*x*(1-x)
    else:
        if x > 2:
            return 1
        elif x < -2:
            return 0
        return 1/(1+np.exp(-4*x))

def softmax(w):
    e = np.exp(w - np.amax(w))
    dist = e / np.sum(e)
    return dist

def lninv(x):
    y = 1/(np.log(x+2))
    if y < 0.0056:
        y = 0.0056
    return y


class Layer:
    def __init__(self, numNeurons = 1):
        self._neurons = np.matrix([])
        self._neurLength = numNeurons
        self._weightLength = None
        self._nextLayer = None
        self._prevLayer = None

    def connect(self, layer):
        '''A -> B : B.connect(A)'''
        self._weightLength = layer._neurLength + 1
        layer._nextLayer = self
        self._prevLayer = layer

    def setNeurons(self, neurarray):
        '''neuron array should be a 2D matrix, rows are neurons, columns are weights, dim is neurLengthxweightLength'''
        self._neurons = neurarray.T

    def getNeurons(self):
        return self._neurons.T

    def out(self, aInput):
        '''input should be a (weightLength - 1) vector'''
        inp = np.append(aInput, -1)
        val = np.ravel(inp.dot(self._neurons))
        for i in range(0, val.size):
            val[i] = sigmoid(val[i])
        return val

    def __str__(self):
        x = '['
        n = self.getNeurons()
        r,c = n.shape
        for i in range(0, r):
            x += '['
            for j in range(0, c):
                x += str(n[i, j])+','
            x += '],'
        x += ']'
        return x


class Input(Layer):
    def out(self, aInput):
        return aInput

class Output(Layer):
    def out(self, aInput):
            inp = np.append(aInput, -1)
            val = np.ravel(inp.dot(self._neurons))
            for i in range(0, val.size):
                val[i] = sig(val[i])
            return val

class MutOutput(Layer):
    def out(self, aInput):
        inp = np.append(aInput, -1)
        return softmax(np.ravel(inp.dot(self._neurons)))

class Network:
    def __init__(self, inp=None, hid=None, out=None, layerWeights = None, exclusive = False):
        if inp is not None:
            self._input = Input(inp)
            self._output = Output(out)
            self._exclusive = exclusive
            if self._exclusive:
                self._output = MutOutput(out)
            self._layers = []
            self._layers.append(self._input)
            for i in hid:
                self._layers.append(Layer(i))
            self._layers.append(self._output)

            for i in range(1, len(self._layers)):
                self._layers[i].connect(self._layers[i-1])
                if layerWeights is None:
                    self._layers[i].setNeurons(np.random.rand(self._layers[i]._neurLength, self._layers[i]._weightLength) * 2 - 1)
                else:
                    self._layers[i].setNeurons(layerWeights[i-1])

            del self._layers[0]

    def save(self, filename):
        f = open(filename, 'r+')
        f.truncate()
        f.write(self.__str__())
        f.close()

    def load(self, filename):
        f = open(filename, 'r')
        self._exclusive = f.readline().strip()=='True'
        self._input = Input(int(f.readline().strip()))

        self._layers = []
        self._layers.append(self._input)

        hid = map(int, f.readline().strip().split(' '))
        for i in hid:
            self._layers.append(Layer(i))

        z = int(f.readline().strip())
        self._output = Output(z)
        if self._exclusive:
            self._output = MutOutput(z)
        self._layers.append(self._output)

        for i in range(1, len(self._layers)):
            self._layers[i].connect(self._layers[i-1])
            self._layers[i].setNeurons(np.matrix(ast_literal_eval(f.readline().strip())))

        del self._layers[0]
        f.close()

    def out(self, aInput):
        arr = aInput
        for layer in self._layers:
            arr = layer.out(arr)
        return arr

    def trainOut(self, aInput):
        arr = aInput
        history = []
        for layer in self._layers:
            arr = layer.out(arr)
            history.append(np.append(arr, -1))
        return history

    def train(self, example, rate):
        '''example is a tuple x, y; both arrays input and output'''
        x, y = example

        history = self.trainOut(x)

        deltas = []

        prediction = np.delete(history[-1], -1)
        actual = y
        error = (prediction - actual)
        slope = [sig(i, True) for i in prediction]
        deltaOut = np.multiply(error, slope)
        if self._exclusive:
            deltaOut = error
        deltas.insert(0, deltaOut)

        for i in range(len(history)-2, -1, -1):
            delta = np.array([])
            kNeuron = self._layers[i+1].getNeurons()
            for j in range(0, len(self._layers[i].getNeurons())):
                s = 0
                for k in range(0, len(kNeuron)):
                    s += np.ravel(kNeuron[k])[j]*deltas[0][k]
                delta = np.append(delta, sigmoid(history[i][j], True) * s)
            deltas.insert(0, delta)
        del history[-1]
        history.insert(0, np.append(x, -1))

        for delta, output, layer in zip(deltas, history, self._layers):
            deltaWeights = np.matrix(delta).T.dot(np.matrix(output)) * rate * -1
            layer.setNeurons(layer.getNeurons() + deltaWeights)

        return (np.rint(prediction) == actual).all()

    def trainingSchedule(self, trainingSet, targetAccuracy = 0.996, rate = 1, printRate = 4096):
        '''set is list of tuples x, y'''
        print('begin training')
        l = len(trainingSet)
        totalCount = 0
        numC = 0
        count = 0
        accuracy = 0
        while accuracy < targetAccuracy or count < 2048 or totalCount < 4096:
            correct = self.train(trainingSet[random.randrange(0, l)], rate / 512)
            totalCount += 1
            count += 1
            if correct:
                numC += 1
            accuracy = numC/count
            if totalCount % printRate == 0:
                print('iteration: {0} accuracy: {1}'.format(str(totalCount), str(accuracy)))
                sys.stdout.flush()
                numC = 0
                count = 0
        print('iteration: {0} accuracy: {1}\n finished training\n\n'.format(str(totalCount), str(accuracy)))


    def __str__(self):
        x = '{0}\n'.format(str(self._exclusive))
        x += '{0}\n'.format(str(self._input._neurLength))
        for i in range(len(self._layers)-1):
            x += '{0} '.format(str(self._layers[i]._neurLength))
        x+='\n{0}\n'.format(str(self._layers[len(self._layers)-1]._neurLength))
        for layer in self._layers:
            x += '{0}\n'.format(str(layer))
        return x

'''
To do:
- convolutional net
- use theano
- genetic algorithm
'''
