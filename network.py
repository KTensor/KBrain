import numpy as np
import random as random

def sigmoid(x, deriv = False):
    if deriv == True:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

def lninv(x):
    return 1/(np.log(x+2))

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
        return str(self.getNeurons())


class Input(Layer):
    def __init__(self, length):
        super().__init__(length)

    def out(self, aInput):
        return aInput


class Network:
    def __init__(self, inp, hid, out, layerWeights = None):
        self._input = Input(inp)
        self._output = Layer(out)
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
        print('self ' + str(self))
        x, y = example

        history = self.trainOut(x)
        print('history ' + str(history))

        deltas = []

        prediction = history[-1]
        actual = y
        print('prediction ' + str(prediction))
        print('actual ' + str(actual))
        error = (np.delete(prediction, -1) - actual)
        slope = [sigmoid(i, True) for i in prediction]
        print('error ' + str(error))
        print('slope ' + str(slope))
        deltaOut = np.multiply(error, slope)
        deltas.insert(0, deltaOut)
        print('deltaOut ' + str(deltaOut))

        # print(str(list(range(len(history)-2, -1, -1))))
        for i in range(len(history)-2, -1, -1):
            delta = np.array([])
            # print('i ' + str(i))
            for j in range(0, len(self._layers[i].getNeurons())):
                kNeuron = self._layers[i+1].getNeurons()
                # print('layer ' + str(kNeuron))
                s = 0
                for k in range(0, len(kNeuron)):
                    # print('kNeuron1' + str(kNeuron[k]))
                    # print('kNeuron ' + str(kNeuron[k][j]))
                    # print('delta ' + str(deltas[0][k]))
                    s += kNeuron[k][j]*deltas[0][k]
                delta = np.append(delta, sigmoid(history[i][j], True) * s)
            deltas.insert(0, delta)
        # print('deltasBefore ' + str(deltas))
        # deltas.reverse()
        print('deltas ' + str(deltas))
        del history[-1]
        history.insert(0, np.append(x, -1))

        for delta, output, layer in zip(deltas, history, self._layers):
            print('delta ' + str(delta))
            print('output ' + str(output))
            deltaWeights = np.matrix(delta).T.dot(np.matrix(output)) * rate * -1
            print('deltaWeights ' + str(deltaWeights))
            layer.setNeurons(layer.getNeurons() + deltaWeights)

        return (np.rint(prediction) == actual).all()

    def trainingSchedule(self, trainingSet, iterations, rate = 1, printRate = 512):
        '''set is list of tuples x, y'''
        l = len(trainingSet)
        numC = 0
        for i in range(0, iterations):
            correct = self.train(trainingSet[random.randrange(0, l)], rate * lninv(i))
            if(correct):
                numC += 1
            if (i+1) % printRate == 0:
                print('iteration: {0} accuracy: {1}'.format(str(i), str(numC/(i+1))))

    def __str__(self):
        x = 'row is neuron\nlast element of neuron is threshold\n\n'
        x += 'input\n {0}\n\n'.format(str(self._input._neurLength))
        for layer in self._layers:
            x += 'layer\n {0}\n\n'.format(str(layer))
        return x

'''
To do:
- training, backpropagation
- genetic algorithm
'''
