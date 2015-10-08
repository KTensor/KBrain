import numpy as np
import ffnet as F

#Kevin Wang
#github:xorkevin

class Tester:
    def __init__(self, filename, dim, exclusive, targetAccuracy, trainingSet, testingSet):
        self._filename = filename
        self._mode = 2
        self._dimensions = dim
        self._exclusive = exclusive
        self._training = trainingSet
        self._testing = testingSet
        self._targetAccuracy = targetAccuracy

    def start(self):
        while self._mode != 0:
            self._mode = int(input('1 | train \n2 | test \n3 | test (printing) \n7 | change target accuracy \n8 | restart training *DANGER* \n9 | change filename \n0 | quit \n$: '))
            if self._mode == 1:
                self.train()
            elif self._mode == 2:
                self.test()
            elif self._mode == 3:
                self.test(True)
            elif self._mode == 7:
                self._targetAccuracy = float(int(input('enter new accuracy')))
            elif self._mode == 8:
                self.train(True)
            elif self._mode == 9:
                self._filename = input('enter new filename: ')

    def train(self, restart = False):
        print('TRAINING\n')

        i, hid, o = self._dimensions
        network = F.Network()
        if restart:
            network = F.Network(i, hid, o, None, self._exclusive)
        else:
            network.load(self._filename)
        network.trainingSchedule(self._training, self._targetAccuracy)
        network.save(self._filename)

        print('file saved to {0}\n\n'.format(self._filename))

    def test(self, printing = False):
        print('TESTING\n')

        net = F.Network()
        net.load(self._filename)

        print(net)

        total = 0
        numcorrect = 0

        for test, actual in self._testing:
            prediction = net.out(test)
            correct = (np.rint(prediction) == actual).all()
            if correct:
                numcorrect+=1
            total += 1
            if printing:
                print('input: {0} | output: {1} | correct: {2}'.format(str(test), str(net.out(test)), str(correct)))

        accuracy = numcorrect/total
        print('total tests: {0} | num correct: {1} | accuracy: {2}\n\n'.format(str(total), str(numcorrect), str(accuracy)))
