import mndatabase as m
import test as T
import numpy as np
import sys

#Kevin Wang
#github:xorkevin

dim = (784, [28, 10], 1)
trainingSet = m.mnist(0, './database/mnist')
testingSet = m.mnist(1, './database/mnist')

ocrtester = T.Tester(sys.argv[1], dim, 0.9375, trainingSet, testingSet)

ocrtester.start()
