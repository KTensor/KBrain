import test as T
import numpy as np
import sys

#Kevin Wang
#github:xorkevin

dim = (2, [4], 1)
trainingSet = [(np.array([0, 0]), np.array([0])), (np.array([0, 1]), np.array([1])), (np.array([1, 0]), np.array([1])), (np.array([1, 1]), np.array([0]))]

xortester = T.Tester(sys.argv[1], dim, False, 0.93, trainingSet, trainingSet)

xortester.start()
