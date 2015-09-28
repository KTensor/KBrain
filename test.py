import numpy as np
import network as N

np.random.seed(0)

net = N.Network(2, [2], 1)

# print(str(net))


trainingSet = [(np.array([0, 0]), np.array([0])), (np.array([0, 1]), np.array([1])), (np.array([1, 0]), np.array([1])), (np.array([1, 1]), np.array([0]))]

net.trainingSchedule(trainingSet, 131072)
