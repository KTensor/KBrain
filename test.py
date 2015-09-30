import numpy as np
import ffnet as F

np.random.seed(0)

net = F.Network(2, [2], 1)

print(str(net))


trainingSet = [(np.array([0, 0]), np.array([0])), (np.array([0, 1]), np.array([1])), (np.array([1, 0]), np.array([1])), (np.array([1, 1]), np.array([0]))]

net.trainingSchedule(trainingSet, 65536)

print(str(net))

print('input: {0}; output: {1}'.format('[0, 0]', str(net.out(np.array([0, 0])))))
print('input: {0}; output: {1}'.format('[0, 1]', str(net.out(np.array([0, 1])))))
print('input: {0}; output: {1}'.format('[1, 0]', str(net.out(np.array([1, 0])))))
print('input: {0}; output: {1}'.format('[1, 1]', str(net.out(np.array([1, 1])))))
