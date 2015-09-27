import numpy as np
import network as N

np.random.seed(0)

net = N.Network(2, [2], 1)

print(str(net))
#
# print(str(net.out(np.array([1, 0]))))
