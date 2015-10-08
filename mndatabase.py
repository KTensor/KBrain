import os, struct, array
import numpy as np

#Kevin Wang
#github:xorkevin

key = [
    np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    ]

def mnist(dataset=0, path='.', digits=np.arange(10)):
    fname_img = None
    fname_label = None
    if dataset == 0:
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_label = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == 1:
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_label = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError('dataset must be testing:1 or training:0')

    flabel = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flabel.read(8))
    label = array.array("b", flabel.read())
    flabel.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array.array("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if label[k] in digits ]
    N = len(ind)

    images = np.zeros((N, rows*cols), dtype=np.uint8)
    labels = np.zeros((N, 10), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.ravel(np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols)))
        labels[i] = key[label[ind[i]]]

    return list(zip(images, labels))
