#### Libraries
# Standard library
import pickle
import numpy as np
import theano
import theano.tensor as T

def load_data(num=5):
    """Return the CIFAR-10 data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 3*32*32 values, representing the 32 * 32 = 1024
    pixels of 3 colors in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the 
    values (0...9) for the corresponding class of the image contained in the first
    entry of the tuple.
    The ``test_data`` is similar, except each contains only 10,000 images.    
    """
    train_x = []
    train_y = []    
    for i in range(1, 1+num):
        fn = 'cifar-10-batches-py/data_batch_'+str(i)
        f = open(fn, 'rb')        
        dic = pickle.load(f, encoding="bytes")        
        train_x = train_x + list(dic[b'data']/255.0)
        train_y = train_y + dic[b'labels']
        f.close()

    fn = 'cifar-10-batches-py/test_batch'
    f = open(fn, 'rb')        
    dic = pickle.load(f, encoding="bytes")
    test_x = dic[b'data']
    test_y = dic[b'labels']
    print(len(train_x), 'training data', len(test_y), 'test data')
    return train_x, train_y, test_x, test_y

def shared_data(x, y):
    """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available. """
    shared_x = theano.shared(
    np.asarray(x, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
    np.asarray(y, dtype=theano.config.floatX), borrow=True)
    return shared_x, T.cast(shared_y, "int32")

# mini-batch size:
batch_size = 10
num_vali = 10*batch_size  # number of examples in validation set
tr_x, tr_y, te_x, te_y = load_data(5)
train_data = shared_data(tr_x[:-num_vali], tr_y[:-num_vali])
vali_data = shared_data(tr_x[-num_vali:], tr_y[-num_vali:])
test_data = shared_data(te_x, te_y)

import time
start_time = time.time()

import network3A_Momentum
from network3A_Momentum import Network, FullyConnectedLayer, SoftmaxLayer 
# softmax plus log-likelihood cost is more common in modern image classification networks.

net = Network([
    FullyConnectedLayer(n_in=3*32*32, n_out=400),
    FullyConnectedLayer(n_in=400, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], batch_size)
    # Last layer must be SoftmaxLayer because the cost function
    # and y_out are defined only in this layer.

net.SGD(train_data, 30, batch_size, 0.05, vali_data, test_data, lmbda=0.05)

print("run time: %s seconds" % (time.time() - start_time))
    
""" Draw one image """
from matplotlib import pyplot as plt
pics = (vali_data[0]).get_value()
x = pics[np.random.randint(len(pics))]
x1 = x.reshape(3, 32, 32)
x2 = np.moveaxis(x1, 0, 2)
print('shapes of x, x1, and x2:', x.shape, x1.shape, x2.shape)
plt.imshow(x2)
plt.show()