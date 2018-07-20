import numpy as np


class CrossEntropyLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.x = None
        self.t = None

    def forward(self, x, t):
        """
        Implements forward pass of cross entropy

        l(x,t) = 1/N * sum(log(x) * t)

        where
        x = input (number of samples x feature dimension)
        t = target with one hot encoding (number of samples x feature dimension)
        N = number of samples (constant)

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x feature dimension
        t : np.array
            The target data (one-hot) of size number of training samples x feature dimension

        Returns
        -------
        np.array
            The output of the loss

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        self.t : np.array
             The target data (need to store for backwards pass)
        """
        #raise NotImplementedError
        self.x = x
        self.t = t
        temp = np.log(x+0.000000000000001)* t
        N = x.shape[0]
        loss = -np.sum(temp) / N
        return loss

    def backward(self, y_grad=None):
        """
        Compute "backward" computation of softmax loss layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        #print "Shape of t = ", np.shape(self.t), " and x = ", np.shape(self.x)
        n_b = self.x.shape[0]
        #print n_b
        back = (self.t/(self.x+0.0000000000000001))/(-n_b)
        #print "back = ", back
        return back
