import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of softmax

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.array
            The output of the layer

        Stores
        -------
        self.y : np.array
             The output of the layer (needed for backpropagation)
        """
        #raise NotImplementedError

        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        self.y = probs

        return probs

    def backward(self, y_grad):
        """
        Compute "backward" computation of softmax

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        """
        #raise NotImplementedError
        #print (self.y)
        #print "y_grad = ", y_grad
        grad1 = np.zeros(np.shape(y_grad))
        for i in range(np.shape(self.y)[0]):
            temp = self.y[i].reshape((-1, 1))
            prod = np.dot(temp, temp.T)
            grad = np.diag((self.y)[i]) - prod
            grad1[i] = np.dot(y_grad[i],grad)
        #print grad1
        return grad1

    def update_param(self, lr):
        pass  # no learning for softmax layer
