import numpy as np
class ReluLayer(object):
    def __init__(self):
        """
        Rectified Linear Unit
        """
        self.y = None


    def forward(self, x):
        """
        Implement forward pass of Relu

        y = x if x > 0
        y = 0 otherwise

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
             The output data (need to store for backwards pass)
        """
        #raise NotImplementedError

        op = np.maximum(0,x)
        self.y = op
        #print "op = ", np.array(op)
        return op



    def backward(self, y_grad):
        """
        Implement backward pass of Relu

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
        op1 = np.maximum(0,self.y)

        op1[op1>0] = 1
        x_grad = op1 * y_grad
        return x_grad


    def update_param(self, lr):
        pass  # no parameters to update
