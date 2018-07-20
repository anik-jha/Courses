import numpy as np


class FullLayer(object):
    def __init__(self, n_i, n_o):
        """
        Fully connected layer

        Parameters
        ----------
        n_i : integer
            The number of inputs
        n_o : integer
            The number of outputs
        """
        self.x = None
        self.W_grad = None
        self.b_grad = None

        #print ("n_i =", n_i, " n_o = ", n_o)
        sigma = np.sqrt(2.0/(n_i+n_o))
        self.W= np.random.normal(0,sigma, (n_o,n_i))
        self.b = 0#np.zeros((1,n_o))#np.random.randn(1, n_o)

        # need to initialize self.W and self.b
        #raise NotImplementedError

    def forward(self, x):
        """
        Compute "forward" computation of fully connected layer

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
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        #raise NotImplementedError
        #print "self_b = ", np.shape(self.b)
        in_prod = np.dot(x, np.transpose(self.W)) + self.b
        self.x = x
        # in_prod = (np.transpose(self.w)*x) + self.b
        #print in_prod
        #print self.W
        return in_prod

    def backward(self, y_grad):
        """
        Compute "backward" computation of fully connected layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.W_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """
        #raise NotImplementedError
        '''
        N = self.x.shape[0]
        D = np.prod(self.x.shape[1:])
        x2 = np.reshape(self.x, (N, D))

        dx2 = np.dot(y_grad, self.W_grad.T)  # N x D
        dw = np.dot(x2.T, y_grad)  # D x M
        db = np.dot(y_grad.T, np.ones(N))  # M x 1
        dx = np.reshape(dx2, self.x.shape)
        return dx
        '''


        self.W_grad = np.dot(y_grad.T, self.x)
        self.b_grad = np.dot(np.ones((1, y_grad.shape[0])), y_grad)
        output = np.dot(y_grad, self.W)
        return output

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate

        Stores
        -------
        self.W : np.array
             The updated value for self.W
        self.b : np.array
             The updated value for self.b
        """
        #raise NotImplementedError
        self.W = self.W - lr * self.W_grad
        self.b = self.b - lr * self.b_grad
