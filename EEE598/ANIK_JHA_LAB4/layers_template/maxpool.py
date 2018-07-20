import numpy as np


class MaxPoolLayer(object):
    def __init__(self, size=2):
        """
        MaxPool layer
        Ok to assume non-overlapping regions
        """
        self.locs = None  # to store max locations
        self.size = size  # size of the pooling

    def forward(self, x):
        """
        Compute "forward" computation of max pooling layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the maxpooling

        Stores
        -------
        self.locs : np.array
             The locations of the maxes (needed for back propagation)
        """
        size = self.size
        ind = np.zeros(np.shape(x))
        max_op = np.zeros((x.shape[0],x.shape[1],x.shape[2]/self.size, x.shape[3]/self.size))
        #print np.shape(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for r in range(x.shape[2]/self.size):
                    for c in range(x.shape[3]/self.size):
                        temp1 = np.max(x[i][j][r*size:(r*size+size), c*size:(c*size+size)])
                        max_op[i][j][r][c] =temp1
                        temp2 = ((x[i][j][r*self.size:r*self.size + self.size,c*self.size:c*self.size + self.size]))
                        for row in range(self.size):
                            for column in range(self.size):
                                if (temp2[row][column] == temp1):
                                    ind[i][j][r * self.size + row][c * self.size + column] = 1




        #print "ind = ", ind
        self.locs = ind
        return max_op




    def backward(self, y_grad):
        """
        Compute "backward" computation of maxpool layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        size = self.size
        ind = self.locs
        updated_ip = np.zeros(np.shape(ind))
        #print "y_grad = ", y_grad[0][2]
        for i in range(y_grad.shape[0]):
            for j in range(y_grad.shape[1]):
                for r in range(y_grad.shape[2]):
                    for c in range(y_grad.shape[3]):
                        updated_ip[i][j][r*size:(r*size+size), c*size:(c*size+size)] = ind[i][j][r*size:(r*size+size), c*size:(c*size+size)]*y_grad[i][j][r][c]


        return updated_ip
        #raise NotImplementedError

    def update_param(self, lr):
        pass
