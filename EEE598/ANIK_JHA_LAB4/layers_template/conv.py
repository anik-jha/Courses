import numpy as np
import scipy.signal


class ConvLayer(object):
    def __init__(self, n_i, n_o, h):
        """
        Convolutional layer

        Parameters
        ----------
        n_i : integer
            The number of input channels
        n_o : integer
            The number of output channels
        h : integer
            The size of the filter
        """
        # glorot initialization
        self.n_i = n_i
        self.n_o = n_o

        sigma = np.sqrt(2.0 / (n_i + n_o))
        self.W = np.random.normal(0, sigma, (n_o, n_i, h, h))
        self.b = np.zeros((1,n_o))

        self.W_grad = None
        self.b_grad = None

    def forward(self, x):
        """
        Compute "forward" computation of convolutional layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the convolutiona

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        """

        conv_out = np.zeros((x.shape[0],self.W.shape[0],np.shape(x)[2],np.shape(x)[3]))

        x_zp = np.zeros((x.shape[0], x.shape[1], x.shape[2] + (self.W.shape[2] - 1), x.shape[3] + (self.W.shape[3] - 1)))
        # print "Shape Zero:",np.shape(x_zp)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_zp[i][j] = (np.pad(x[i][j], (self.W.shape[2] - 1) / 2, mode='constant'))

        self.x = x_zp
        t_size = np.shape(x[0][0])
        #print "t_size = ",t_size

        for i in range(x.shape[0]):
            for j in range(self.W.shape[0]):
                temp = np.zeros(t_size)
                for k in range(x.shape[1]):

                    temp += (scipy.signal.correlate(x_zp[i][k],self.W[j][k],'valid'))
                conv_out[i][j] = temp+ self.b[0][j]
        return conv_out

    def backward(self, y_grad):
        """
        Compute "backward" computation of convolutional layer

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
        self.w_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """
        #raise NotImplementedError

        temp_B = np.zeros((1,y_grad.shape[1]))
        temp_b = np.zeros((1, y_grad.shape[1]))
        # WRT B
        #self.b_grad = np.sum(np.sum(np.sum(y_grad,axis=3),axis=2),axis=0)
        for i in range(y_grad.shape[1]):
            for j in range(y_grad.shape[0]):
                self.b_grad = np.sum((np.sum(np.sum(y_grad, axis=3), axis=2)),axis = 0)
        #print "Frm back:",self.b_grad
        #self.b_grad  = temp_B

        conv_out = np.zeros((y_grad.shape[0], self.W.shape[1], y_grad.shape[2], y_grad.shape[3]))
        # print "Conv out:",np.shape(conv_out)
        for i in range(y_grad.shape[0]):
            for a in range(self.W.shape[1]):
                for j in range(self.W.shape[0]):
                    # print "I ",i
                    # print "J ",j
                    # print "We are get:", np.shape(scipy.signal.correlate(y_grad[i][j], self.W[i][j], 'same'))
                    conv_out[i][a] += scipy.signal.convolve(y_grad[i][j], self.W[j][a], 'same')
                # print "We are get:",np.shape(scipy.signal.correlate(y_grad[i][j],self.W[i][j],'valid'))
        # print "Conv Output",conv_out
        # print "np shape:",np.shape(conv_out)
        x_grad = conv_out
        # print "X_Grad: ",np.shape(x_grad)

        # WRT W:
        # print "YGRAD:", np.shape(y_grad)
        # print "Self.x", np.shape(self.x)
        # print "SElf.w ",np.shape(self.W)
        conv_out_2 = np.zeros(np.shape(self.W))
        conv_out_3 = np.zeros(np.shape(self.W))
        for k in range(self.x.shape[0]):
            for i in range(y_grad.shape[1]):
                for j in range(self.x.shape[1]):
                    # print "X size:",np.shape(self.x[k][j])
                    # print "Y_Grad size:",np.shape(y_grad[k][i])
                    conv_out_2[i][j] = scipy.signal.correlate(self.x[k][j], y_grad[k][i], 'valid')
            conv_out_3 += conv_out_2
        self.W_grad = conv_out_3
        # print "W_GRad",np.shape(self.W_grad)
        # print "W",np.shape(self.W)

        return x_grad

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
        self.W = self.W-lr*self.W_grad
        self.b = self.b - lr * self.b_grad
