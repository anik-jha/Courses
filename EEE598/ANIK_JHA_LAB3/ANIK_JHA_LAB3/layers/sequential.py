from __future__ import print_function
import numpy as np


class Sequential(object):
    def __init__(self, layers, loss):
        """
        Sequential model

        Implements a sequence of layers

        Parameters
        ----------
        layers : list of layer objects
        loss : loss object
        """
        self.layers = layers
        self.loss = loss

    def forward(self, x, target=None):
        """
        Forward pass through all layers
        
        if target is not none, then also do loss layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features
        target : np.array
            The target data of size number of training samples x number of features (one-hot)

        Returns
        -------
        np.array
            The output of the model
        """
        #raise NotImplementedError
        #print self.layers
        self.x = x
        y1 = self.layers[0].forward(x)
        relu1 = self.layers[1].forward(y1)
        y2 = self.layers[2].forward(relu1)
        soft = self.layers[3].forward(y2)

        #print loss
        if target is None:
            return soft
        else:
            loss = self.loss.forward(soft, target)
            return loss

    def backward(self):
        """
        Compute "backward" computation of fully connected layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        #raise NotImplementedError
        loss_b = self.loss.backward()
        soft_b = self.layers[3].backward(loss_b)
        y2_b = self.layers[2].backward(soft_b)
        relu1_b = self.layers[1].backward(y2_b)
        y1_b = self.layers[0].backward(relu1_b)
        #print self.forward(self.x)
        return y1_b


    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate
        """
        #raise NotImplementedError
        self.layers[0].update_param(lr)
        self.layers[2].update_param(lr)

    def fit(self, x, y, epochs=10, lr=0.1, batch_size=128):
        """
        Fit parameters of all layers using batches

        Parameters
        ----------
        x : numpy matrix
            Training data (number of samples x number of features)
        y : numpy matrix
            Training labels (number of samples x number of features) (one-hot)
        epochs: integer
            Number of epochs to run (1 epoch = 1 pass through entire data)
        lr: float
            Learning rate
        batch_size: integer
            Number of data samples per batch of gradient descent
        """
        #raise NotImplementedError

        epoch_loss = np.zeros(epochs,)
        batches = np.shape(x)[0]/batch_size
        for i in range(epochs):
            cnt=0
            batch_loss = 0
            for j in range(batches):
                temp_x = x[cnt:(j+1)*batch_size]
                temp_y = y[cnt:(j+1)*batch_size]
                #for k in range(batch_size):
                batch_loss += self.forward(temp_x, temp_y)
                grad_temp = self.backward()
                self.update_param(lr)
                cnt+= batch_size
            batch_loss/= (batch_size * batches)
            epoch_loss[i] = batch_loss

        return epoch_loss



    def predict(self, x):
        """
        Return class prediction with input x

        Parameters
        ----------
        x : numpy matrix
            Testing data data (number of samples x number of features)

        Returns
        -------
        np.array
            The output of the model (integer class predictions)
        """
        #raise NotImplementedError
        temp = self.forward(x)
        #print ("temp =", np.shape(temp))
        #return temp.index(max(temp))
        return np.argmax(temp,axis=1)

