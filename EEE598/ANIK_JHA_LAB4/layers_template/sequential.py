from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


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
        ##print(self.layers)

        # Full1
        # print("shape of self.layers:",np.size(self.layers))
        input = x

        for i in range(np.size(self.layers)):
            print("I", i)
            fwd = (self.layers[i].forward(input))

            input = fwd

        # print("fwd1 Shape:", np.shape(fwd))

        # # Relu1
        # relu = self.layers[1].forward(fwd)
        #
        #
        # #Full 2
        # fwd2 = self.layers[2].forward(relu)
        # #print("full2 Shape:", np.shape(fwd2))
        #
        # #Relu 2
        # fwd3 = self.layers[3].forward(fwd2)
        #
        # #Full3
        # fwd4 = self.layers[4].forward(fwd3)
        # #print("Full3 Shape:", np.shape(fwd4))
        #
        # # Soft
        # fwd5 = self.layers[5].forward(fwd4)
        # #print("Soft Shape:", np.shape(fwd5))

        if target is None:
            # print("Loose 2:",fwd[np.size(fwd)-1])
            return fwd
        else:
            loss = self.loss.forward(fwd, target)
            return loss
        # raise NotImplementedError

    def backward(self):
        """
        Compute "backward" computation of fully connected layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        prev = 0
        loss = self.loss.backward()
        prevInput = loss
        # print("Works?")
        for j in range(np.size(self.layers)):
            # print("J",np.size(self.layers) - j-1)
            prev = self.layers[np.size(self.layers) - j - 1].backward(prevInput)
            prevInput = prev
            # print(prevInput)
        return prev

        #
        # prev0 = self.layers[5].backward(loss)
        # ##print("Prev 1 ",np.shape(prev1))
        #
        # #sOFT
        # prev1 = self.layers[4].backward(prev0)
        #
        #
        # prev2 = self.layers[3].backward(prev1)
        #
        # prev3 = self.layers[2].backward(prev2)
        #
        # prev4 = self.layers[1].backward(prev3)
        #
        # prev5 = self.layers[0].backward(prev4)
        #

        # raise NotImplementedError

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate
        """

        for k in range(np.size(self.layers)):
            self.layers[k].update_param(lr)
        # self.layers[0].update_param(lr)
        # self.layers[2].update_param(lr)
        # self.layers[4].update_param(lr)

        # raise NotImplementedError

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
        epoch_loss = np.zeros(epochs, )
        numBatches = np.shape(x)[0] / batch_size
        print("Shape:", np.shape(x)[0] / batch_size)
        for i in range(epochs):

            count = 0
            loss = 0
            # grad = 0
            for j in range(numBatches):
                print("Epoch :", i)
                print("Batch:", j)
                xBatch = x[count:count + batch_size]
                yBatch = y[count:count + batch_size]
                loss = loss + self.forward(xBatch, yBatch)
                grad = self.backward()
                self.update_param(lr)
                count = count + batch_size
            epoch_loss[i] = loss / (batch_size)
            print("Epoch loss", epoch_loss)

        # plt.plot(epoch_loss)

        # raise NotImplementedError

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
        softOp = self.forward(x)

        return np.argmax(softOp, axis=1)