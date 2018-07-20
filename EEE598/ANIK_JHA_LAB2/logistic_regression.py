import numpy as np


class LogisticRegression(object):
    def __init__(self, n_epochs=10, lr=0.1, l2_reg=0):
        """
        Initialize variables
        """
        self.b = None
        self.w = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg

    def forward(self, x):
        """
        Compute "forward" computation of logistic regression

        This will return the squashing function:
        f(x) = 1 / (1 + exp(-(w^T x + b)))

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A 1 dimensional vector of the logistic function
        """
        #in_prod = np.transpose(self.w).dot(x)+self.b
        N= self.w.shape[0]
        D = self.w.shape[1]
        print "N =", N, " D = ", D
        print "x =", x
        print "Shape of x = ", x.shape[0], " ", x.shape[1:]
        in_prod = np.dot(x,np.transpose(self.w)) + self.b
        #in_prod = (np.transpose(self.w)*x) + self.b
        f = (1/(1+np.exp(-in_prod)))
        #print "f = ", f[:, 0]

        return f[:, 0]

    def loss(self, x, y):
        """
        Return the logistic loss
        L(x) = ln(1 + exp(-y * (w^Tx + b)))

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The logistic loss value
        """
        in_prod = np.dot(x, np.transpose(self.w)) + self.b
        loss = np.log (1 + np.exp(np.dot(-y, in_prod)))
        #loss = np.mean(np.log(1 + np.exp((-y)* in_prod)))
        print "Loss = ", loss
        return loss

    def grad_loss_wrt_b(self, x, y):
        """
        Compute the gradient of the loss with respect to b

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The gradient
        """
        in_prod = np.dot(x, np.transpose(self.w)) + self.b
        loss = np.sum((-y)/(1 + np.exp(np.dot(y, in_prod))))/len(x)
        return loss

    def grad_loss_wrt_w(self, x, y):
        """
        Compute the gradient of the loss with respect to w

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        np.array
            The gradient (should be the same size as self.w)
        """

        in_prod = np.dot(x, np.transpose(self.w)) + self.b

        loss = 1 / (1 + np.exp(np.dot(y, in_prod)))

        grad = np.dot((-y), x)*loss

        #print "Grad  =", grad
        return grad

    def fit(self, x, y):
        """
        Fit the model to the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """

        for i in range(self.n_epochs):
            self.b -= self.lr * self.grad_loss_wrt_b(self, x, y)
            self.w -= self.lr * self.grad_loss_wrt_w(self, x, y)

        return self.b, self.w

    def predict(self, x):
        """
        Predict the labels for test data x

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            Vector of predicted class labels for every training sample
        """

        f = self.forward(x)

        y =[]
        for i in range(len(f)):
            if f[i] <= 0.5:
                temp = -1
            else:
                temp = 1
            y.append(temp)
        return y
        #self.b, self.w = self.fit(self, x)
