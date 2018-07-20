import numpy as np


class SVM(object):
    def __init__(self, n_epochs=10, lr=0.1, l2_reg=1):
        """
        """
        self.b = None
        self.w = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg

    def forward(self, x):
        """
        Compute "forward" computation of SVM f(x) = w^T + b

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
        #in_prod = np.dot(np.transpose(self.w),x) + self.b

        in_prod = np.matmul(x, np.transpose(self.w)) + self.b
        return in_prod[:,0]

    def loss(self, x, y):
        """
        Return the SVM hinge loss
        L(x) = max(0, 1-y(f(x))) + 0.5 w^T w

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
        #print "y = ", y, " x = ", x, "b = ", self.b

        #print "forward_x", self.forward(x)
        loss1 = 0
        for i in range(len(y)):
            loss1 += np.maximum(0,(1-y[i]*np.matmul(x[i], np.transpose(self.w)) + self.b))
        loss2 = 0.5*self.l2_reg*np.matmul(self.w, np.transpose(self.w))
        loss = loss1/len(y) + loss2
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

        loss1 = 0
        for i in range(len(y)):
            temp_loss = (1 - y[i] * np.matmul(x[i], np.transpose(self.w)) + self.b)
            if temp_loss < 0:
                loss1 = 0
            else:
                loss1 += -y[i]
                print "loss1 = ", loss1
        #loss2 = np.sum(0.5 * self.l2_reg * np.transpose(self.w))
        loss = float(loss1) / len(y)
        print "loss = ", loss
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
        loss1 = 0
        for i in range(len(y)):
            temp_loss = 1 - y[i] * np.matmul(x[i], np.transpose(self.w)) + self.b
            #temp_loss = 1 - np.matmul(y[i],np.matmul(x[i], np.transpose(self.w)) + self.b))
            if temp_loss < 0:
                loss1 += 0
            else:
                loss1 += -y[i] * (x[i])
        print "loss1 = ", loss1
        loss2 = self.l2_reg * (self.w)
        print "loss2 = ", loss2
        loss = loss1 / len(x) + loss2
        loss_final = np.array([[[loss[:,0]], [[loss[:,1]]]]])
        print "loss = ", loss_final
        return loss_final

    def fit(self, x, y, plot=False):
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
        self.w = np.random.rand(2)
        self.b = 0.2
        #np.random.uniform(0, 1)
        temp_b = []

        #print self.grad_loss_wrt_b(x, y)

        a = []
        self.b -= self.lr * self.grad_loss_wrt_b(x, y)
        #print "b = ", self.b,
        self.w -= self.lr * self.grad_loss_wrt_w(x, y)

        return self.w,self.b



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
        print "f = ", f
        y = []
        for i in range(len(f)):
            if f[i] <= 0:
                temp = -1
            else:
                temp = 1
            print "temp = ", temp
            y.append(temp)
        print "y = ", y
        return y
