import numpy as np


class LeastSquares(object):

    def __init__(self, k):
        """
        Initialize the LeastSquares class

        The input parameter k specifies the degree of the polynomial
        """
        self.k = k
        self.coeff = None


    def fit(self, x, y):
        """
        Find coefficients of polynomial that predicts y given x with
        degree self.k

        Store the coefficients in self.coeff

        """
        k= self.k
        N = len(x)
        A = []

        #print "Value of k = ", k


        for i in range(0,N):
            temp = []
            for j in range(0,k+1):
                t = np.power(x[i],j)
                temp.append(t)
            A.append(temp)


        pinv_a = np.linalg.pinv(A)
        self.coeff = np.dot(pinv_a, y)
        return self.coeff


    def predict(self, x):
        """
        Predict the output given x using the learned coeffecients in
        self.coeff
        """

        #print self.coeff
        k = self.k
        N = len(x)
        A = []

        for i in range(0, N):
            temp = []
            for j in range(0, k + 1):
                t = np.power(x[i], j)
                temp.append(t)
            A.append(temp)
        y = np.dot(A, self.coeff)
        return y




