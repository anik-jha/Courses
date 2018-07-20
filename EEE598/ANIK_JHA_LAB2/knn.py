import numpy as np
from scipy import stats


class KNN(object):
    def __init__(self, k=3):
        self.x_train = None
        self.y_train = None
        self.k = k

    def fit(self, x, y):
        """
        Fit the model to the data

        For K-Nearest neighbors, the model is the data, so we just
        need to store the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """
        self.x_train = np.array(x)
        self.y_train = np.array(y)

        return self.x_train,self.y_train

    def predict(self, x):
        """
        Predict x from the k-nearest neighbors

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A vector of size N of the predicted class for each sample in x
        """
        indices = []
        for i in range(0, len(x)):
            temp = []
            for j in range(0, len(self.x_train)):
                dist = np.linalg.norm(x[i] - self.x_train[j])
                temp.append(dist)
            temp = np.argsort(temp)

            indices.append(temp)

        y_fin = []
        for i in range(0,len(x)):
            temp = []
            for j in range(0, self.k):
                a = self.y_train[indices[i][j]]
                temp.append(a)

            b = stats.mode(temp,axis = None)
            y_fin.append(int(b[0][0]))
        y_final = np.array(y_fin)
        #print y_final

        return y_final



