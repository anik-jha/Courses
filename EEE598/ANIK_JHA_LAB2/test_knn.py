import unittest
from knn import KNN
import numpy as np


class TestKNN(unittest.TestCase):
    def test_fit(self):
        """
        Test KNN.fit is actually storing the training data
        """
        x_train = np.array([
            [1, 2],
            [1, 3],
            [2, 2],
            [2, 3],
            [1, 1],
            [2, 1]
            ])
        y_train = np.array([1, 1, 1, -1, -1, -1])

        model = KNN()
        model.fit(x_train, y_train)

        self.assertTrue(np.array_equal(x_train, model.x_train))
        self.assertTrue(np.array_equal(y_train, model.y_train))
        
    def test_synthetic_data(self):
        """
        Test KNN.predict using some synthetic data
        """
        x_train = np.array([
            [1, 2],
            [1, 3],
            [2, 2],
            [2, 3],
            [1, 1],
            [2, 1]
            ])
        y_train = np.array([1, 1, 1, -1, -1, -1])

        model = KNN(k=3)
        model.fit(x_train, y_train)

        x_test = np.array([
            [1.8, 2.6],
            [2.0, 1.8],
            [1.5, 2.0],
            [1.0, 2.5],
            [1.5, 1.0],
            [2.0, 1.0],
            ])

        pred = model.predict(x_test)

        self.assertTrue(np.array_equal(pred,
                                       np.array([1, 1, 1, 1, -1, -1])))

        # one labels should change if using 1-nn
        model.k = 1
        pred2 = model.predict(x_test)

        self.assertTrue(np.array_equal(pred2,
                                       np.array([-1, 1, 1, 1, -1, -1])))


if __name__ == '__main__':
    unittest.main()
