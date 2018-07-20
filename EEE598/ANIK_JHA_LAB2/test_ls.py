import unittest
from ls import LeastSquares
import numpy as np


class TestLeastSquares(unittest.TestCase):
    def test_fit(self):
        """
        Test LeastSquares.fit
        """
        ls = LeastSquares(3)

        # if I generate data, then the least squares should be able to
        # find the coefficients I generated the data with
        x = np.linspace(0, 1, 100)
        y = 0.1 + -0.3 * x + 1.2 * x**2 - 0.5 * x**3

        ls.fit(x, y)

        should_be = np.array([0.1, -0.3, 1.2, -0.5])

        self.assertTrue(np.allclose(ls.coeff, should_be))

    def test_predict(self):
        """
        Test LeastSquares.predict using some synthetic data
        """
        ls = LeastSquares(3)

        # y = 0.3 + 1.5x -0.1x^2 + 0.5x^3
        ls.coeff = [0.3, 1.5, -0.1, 0.5]

        x = np.array([3, -5, 6, 8])

        y = ls.predict(x)

        should_be = np.array([17.4, -72.2, 113.7, 261.9])

        self.assertTrue(np.array_equal(y, should_be))


if __name__ == '__main__':
    unittest.main()
