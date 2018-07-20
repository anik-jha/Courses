import unittest
import numpy as np
from layers.softmax import SoftMaxLayer


class TestSoftMax(unittest.TestCase):
    def setUp(self):
        self.layer = SoftMaxLayer()

    def test_forward(self):
        """
        Test the forward function with some values
        """
        x = np.array(((0.5, 0.5, 0.5),
                      (0.3, 0.3, 0.3)))
        y = self.layer.forward(x)
        should_be = np.array(((1./3., 1./3., 1./3.),
                              (1./3., 1./3., 1./3.)))

        self.assertTrue(np.allclose(y, should_be))

    def test_backward(self):
        """
        Test the backward function using the numerical gradient
        """
        x = np.array(((0.5, 0.6, 0.4),
                      (0.3, 0.2, 0.1)))
        out1 = self.layer.forward(x)

        h = 0.0001
        x3 = x + np.array(([0, 0, -h],
                           [0, 0, -h]))
        out3 = self.layer.forward(x3)
        diff = (out1 - out3) / h
        #print diff

        y = np.ones((2, 3))
        y[:, 0] = 0
        y[:, 1] = 0
        y[:, 2] = 1
        x_grad = self.layer.backward(y)

        self.assertTrue(np.mean(diff - x_grad) < 1e-10)

if __name__ == '__main__':
    unittest.main()
