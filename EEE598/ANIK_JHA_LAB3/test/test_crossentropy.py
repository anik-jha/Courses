import unittest
import numpy as np
from layers.cross_entropy import CrossEntropyLayer


class TestCrossEntropy(unittest.TestCase):
    def setUp(self):
        """
        Set up the layer with some values
        """
        self.layer = CrossEntropyLayer()
        self.x = np.array([[0.6, 0.4],
                           [0.5, 0.5]])
        self.target = np.array([[0, 1],
                                [1, 0]])
        self.int_target = np.array((1, 0))

    def test_forward(self):
        """
        Test the forward function with the values from setUp
        """
        y = self.layer.forward(self.x, self.target)

        should_be = 0.8047189
        self.assertTrue(abs(y - should_be) < 0.001)

    def test_backward(self):
        """
        Test the backward function using the numerical gradient
        """
        out1 = self.layer.forward(self.x, self.target)

        x_grad = self.layer.backward(None)

        h = 0.0001

        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                add = np.zeros_like(self.x)
                add[i, j] = h
                x2 = self.x + add

                out2 = self.layer.forward(x2, self.target)
                diff = (out2 - out1) / h
                #print diff
                self.assertTrue(abs(diff - x_grad[i, j]) < 0.001)


if __name__ == '__main__':
    unittest.main()
