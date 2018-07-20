import unittest
import numpy as np
from layers.relu import ReluLayer


class TestRelu(unittest.TestCase):
    def setUp(self):
        self.layer = ReluLayer()

    def test_forward(self):
        """
        Test the forward function with some values
        """
        x = np.array([[-1, 1, 0, 3]]).T
        y = self.layer.forward(x)
        should_be = np.array([[0, 1, 0, 3]]).T

        self.assertTrue(np.allclose(y, should_be))

    def test_backward(self):
        """
        Test the backward function with some values
        """
        x = np.array([[-1, 1, 0, 3]]).T
        y = self.layer.forward(x)

        z = np.ones((4, 1))
        x_grad = self.layer.backward(z)

        should_be = np.array([[0, 1, 0, 1]]).T

        self.assertTrue(np.allclose(x_grad, should_be))

    def test_backward2(self):
        """
        Test the backward function using the numerical gradient
        """
        x = np.array([[-1, 1, 0, 3]]).T
        out1 = self.layer.forward(x)

        z = np.ones((4, 1))
        x_grad = self.layer.backward(z)

        h = 0.0001
        x2 = x + np.array([[h, 0, 0, 0]]).T
        out2 = self.layer.forward(x2)
        diff = (out2 - out1) / h
        self.assertTrue(np.allclose(np.sum(diff), x_grad[0]))

        h = 0.0001
        x2 = x + np.array([[0, h, 0, 0]]).T
        out2 = self.layer.forward(x2)
        diff = (out2 - out1) / h
        self.assertTrue(np.allclose(np.sum(diff), x_grad[1]))

        h = 0.0001
        x2 = x + np.array([[0, 0, -h, 0]]).T
        out2 = self.layer.forward(x2)
        diff = (out1 - out2) / h
        self.assertTrue(np.allclose(np.sum(diff), x_grad[2]))

        h = 0.0001
        x2 = x + np.array([[0, 0, 0, h]]).T
        out2 = self.layer.forward(x2)
        diff = (out2 - out1) / h
        self.assertTrue(np.allclose(np.sum(diff), x_grad[3]))


if __name__ == '__main__':
    unittest.main()
