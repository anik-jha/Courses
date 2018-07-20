from layers.full import FullLayer
import unittest
import numpy as np


class TestFull(unittest.TestCase):
    def setUp(self):
        """
        Setup layer with some values
        """
        layer = FullLayer(2, 3)
        layer.W = np.array(((1, 2, 3), (4, 5, 6)), 'float64').T
        layer.b = np.array(((2, 4, 6),), 'float64')
        self.layer = layer

    def test_forward(self):
        """
        Test the forward function with some values
        """
        x = np.array([[1, 2],
                      [-1, 1]])
        out = self.layer.forward(x)

        should_be = np.array([(11, 16, 21),
                              (5, 7, 9)])

        self.assertTrue(np.allclose(out, should_be))

    def test_update_param(self):
        """
        Test the update_param function with some values
        """
        x = np.array(((1, 2),))
        out = self.layer.forward(x)

        y = np.ones((1, 3))
        self.layer.backward(y)

        # updating grad should decrease value
        old_val = np.sum(out)
        valid = True
        for i in range(10):
            self.layer.update_param(0.1)
            new_val = np.sum(self.layer.forward(x))
            self.layer.backward(y)

            if new_val >= old_val:
                valid = False
            old_val = new_val

        self.assertTrue(valid)

    def test_backward(self):
        """
        test gradients numerically
        """
        x = np.array(((1, 2),))
        out1 = self.layer.forward(x)

        y = np.ones((1, 3))
        x_grad = self.layer.backward(y)

        # test x gradient
        h = 0.0001
        x2 = x + np.array([[h, 0]])
        out2 = self.layer.forward(x2)
        diff = (out2 - out1) / h
        self.assertAlmostEqual(np.sum(diff), x_grad[0][0])

        x3 = x + np.array([[0, h]])
        out3 = self.layer.forward(x3)
        diff = (out3 - out1) / h
        self.assertAlmostEqual(np.sum(diff), x_grad[0][1])

        # test w gradient
        for i in range(self.layer.W.shape[0]):
            for j in range(self.layer.W.shape[1]):
                w_h = np.zeros_like(self.layer.W)
                w_h[i, j] = h
                w_old = np.copy(self.layer.W)
                w_new = self.layer.W + w_h
                self.layer.W = w_new

                out_new = self.layer.forward(x)
                diff = (out_new - out1) / h
                self.assertAlmostEqual(np.sum(diff), self.layer.W_grad[i,j])

                self.layer.W = w_old

        # test b gradient
        for i in range(self.layer.b.shape[1]):
            b_h = np.zeros_like(self.layer.b)
            b_h[0, i] = h
            b_old = np.copy(self.layer.b)
            b_new = self.layer.b + b_h
            self.layer.b = b_new

            out_new = self.layer.forward(x)
            diff = (out_new - out1) / h
            self.assertAlmostEqual(np.sum(diff), self.layer.b_grad[0][i])

            self.layer.b = b_old


if __name__ == '__main__':
    unittest.main()
