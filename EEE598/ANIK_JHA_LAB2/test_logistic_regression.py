from logistic_regression import LogisticRegression
import unittest
import numpy as np


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.model = LogisticRegression()

    def test_forward(self):
        """
        Test LogisticRegression.forward function using some hand chosen values
        """
        # test single input
        self.model.w = np.array([[0.5, 0.25]])
        self.model.b = 0.5
        x = np.array([[0.2, 0.1]])
        out = self.model.forward(x)
        self.assertTrue(np.abs(out[0] - 0.6514) < 0.01)

        # test multiple inputs
        self.model.w = np.array([[0.1, 0.2]])
        self.model.b = 0.2
        x = np.array([[0.3, 0.4],
                      [0.5, 0.6]])
        out = self.model.forward(x)
        should_be = np.array([0.5769,0.5915])
        self.assertTrue(np.allclose(out, should_be, atol=0.01))

    def test_loss(self):
        """
        Test LogisticRegression.loss
        """
        """
        self.model.w = np.array([[0.1, 0.2]])
        self.model.b = 0.2
        x = np.array([[0.3, 0.4],
                      [0.5, 0.6]])
        y = np.array([1, -1])
        out = self.model.loss(x, y)
        should_be = 0.7226
        """
        self.model.w = np.array([[-0.3, 0.7]])
        self.model.b = 0.1
        x = np.array([[0.1, 0.3],
                      [0.3, -0.1]])
        y = np.array([1, -1])
        out = self.model.loss(x, y)
        should_be = 0.6133
        self.assertTrue(np.abs(out - should_be) < 0.01)

    def test_loss_with_reg(self):
        """
        Test LogisticRegression.loss with regularization
        """
        self.model.w = np.array([[0.1, 0.2]])
        self.model.b = 0.2
        self.model.l2_reg = 0.1
        x = np.array([[0.3, 0.4],
                      [0.5, 0.6]])
        y = np.array([1, -1])
        out = self.model.loss(x, y)
        should_be = 0.7226 + 0.5 * 0.1 * np.sum(self.model.w ** 2)

        # test numerically
        self.assertTrue(np.abs(out - should_be) < 0.01)

    def test_grad_loss_wrt_b(self):
        """
        Test LogisticRegression.grad_loss_wrt_b
        """
        # test numerically
        # first compute output
        self.model.w = np.array([[0.1, 0.2]])
        self.model.b = 0.2
        x = np.array([[0.3, 0.4],
                      [0.5, 0.6]])
        y = np.array([1, -1])
        out1 = self.model.loss(x, y)
        #print out1


        # output with perturbed b
        h = 0.0001
        self.model.b += h
        out2 = self.model.loss(x, y)
        #print out2

        should_be = (out2 - out1) / h
        #print should_be


        b_grad = self.model.grad_loss_wrt_b(x, y)

        self.assertTrue(np.abs(b_grad - should_be) < 0.001)

    def test_grad_loss_wrt_w(self):
        """
        Test LogisticRegression.grad_loss_wrt_w
        """
        # test numerically
        # first compute output
        self.model.w = np.array([[0.1, 0.2]])
        self.model.b = 0.2
        x = np.array([[0.3, 0.4],
                      [0.5, 0.6]])
        y = np.array([1, -1])
        out1 = self.model.loss(x, y)

        # output with perturbed w1
        h = 0.0001
        self.model.w += np.array([[h, 0]])
        out2 = self.model.loss(x, y)
        should_be_w1 = (out2 - out1) / h
        
        # output with perturbed w2
        self.model.w -= np.array([[h, 0]])
        self.model.w += np.array([[0, h]])

        out3 = self.model.loss(x, y)
        should_be_w2 = (out3 - out1) / h

        w_grad = self.model.grad_loss_wrt_w(x, y)
        should_be = np.array([[should_be_w1,
                               should_be_w2]])
        #print "should_be = ", should_be

        self.assertTrue(np.allclose(w_grad, should_be, atol=0.001))

    def test_predict(self):
        """
        Test LogisticRegression.predict
        """
        self.model.w = np.array([[0.1, 0.2]])
        self.model.b = 0.1
        x = np.array([[-0.5, -0.3],
                      [0.5, 0.6]])

        y = self.model.predict(x)

        self.assertTrue(np.array_equal(y, np.array([-1, 1])))


if __name__ == '__main__':
    unittest.main()
