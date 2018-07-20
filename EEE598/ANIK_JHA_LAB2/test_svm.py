from svm import SVM
import unittest
import numpy as np


class TestSVM(unittest.TestCase):
    def setUp(self):
        self.model = SVM()

    def test_forward(self):
        """
        Test SVM.forward function using some hand chosen values
        """

        # test single input
        self.model.w = np.array([[0.5, 0.25]])
        self.model.b = 0.5
        x = np.array([[0.2, 0.1]])
        out = self.model.forward(x)
        self.assertTrue(np.abs(out[0] - 0.6250) < 0.01)

        # test multiple inputs
        self.model.w = np.array([[0.1, 0.2]])
        self.model.b = 0.2
        x = np.array([[0.3, 0.4],
                      [0.5, 0.6]])
        out = self.model.forward(x)
        should_be = np.array([0.31, 0.37])
        self.assertTrue(np.allclose(out, should_be, atol=0.01))

    def test_loss(self):
        """
        Test SVM.loss
        """
        self.model.w = np.array([[0.5, 0.6]])
        self.model.b = 0.05
        self.model.l2_reg = 0.1
        x = np.array([[1.7, 0.9],
                      [0.5, 0.6]])
        y = np.array([1, -1])
        out = self.model.loss(x, y)
        should_be = 0.8605

        self.assertTrue(np.abs(out - should_be) < 0.01)

    def test_grad_loss_wrt_b(self):
        """
        Test SVM.grad_loss_wrt_b
        """
        # test numerically
        # first compute output
        self.model.w = np.array([[0.5, 0.6]])
        self.model.b = 0.05
        self.model.l2_reg = 0.1
        x = np.array([[1.7, 0.9],
                      [0.5, 0.6]])
        y = np.array([1, -1])
        out1 = self.model.loss(x, y)

        # output with perturbed b
        h = 0.0001
        self.model.b += h
        out2 = self.model.loss(x, y)
        
        should_be = (out2 - out1) / h

        b_grad = self.model.grad_loss_wrt_b(x, y)

        self.assertTrue(np.abs(b_grad - should_be) < 0.001)

    def test_grad_loss_wrt_w(self):
        """
        Test LogisticRegression.grad_loss_wrt_w
        """
        # test numerically
        # first compute output
        self.model.w = np.array([[0.5, 0.6]])
        self.model.b = 0.05
        self.model.l2_reg = 0.1
        x = np.array([[1.7, 0.9],
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
        print "Should_be = ", should_be
        self.assertTrue(np.allclose(w_grad, should_be, atol=0.001))

    def test_predict(self):
        """
        Test SVM.predict
        """
        self.model.w = np.array([[0.1, 0.2]])
        self.model.b = 0.1
        x = np.array([[-0.5, -0.3],
                      [0.5, 0.6]])

        y = self.model.predict(x)

        self.assertTrue(np.array_equal(y, np.array([-1, 1])))


if __name__ == '__main__':
    unittest.main()
