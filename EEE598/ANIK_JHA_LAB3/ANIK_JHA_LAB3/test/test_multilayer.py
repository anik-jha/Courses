import unittest
import numpy as np
from layers.full import FullLayer
from layers.softmax import SoftMaxLayer
from layers.cross_entropy import CrossEntropyLayer
from layers.sequential import Sequential
from layers.relu import ReluLayer


class TestMultiLayer(unittest.TestCase):
    """
    test a multilayer network (check gradients)
    """
    def setUp(self):
        """
        Set up a small sequential network
        """
        self.layer1 = FullLayer(5, 10)
        self.relu1 = ReluLayer()
        self.layer2 = FullLayer(10, 2)
        self.softmax = SoftMaxLayer()
        self.loss = CrossEntropyLayer()

        self.model = Sequential(
            (self.layer1,
             self.relu1,
             self.layer2,
             self.softmax),
            self.loss
            )

    def test_forward(self):
        """
        Test forward pass with some fake input
        """
        # make some fake input
        x = np.array([[0.5, 0.2, 0.1, 0.3, 0.7],
                      [0.3, 0.1, 0.05, 0.8, 0.9]])
        y = np.array([[0, 1],
                      [1, 0]])

        # test layers individually
        y1 = self.layer1.forward(x)
        relu = self.relu1.forward(y1)
        y2 = self.layer2.forward(relu)
        soft = self.softmax.forward(y2)
        loss = self.loss.forward(soft, y)

        # test sequential model
        y_model = self.model.forward(x, y)
        #print "test loss: ", loss
        self.assertAlmostEquals(loss, y_model)

    def test_backward(self):
        """
        Test the backward function using the numerical gradient
        """
        # make some fake input
        x = np.array([[0.5, 0.2, 0.1, 0.3, 0.7],
                      [0.3, 0.1, 0.05, 0.8, 0.9],
                  ])
        y = np.array([[0, 1],
                      [1, 0]])

        out = self.model.forward(x, y)
        grads = self.model.backward()

        # test some gradients at the input
        h = 0.001
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                new_x = np.copy(x)
                new_x[i, j] += h

                out2 = self.model.forward(new_x, y)
                
                diff = (out2 - out) / h

                self.assertTrue(np.abs(diff - grads[i,j]) < 0.001)

if __name__ == '__main__':
    unittest.main()
