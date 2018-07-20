import unittest
from image_class import ImageClass
import numpy as np


class TestImageClass(unittest.TestCase):
    def setUp(self):
        """
        Function that sets up the test by loading an example image
        """
        self.im_object = ImageClass('sparky.png')

    def test_init(self):
        """
        Test ImageClass.__init__ function
        """
        # make sure image is loaded
        self.assertTrue(self.im_object.im is not None)

        # make sure type is uint8
        self.assertEqual(self.im_object.im.dtype, np.dtype('uint8'))

        # make sure size looks correct
        self.assertTrue(np.allclose(self.im_object.im.shape, [300, 300, 3]))

    def test_crop(self):
        """
        Test ImageClass.crop function
        """
        # actually perform cropping
        # while saving original image
        orig_im = np.copy(self.im_object.im)
        self.im_object.crop(10, 15, 150, 170)

        # make sure size is correct
        self.assertTrue(np.array_equal(self.im_object.im.shape, [150, 170, 3]))

        # test a small window in the crop and original
        self.assertTrue(np.array_equal(orig_im[10:20, 15:25, :],
                                       self.im_object.im[0:10, 0:10, :]))

    def test_flip_horizontal(self):
        """
        Test ImageClass.flip_horizontal function
        """

        # Test with some random noise, because sparky has too many
        # same valued pixels
        np.random.seed(1337)
        self.im_object.im = np.random.randint(0, 255,
                                              size=self.im_object.im.shape)

        # copy the original and perform flipping
        orig_im = np.copy(self.im_object.im)
        self.im_object.flip_horizontal()

        # check some pixels that should be flipped
        self.assertTrue(np.array_equal(self.im_object.im[5, 5, :],
                                       orig_im[5, -6, :]))

        # check some pixels that should not be flipped
        self.assertFalse(np.array_equal(self.im_object.im[5, 5, :],
                                        orig_im[-6, 5, :]))

    def test_transpose(self):
        """
        Test ImageClass.transpose function
        """
        # Test with some random noise, because sparky has too many
        # same valued pixels
        np.random.seed(1337)
        self.im_object.im = np.random.randint(0, 255,
                                              size=self.im_object.im.shape)

        # copy the original and perform tranpose
        orig_im = np.copy(self.im_object.im)
        self.im_object.transpose()

        # check some pixels that should be transposed
        self.assertTrue(np.array_equal(self.im_object.im[5, 10, :],
                                       orig_im[10, 5, :]))

        self.assertTrue(np.array_equal(self.im_object.im[10, 5, :],
                                       orig_im[5, 10, :]))

    def test_reverse_channels(self):
        """
        Test ImageClass.reverse_channels function
        """
        # copy the data before and after reversing the channels
        orig_im = np.copy(self.im_object.im)
        self.im_object.reverse_channels()
        new_im = np.copy(self.im_object.im)

        # check that the channels are reversed
        self.assertTrue(np.array_equal(orig_im[:, :, 0], new_im[:, :, -1]))
        self.assertTrue(np.array_equal(orig_im[:, :, 1], new_im[:, :, -2]))
        self.assertTrue(np.array_equal(orig_im[:, :, 2], new_im[:, :, -3]))

    def test_split_channels(self):
        """
        Test ImageClass.split_channels function
        """
        r, g, b = self.im_object.split_channels()

        # check some pixel values in the resulting channels
        self.assertTrue(np.array_equal(self.im_object.im[:100, :100, 0],
                                       r[:100, :100]))
        self.assertTrue(np.array_equal(self.im_object.im[:100, :100, 1],
                                       g[:100, :100]))
        self.assertTrue(np.array_equal(self.im_object.im[:100, :100, 2],
                                       b[:100, :100]))

    def test_to_grayscale(self):
        """
        Test ImageClass.to_grayscale function
        """
        # perform transformation and copy original data
        orig_im = np.copy(self.im_object.im)
        self.im_object.to_grayscale()

        # check data type
        self.assertEqual(self.im_object.im.dtype, np.dtype('uint8'))

        # check some pixels
        # there will be a difference due to rounding, but should be less than 1
        self.assertTrue(self.im_object.im[150, 150] -
                        np.sum(orig_im[150, 150, :] *
                               np.array((0.30, 0.59, 0.11)))
                        < 1.0)

        self.assertTrue(self.im_object.im[150, 175] -
                        np.sum(orig_im[150, 175, :] *
                               np.array((0.30, 0.59, 0.11)))
                        < 1.0)

    def test_blur(self):
        """
        Test ImageClass.blur function
        """
        # save the original and blur the image
        orig_im = np.copy(self.im_object.im)
        self.im_object.blur()

        # check data type
        self.assertEqual(self.im_object.im.dtype, np.dtype('uint8'))

        # check that we are computing the average of 5x5 regions
        # allow a tolerance of 1 grayscale value
        # we check the three channels separately
        off_r = 60
        off_c = 120
        self.assertTrue(np.mean(orig_im[off_r:off_r+5, off_c:off_c+5, 0]) -
                        self.im_object.im[off_r, off_c, 0] <= 1)
        self.assertTrue(np.mean(orig_im[off_r:off_r+5, off_c:off_c+5, 1]) -
                        self.im_object.im[off_r, off_c, 1] <= 1)
        self.assertTrue(np.mean(orig_im[off_r:off_r+5, off_c:off_c+5, 2]) -
                        self.im_object.im[off_r, off_c, 2] <= 1)

    def test_histogram(self):
        """
        Test ImageClass.plot_histogram function
        """
        hist = self.im_object.plot_histogram(show=False)

        # histogram should sum to number of pixels
        self.assertEqual(np.sum(hist),
                         self.im_object.im.shape[0] * self.im_object.im.shape[1])

        # check some values of the histogram of sparky
        self.assertEqual(hist[255],
                         np.sum(self.im_object.im == 255))
        self.assertEqual(hist[0],
                         np.sum(self.im_object.im == 0))
        self.assertEqual(hist[200],
                         np.sum(self.im_object.im == 200))

    def test_dft(self):
        """
        Test ImageClass.compute_dft function

        Note: your implementation should not use numpy.fft.fft
        """
        # compute implemented dft with dft matrices
        dft1 = self.im_object.compute_dft()

        # use np's fft2 for comparison
        dft2 = np.fft.fft2(self.im_object.im, norm="ortho")

        # check if the two results are approximately equal
        self.assertTrue(np.allclose(dft1, dft2))


if __name__ == '__main__':
    unittest.main()