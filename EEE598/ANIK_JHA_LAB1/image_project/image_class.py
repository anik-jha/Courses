import numpy as np
import imageio
import scipy.signal as ss
import matplotlib.pyplot as plt


class ImageClass(object):
    def __init__(self, path):
        """ Loads the image specified by path """
        self.im = imageio.imread(path).astype('uint8')

    def show(self):
        """ Shows the image using Matplotlib's pyplot """
        plt.imshow(self.im)
        plt.show()

    def save(self, path):
        """ Saves the image to path """
        imageio.imsave(path, self.im)

    def crop(self, r, c, h, w):
        """
        Return a cropped version of the input with size w x h
        
        Parameters
        ----------
        r : int
            The row index to start the crop.
        c : int
            The column index to start the crop.
        h : int
            The height of the crop.
        w : int
            The width of the crop.
        """
        self.im = self.im[r:r+h, c:c+w]
        return self.im




    def flip_horizontal(self):
        """ Flip the image horizontally """
        self.im = np.fliplr(self.im)

        return self.im

    def transpose(self):
        """ Rotate the image by by 90 degrees """
        self.im = np.transpose(self.im, (1,0,2))

        return self.im

    def reverse_channels(self):
        """ Reverse the RGB channels to BGR
        """
        self.im = self.im[:, :, ::-1]

        return self.im

    def split_channels(self):
        """
        Return a list of Numpy arrays corresponding to the three channels

        E.g. return (red, green, blue)
        """
        red, green, blue = self.im[:, :, 0], self.im[:, :, 1], self.im[:, :, 2]

        return(red, green, blue)

    def to_grayscale(self):
        """ Convert the image to grayscale """
        red, green, blue = self.split_channels()
        temp = 0.30*red + 0.59*green + 0.11*blue
        self.im = temp.astype(np.uint8)

        return self.im

    def blur(self):
        """ Blur the image """
        filter1 = np.ones((5, 5))/25
        #red, green, blue = self.split_channels()
        red, green, blue = self.im[:, :, 0], self.im[:, :, 1], self.im[:, :, 2]

        red2 = ss.convolve2d(red, filter1, mode='valid')

        green2 = ss.convolve2d(green, filter1, mode='valid')
        blue2 = ss.convolve2d(blue, filter1, mode='valid')

        self.im = np.dstack((red2, green2, blue2))
        self.im = self.im.astype('uint8')

        return self.im

    def plot_histogram(self, show=True):
        """
        Plot and return the 1D histogram of the grayscale version of the image
        """
        self.im = self.to_grayscale()
        hist, bin_edges = np.histogram(self.im, bins = np.arange(0, 257))
        hist1 = np.asarray(hist)

        return (hist1)

    def compute_dft(self):
        """ Return the dft of the image """
        self.im = self.to_grayscale()
        W = self._DFT_matrix()
        dft = W.dot(self.im).dot(W)

        return dft

    def _DFT_matrix(self):
        """ Returns the dft matrix for the current image size """
        N = self.im.shape[0]
        i, j = np.meshgrid(np.arange(self.im.shape[0]),
                           np.arange(self.im.shape[1]))
        omega = np.exp(- 2 * np.pi * 1J / N)
        W = np.power(omega, i * j) / np.sqrt(N)
        return W
