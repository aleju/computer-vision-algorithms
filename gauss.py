"""Apply gaussian filters to an image."""
from __future__ import division, print_function
from scipy import signal
from scipy.ndimage import filters as filters
import numpy as np
import random
from skimage import data
import util
np.random.seed(42)
random.seed(42)

def main():
    """Apply several gaussian filters one by one and plot the results each time."""
    img = data.checkerboard()
    shapes = [(5, 5), (7, 7), (11, 11), (17, 17), (31, 31)]
    sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]
    smoothed = []
    for shape, sigma in zip(shapes, sigmas):
        smoothed = apply_gauss(img, gaussian_kernel(shape, sigma=sigma))
        ground_truth = filters.gaussian_filter(img, sigma)
        util.plot_images_grayscale(
            [img, smoothed, ground_truth],
            ["Image", "After gauss (sigma=%.1f)" % (sigma), "Ground Truth (scipy)"]
        )

def apply_gauss(img, filter_mask):
    """Apply a gaussian filter to an image.
    Args:
        img The image
        filter_mask The filter mask (2D numpy array)
    Returns:
        Modified image
    """
    return signal.correlate(img, filter_mask, mode="same") / np.sum(filter_mask)

# from http://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
def gaussian_kernel(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

if __name__ == "__main__":
    main()
