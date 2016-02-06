from __future__ import division, print_function
from scipy import signal, ndimage
from scipy.ndimage import filters as filters
import numpy as np
import random
from skimage import data
from skimage import util as skiutil
import util
np.random.seed(42)
random.seed(42)

def main():
    im = data.checkerboard()
    shapes = [(5,5), (7,7), (11,11), (17,17), (31,31)]
    sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]
    smoothed = []
    for shape, sigma in zip(shapes, sigmas):
        smoothed = apply_gauss(im, gaussian_kernel(shape, sigma=sigma))
        gt = filters.gaussian_filter(im, sigma)
        util.plot_images_grayscale(
            [im, smoothed, gt],
            ["Image", "After gauss (sigma=%.1f)" % (sigma), "Ground Truth (scipy)"]
        )

def apply_gauss(im, filter):
    return signal.correlate(im, filter, mode="same") / np.sum(filter)

# from http://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
def gaussian_kernel(shape=(3,3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

if __name__ == "__main__":
    main()
