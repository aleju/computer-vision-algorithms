from __future__ import division, print_function
from scipy import signal, ndimage
from scipy.ndimage import filters as filters
import numpy as np
import random
from skimage import data
from skimage import filters as skifilters
from skimage import util as skiutil
import util
np.random.seed(42)
random.seed(42)

def main():
    im = data.camera()

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    sobel_x = np.rot90(sobel_y) # rotates counter-clockwise

    im_sx = signal.correlate(im, sobel_x, mode="same")
    im_sy = signal.correlate(im, sobel_y, mode="same")

    # combine via gradient magnitude
    # scikit-image's implementation divides by sqrt(2), not sure why
    im_s = np.sqrt(im_sx**2 + im_sy**2) / np.sqrt(2)

    threshold = np.average(im_s)
    im_s_bin = np.zeros(im_s.shape)
    im_s_bin[im_s > threshold] = 1

    gt = skifilters.sobel(data.camera())

    util.plot_images_grayscale([im, im_sx, im_sy, im_s, im_s_bin, gt], ["Image", "Sobel (x)", "Sobel (y)", "Sobel (both)", "Sobel (both, binarized)", "Sobel (Ground Truth)"])

if __name__ == "__main__":
    main()
