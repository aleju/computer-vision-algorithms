"""Calculate the harris corner score of an image."""
from __future__ import division, print_function
from scipy import signal, ndimage
import numpy as np
import random
from skimage import data
from skimage import feature
from skimage import util as skiutil
import util
np.random.seed(42)
random.seed(42)

def main():
    """Load image, calculate harris scores (window functions: matrix of ones, gauss)
    and plot the results."""
    img = data.checkerboard()
    score_window = harris_ones(img, 7)
    score_gauss = harris_gauss(img)
    util.plot_images_grayscale(
        [img, score_window, score_gauss, feature.corner_harris(img)],
        ["Image", "Harris-Score (ones)", "Harris-Score (gauss)", "Harris-Score (ground truth)"]
    )

def harris_gauss(img, sigma=1, k=0.05):
    """Calculate the harris score based on a gauss window function.
    Args:
        img The image to use for corner detection.
        sigma The sigma value for the gauss functions.
        k Weighting parameter during the final scoring (det vs. trace).
    Returns:
        Corner score image"""
    # Gradients
    img = skiutil.img_as_float(img)
    imgy, imgx = np.gradient(img)

    imgxy = imgx * imgy
    imgxx = imgx ** 2
    imgyy = imgy ** 2

    # compute parts of harris matrix
    a11 = ndimage.gaussian_filter(imgxx, sigma=sigma, mode="constant")
    a12 = ndimage.gaussian_filter(imgxy, sigma=sigma, mode="constant")
    a21 = a12
    a22 = ndimage.gaussian_filter(imgyy, sigma=sigma, mode="constant")

    # compute score per pixel
    det_a = a11 * a22 - a12 * a21
    trace_a = a11 + a22
    score = det_a - k * trace_a ** 2

    return score

def harris_ones(img, window_size, k=0.05):
    """Calculate the harris score based on a window function of diagonal ones.
    Args:
        img The image to use for corner detection.
        window_size Size of the window (NxN).
        k Weighting parameter during the final scoring (det vs. trace).
    Returns:
        Corner score image
    """
    # Gradients
    img = skiutil.img_as_float(img)
    imgy, imgx = np.gradient(img)

    imgxy = imgx * imgy
    imgxx = imgx ** 2
    imgyy = imgy ** 2

    # window function (matrix of diagonal ones)
    window = np.ones((window_size, window_size))

    # compute parts of harris matrix
    a11 = signal.correlate(imgxx, window, mode="same") / window_size
    a12 = signal.correlate(imgxy, window, mode="same") / window_size
    a21 = a12
    a22 = signal.correlate(imgyy, window, mode="same") / window_size

    # compute score per pixel
    det_a = a11 * a22 - a12 * a21
    trace_a = a11 + a22

    return det_a - k * trace_a ** 2

if __name__ == "__main__":
    main()
