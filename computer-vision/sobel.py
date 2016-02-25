"""Apply Sobel filter to an image."""
from __future__ import division, print_function
from scipy import signal
import numpy as np
import random
from skimage import data
from skimage import filters as skifilters
import util
np.random.seed(42)
random.seed(42)

def main():
    """Load image, apply sobel (to get x/y gradients), plot the results."""
    img = data.camera()

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    sobel_x = np.rot90(sobel_y) # rotates counter-clockwise

    # apply x/y sobel filter to get x/y gradients
    img_sx = signal.correlate(img, sobel_x, mode="same")
    img_sy = signal.correlate(img, sobel_y, mode="same")

    # combine x/y gradients to gradient magnitude
    # scikit-image's implementation divides by sqrt(2), not sure why
    img_s = np.sqrt(img_sx**2 + img_sy**2) / np.sqrt(2)

    # create binarized image
    threshold = np.average(img_s)
    img_s_bin = np.zeros(img_s.shape)
    img_s_bin[img_s > threshold] = 1

    # generate ground truth (scikit-image method)
    ground_truth = skifilters.sobel(data.camera())

    # plot
    util.plot_images_grayscale(
        [img, img_sx, img_sy, img_s, img_s_bin, ground_truth],
        ["Image", "Sobel (x)", "Sobel (y)", "Sobel (magnitude)",
         "Sobel (magnitude, binarized)", "Sobel (Ground Truth)"]
    )

if __name__ == "__main__":
    main()
