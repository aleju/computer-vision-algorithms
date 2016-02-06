from __future__ import division, print_function
from scipy import signal, ndimage
from scipy.ndimage import filters as filters
import numpy as np
import random
from skimage import data
from skimage import filters as skifilters
from skimage import util as skiutil
from skimage import feature
from skimage import draw
import util
import math
np.random.seed(42)
random.seed(42)


def main():
    # generate example image (noisy line)
    im = np.zeros((128, 128))
    for y in range(12, 120):
        im[y, y] = 1
    im = (im * 100) + np.random.binomial(80, 0.5, (im.shape))
    height, width = im.shape

    # fill hough accumulator (2nd plotted image)
    # based on pseudocode from https://de.wikipedia.org/wiki/Hough-Transformation
    max_d = math.sqrt(height**2 + width**2)
    min_d = -max_d
    accumulator = np.zeros((180, max_d-min_d))
    for y in range(height):
        for x in range(width):
            for alpha in range(0, 180):
                d = x * math.cos(np.deg2rad(alpha)) + y * math.sin(np.deg2rad(alpha))
                accumulator[alpha, d] += 1

    peak_idx = np.argmax(accumulator)
    peak_idx_tuple = np.unravel_index(peak_idx, accumulator.shape)
    peak_alpha, peak_d = peak_idx_tuple
    peak_alpha_rad = np.deg2rad(peak_alpha)
    #peak_y = peak_d * math.sin(peak_alpha_rad)
    #peak_x = peak_d * math.cos(peak_alpha_rad)

    # from http://scikit-image.org/docs/dev/auto_examples/plot_line_hough_transform.html
    x0 = 0
    x1 = width-1
    y0 = (peak_d - 0 * np.cos(peak_alpha_rad)) / np.sin(peak_alpha_rad)
    y1 = (peak_d - width * np.cos(peak_alpha_rad)) / np.sin(peak_alpha_rad)
    # ---

    # draw line
    im_hough = np.zeros(im.shape)
    rr, cc = draw.line(int(y0), int(x0), int(y1), int(x1))
    im_hough[rr, cc] = 1

    util.plot_images_grayscale(
        [im, accumulator, im_hough],
        ["Image", "Accumulator content", "Line from Hough Transform"]
    )

if __name__ == "__main__":
    main()
