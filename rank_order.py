from __future__ import division, print_function
from scipy import signal, ndimage
from scipy.ndimage import filters as filters
import numpy as np
import random
from skimage import data
from skimage import filters as skifilters
from skimage import util as skiutil
from skimage import feature
import util
import bisect
np.random.seed(42)
random.seed(42)

def main():
    im = data.moon()
    util.plot_images_grayscale(
        [im, median(im), morphological_edge(im),
         erosion(im), dilation(im),
         opening(im), closing(im)],
        ["Image", "Median", "Morphological Edge", "Erosion", "Dilation", "Opening", "Closing"]
    )

def median(im):
    return rank_order(im, lambda neighborhood: np.median(neighborhood))

def erosion(im):
    return rank_order(im, lambda neighborhood: neighborhood[0])

def dilation(im):
    return rank_order(im, lambda neighborhood: neighborhood[-1])

def morphological_edge(im):
    return rank_order(im, lambda neighborhood: neighborhood[-1] - neighborhood[0])

def opening(im):
    return dilation(erosion(im))

def closing(im):
    return erosion(dilation(im))

def rank_order(im, callback):
    result = np.zeros(im.shape)
    height, width = im.shape
    for y in range(height):
        for x in range(width):
            y_start = y-1 if y > 0 else 0
            y_end = y+1 if y < height else height-1
            x_start = x-1 if x > 0 else 0
            x_end = x+1 if x < width else width-1
            neighborhood = im[y_start:y_end+1, x_start:x_end+1].flatten()
            neighborhood = np.array(sorted(list(neighborhood)))
            result[y, x] = callback(neighborhood)
    return result

if __name__ == "__main__":
    main()
