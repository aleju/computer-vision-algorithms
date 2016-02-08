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
    im_haystack = skiutil.img_as_float(data.camera())
    im_needle = im_haystack[140:190, 220:270]
    im_sad = np.zeros(im_haystack.shape)

    height_h, width_h = im_haystack.shape
    height_n, width_n = im_needle.shape

    for y in range(height_h - height_n):
        for x in range(width_h - width_n):
            patch = im_haystack[y:y+height_n, x:x+width_n]
            im_sad[y, x] = sad(im_needle, patch)
    im_sad = im_sad / np.max(im_sad)

    im_sad[height_h-height_n:, :] = np.max(im_sad[0:height_h, 0:width_h])
    im_sad[:, width_h-width_n:] = np.max(im_sad[0:height_h, 0:width_h])

    util.plot_images_grayscale(
        [im_haystack, im_needle, im_sad],
        ["Image", "Image (Search Template)", "Matching (darkest = best match)"]
    )

def sad(im1, im2):
    return np.sum(np.abs(im1 - im2))

if __name__ == "__main__":
    main()
