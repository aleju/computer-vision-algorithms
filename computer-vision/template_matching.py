"""Finds an example image in a larger image via template matching."""
from __future__ import division, print_function
import numpy as np
import random
from skimage import data
from skimage import util as skiutil
import util
np.random.seed(42)
random.seed(42)

def main():
    """Load template image (needle) and the large example image (haystack),
    generate matching score per pixel, plot the results."""
    img_haystack = skiutil.img_as_float(data.camera()) # the image in which to search
    img_needle = img_haystack[140:190, 220:270] # the template to search for
    img_sad = np.zeros(img_haystack.shape) # score image

    height_h, width_h = img_haystack.shape
    height_n, width_n = img_needle.shape

    # calculate score for each pixel
    # stop iterating over pixels when the whole template cannot any more (i.e. stop
    # at bottom and right border)
    for y in range(height_h - height_n):
        for x in range(width_h - width_n):
            patch = img_haystack[y:y+height_n, x:x+width_n]
            img_sad[y, x] = sad(img_needle, patch)
    img_sad = img_sad / np.max(img_sad)

    # add highest score to bottom and right borders
    img_sad[height_h-height_n:, :] = np.max(img_sad[0:height_h, 0:width_h])
    img_sad[:, width_h-width_n:] = np.max(img_sad[0:height_h, 0:width_h])

    # plot results
    util.plot_images_grayscale(
        [img_haystack, img_needle, img_sad],
        ["Image", "Image (Search Template)", "Matching (darkest = best match)"]
    )

def sad(img1, img2):
    """Calculate the sum of absolute differences between two image patches.
    Args:
        im1    Image patch
        im2    Image patch
    Results:
        Sum of absolute differences"""
    return np.sum(np.abs(img1 - img2))

if __name__ == "__main__":
    main()
