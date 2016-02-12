"""Binarize an image using the Otsu method."""
from __future__ import division, print_function
import numpy as np
import random
from skimage import data
from skimage import filters as skifilters
import util
np.random.seed(42)
random.seed(42)

def main():
    """Load image, calculate optimal threshold, binarize, plot."""
    # load image
    img = data.coins()
    height, width = img.shape
    nb_pixels = height * width

    # precalculate some values for speedup
    # average pixel value
    g_avg = np.average(img)
    # P(pixel-value), i.e. #pixels-with-value / #all-pixels
    p_g = [0] * 256
    for g in range(0, 256):
        p_g[g] = np.sum(img == g) / nb_pixels

    # Otsu method
    # calculations are based on standard formulas
    q_best = None
    threshold_best = None
    img_bin_best = None
    # iterate over all possible thresholds
    for t in range(1, 255):
        img_bin = np.zeros(img.shape)
        img_bin[img >= t] = 1

        p1 = np.sum(img_bin) / nb_pixels
        p0 = 1 - p1

        g0 = np.average(img[img_bin == 0]) if np.sum(img[img_bin == 0]) > 0 else 0
        g1 = np.average(img[img_bin == 1]) if np.sum(img[img_bin == 1]) > 0 else 0

        var0 = sum([(g-g0)**2 * p_g[g] for g in range(0, t+1)])
        var1 = sum([(g-g1)**2 * p_g[g] for g in range(t+1, 256)])

        var_between = p0 * (g0 - g_avg)**2 + p1 * (g1 - g_avg)**2
        var_inner = p0 * var0**2 + p1 * var1**2

        # q is the relation of variance between classes and variance within classes
        q = var_between / var_inner if var_inner > 0 else 0

        print(t, p0, p1, g0, g1, g_avg, var_between, var_inner, q)
        if q_best is None or q_best < q:
            q_best = q
            threshold_best = t
            img_bin_best = img <= t

    # ground truth, based on scikit-image
    gt_tresh = skifilters.threshold_otsu(img)
    ground_truth = img <= gt_tresh

    # plot
    util.plot_images_grayscale(
        [img, img_bin_best, ground_truth],
        ["Image", "Otsu", "Otsu (Ground Truth)"]
    )

if __name__ == "__main__":
    main()
