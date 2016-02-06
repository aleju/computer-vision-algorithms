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
import math
np.random.seed(42)
random.seed(42)

def main():
    im = data.coins()

    height, width = im.shape
    nb_pixels = height * width

    g_avg = np.average(im)
    p_g = [0] * 256
    for g in range(0, 256):
        p_g[g] = np.sum(im == g) / nb_pixels

    q_best = None
    threshold_best = None
    im_bin_best = None
    for t in range(1, 255):
        im_bin = np.zeros(im.shape)
        im_bin[im >= t] = 1

        p1 = np.sum(im_bin) / nb_pixels
        p0 = 1 - p1

        g0 = np.average(im[im_bin == 0]) if np.sum(im[im_bin == 0]) > 0 else 0
        g1 = np.average(im[im_bin == 1]) if np.sum(im[im_bin == 1]) > 0 else 0

        #g0 = g0 if not math.isnan(g0) else 0
        #g1 = g1 if not math.isnan(g1) else 0

        var0 = sum([(g-g0)**2 * p_g[g] for g in range(0,t+1)])
        var1 = sum([(g-g1)**2 * p_g[g] for g in range(t+1,256)])

        var_between = p0 * (g0 - g_avg)**2 + p1 * (g1 - g_avg)**2
        var_inner = p0 * var0**2 + p1 * var1**2
        q = var_between / var_inner if var_inner > 0 else 0

        print(t, p0, p1, g0, g1, g_avg, var_between, var_inner, q)
        if q_best is None or q_best < q:
            q_best = q
            threshold_best = t
            im_bin_best = im <= t

    gt_tresh = skifilters.threshold_otsu(im)
    gt = im <= gt_tresh
    util.plot_images_grayscale([im, im_bin_best, gt], ["Image", "Otsu", "Otsu (Ground Truth)"])

if __name__ == "__main__":
    main()
