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
    im = skiutil.img_as_float(data.camera())
    im = filters.gaussian_filter(im, sigma=1.5)

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    sobel_x = np.rot90(sobel_y) # rotates counter-clockwise

    im_sx = signal.correlate(im, sobel_x, mode="same")
    im_sy = signal.correlate(im, sobel_y, mode="same")

    g_magnitudes = np.sqrt(im_sx**2 + im_sy**2) / np.sqrt(2)
    g_orientations = np.arctan2(im_sy, im_sx)

    g_mag_nonmax = non_maximum_suppression(g_magnitudes, g_orientations)
    canny = hysteresis(g_mag_nonmax, 0.1, 0.001)

    gt = feature.canny(data.camera())

    util.plot_images_grayscale([im, g_magnitudes, g_mag_nonmax, canny, gt], ["Image", "After Sobel", "After nonmax", "Canny", "Canny (Ground Truth)"])

def non_maximum_suppression(g_magnitudes, g_orientations):
    gm, go = g_magnitudes, g_orientations
    gm_out = np.copy(gm)
    height, width = gm.shape
    for y in range(height):
        for x in range(width):
            theta = np.degrees(go[y, x])
            theta = 180 + theta if theta < 0 else theta
            theta = bisect.bisect([0, 45, 90, 135, 180], theta)

            north = gm[y-1, x] if y > 0 else 0
            south = gm[y+1, x] if y < height-1 else 0
            west = gm[y, x-1] if x > 0 else 0
            east = gm[y, x+1] if x < width-1 else 0
            northwest = gm[y-1, x-1] if y > 0 and x > 0 else 0
            northeast = gm[y-1, x+1] if y > 0 and x < width-1 else 0
            southwest = gm[y+1, x-1] if y < height-1 and x > 0 else 0
            southeast = gm[y+1, x+1] if y < height-1 and x < width-1 else 0

            if theta == 0 or theta == 180:
                gm_out[y, x] = gm[y, x] if gm[y, x] >= north and gm[y, x] >= south else 0
            elif theta == 45:
                gm_out[y, x] = gm[y, x] if gm[y, x] >= northwest and gm[y, x] >= southeast else 0
            elif theta == 90:
                gm_out[y, x] = gm[y, x] if gm[y, x] >= west and gm[y, x] >= east else 0
            else: # theta == 135
                gm_out[y, x] = gm[y, x] if gm[y, x] >= northeast and gm[y, x] >= southwest else 0

    return gm_out

def hysteresis(g_magnitudes, threshold_high, threshold_low):
    gm_strong = np.zeros(g_magnitudes.shape)
    gm_weak = np.zeros(g_magnitudes.shape)
    gm_strong[g_magnitudes >= threshold_high] = 1
    gm_weak[(g_magnitudes < threshold_high) & (g_magnitudes >= threshold_low)] = 1

    height, width = g_magnitudes.shape

    converged = False
    while not converged:
        converged = True
        for y in range(height):
            for x in range(width):
                if gm_weak[y, x] == 1:
                    y_start = y-1 if y > 0 else 0
                    y_end = y+1 if y < height else height-1
                    x_start = x-1 if x > 0 else 0
                    x_end = x+1 if x < width else width-1
                    neighborhood = gm_strong[y_start:y_end+1, x_start:x_end+1]
                    if np.sum(neighborhood) > 0:
                        gm_weak[y, x] = 0
                        gm_strong[y, x] = 1
                        converged = False

    return gm_strong

if __name__ == "__main__":
    main()
