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

    # just like sobel, but no -2/+2 in the middle
    prewitt_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ])
    prewitt_x = np.rot90(prewitt_y) # rotates counter-clockwise

    im_sx = signal.correlate(im, prewitt_x, mode="same")
    im_sy = signal.correlate(im, prewitt_y, mode="same")
    g_magnitude = np.sqrt(im_sx**2 + im_sy**2)

    gt = skifilters.prewitt(data.camera())

    util.plot_images_grayscale([im, im_sx, im_sy, g_magnitude, gt], ["Image", "Prewitt (x)", "Prewitt (y)", "Prewitt (both)", "Prewitt (Ground Truth)"])

if __name__ == "__main__":
    main()
