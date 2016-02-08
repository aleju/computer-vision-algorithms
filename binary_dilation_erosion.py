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
    """
    im = [
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 1],
    ]
    im = np.array(im, dtype=np.uint8) * 255
    from scipy import misc
    im = misc.imresize(im, (64, 64)).astype(np.uint8)
    """
    im = np.zeros((128, 128), dtype=np.uint8)
    im[0, 0:15] = 255
    im[5:10, 5:10] = 255
    im[64:71, 64:71] = 255
    im[72:75, 72] = 255
    im[91:95, 24:29] = 255
    im[91:92, 29:31] = 255
    im[30:60, 40:80] = 255
    im[10:70, 100:102] = 255
    im[50:60, 98:100] = 255
    im[15, 20:100] = 255
    im[90:110, 80:120] = 255
    im[80:120, 90:110] = 255
    im[100:105, 100:105] = 0

    util.plot_images_grayscale(
        [im, dilation(im), erosion(im), opening(im), closing(im)],
        ["Binary Image", "Binary Dilation", "Binary Erosion", "Binary Opening", "Binary Closing"]
    )

def dilation(im):
    im = np.copy(im).astype(np.float32)
    mask = np.array([[1,1,1], [1,1,1], [1,1,1]])
    corr = signal.correlate(im, mask, mode="same") / 9.0
    corr[corr > 1e-8] = 255
    corr = corr.astype(np.uint8)
    return corr

def erosion(im):
    im = np.copy(im).astype(np.float32)
    mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    corr = signal.correlate(im, mask, mode="same") / 9.0
    corr[corr < 255.0 - 1e-8] = 0
    corr = corr.astype(np.uint8)
    return corr

def opening(im):
    return erosion(dilation(im))

def closing(im):
    return dilation(erosion(im))

if __name__ == "__main__":
    main()
