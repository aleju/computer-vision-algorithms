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
np.random.seed(42)
random.seed(42)

def main():
    im = data.camera()
    kernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    cc_response = crosscorrelate(im, kernel)
    cc_gt = signal.correlate(im, kernel, mode="same")

    conv_response = convolve(im, kernel)
    conv_gt = signal.convolve(im, kernel, mode="same")

    util.plot_images_grayscale(
        [im, cc_response, cc_gt, conv_response, conv_gt],
        ["Image", "Cross-Correlation", "Cross-Correlation (Ground Truth)", "Convolution", "Convolution (Ground Truth)"]
    )

def crosscorrelate(im, kernel):
    imheight, imwidth = im.shape
    kheight, kwidth = kernel.shape
    assert len(im.shape) == 2
    assert kheight == kwidth # only square matrices
    assert kheight % 2 != 0 # sizes must be odd
    ksize = int((kheight - 1) / 2)
    im_pad = np.pad(im, ((ksize, ksize), (ksize, ksize)), mode="constant")
    #from scipy import misc
    #misc.imshow(im_pad)
    response = np.zeros(im.shape)
    for y in range(ksize, ksize+imheight):
        for x in range(ksize, ksize+imwidth):
            patch = im_pad[y-ksize:y+ksize+1, x-ksize:x-ksize+1]
            corr = np.sum(patch * kernel)
            response[y-ksize, x-ksize] = corr
    return response

def convolve(im, kernel):
    return crosscorrelate(im, np.flipud(np.fliplr(kernel)))

if __name__ == "__main__":
    main()
