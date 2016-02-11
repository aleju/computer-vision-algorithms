"""Apply crosscorrelation and convolution to an image."""
from __future__ import division, print_function
from scipy import signal
import numpy as np
import random
from skimage import data
import util
np.random.seed(42)
random.seed(42)

def main():
    """Initialize kernel, apply it to an image (via crosscorrelation, convolution)."""
    img = data.camera()
    kernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    cc_response = crosscorrelate(img, kernel)
    cc_gt = signal.correlate(img, kernel, mode="same")

    conv_response = convolve(img, kernel)
    conv_gt = signal.convolve(img, kernel, mode="same")

    util.plot_images_grayscale(
        [img, cc_response, cc_gt, conv_response, conv_gt],
        ["Image", "Cross-Correlation", "Cross-Correlation (Ground Truth)", "Convolution", "Convolution (Ground Truth)"]
    )

def crosscorrelate(img, kernel):
    """Apply a kernel/filter via crosscorrelation to an image.
    Args:
        img The image
        kernel The kernel/filter to apply
    Returns:
        New image
    """
    imheight, imwidth = img.shape
    kheight, kwidth = kernel.shape
    assert len(img.shape) == 2
    assert kheight == kwidth # only square matrices
    assert kheight % 2 != 0 # sizes must be odd
    ksize = int((kheight - 1) / 2)
    im_pad = np.pad(img, ((ksize, ksize), (ksize, ksize)), mode="constant")
    response = np.zeros(img.shape)
    for y in range(ksize, ksize+imheight):
        for x in range(ksize, ksize+imwidth):
            patch = im_pad[y-ksize:y+ksize+1, x-ksize:x-ksize+1]
            corr = np.sum(patch * kernel)
            response[y-ksize, x-ksize] = corr
    return response

def convolve(img, kernel):
    """Apply a kernel/filter via convolution to an image.
    Args:
        img The image
        kernel The kernel/filter to apply
    Returns:
        New image
    """
    return crosscorrelate(img, np.flipud(np.fliplr(kernel)))

if __name__ == "__main__":
    main()
