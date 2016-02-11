"""Apply binary (0/1) dilation, erosion, closing and opening to an example
image."""
from __future__ import division, print_function
from scipy import signal
import numpy as np
import random
import util
np.random.seed(42)
random.seed(42)

def main():
    """Main function."""
    img = np.zeros((128, 128), dtype=np.uint8)
    img[0, 0:15] = 255
    img[5:10, 5:10] = 255
    img[64:71, 64:71] = 255
    img[72:75, 72] = 255
    img[91:95, 24:29] = 255
    img[91:92, 29:31] = 255
    img[30:60, 40:80] = 255
    img[10:70, 100:102] = 255
    img[50:60, 98:100] = 255
    img[15, 20:100] = 255
    img[90:110, 80:120] = 255
    img[80:120, 90:110] = 255
    img[100:105, 100:105] = 0

    util.plot_images_grayscale(
        [img, dilation(img), erosion(img), opening(img), closing(img)],
        ["Binary Image", "Binary Dilation", "Binary Erosion", "Binary Opening", "Binary Closing"]
    )

def dilation(img):
    """Perform Dilation on an image.
    Args:
        img The image to be changed.
    Returns:
        Changed image
    """
    img = np.copy(img).astype(np.float32)
    mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    corr = signal.correlate(img, mask, mode="same") / 9.0
    corr[corr > 1e-8] = 255
    corr = corr.astype(np.uint8)
    return corr

def erosion(img):
    """Perform Erosion on an image.
    Args:
        img The image to be changed.
    Returns:
        Changed image
    """
    img = np.copy(img).astype(np.float32)
    mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    corr = signal.correlate(img, mask, mode="same") / 9.0
    corr[corr < 255.0 - 1e-8] = 0
    corr = corr.astype(np.uint8)
    return corr

def opening(img):
    """Perform Opening on an image.
    Args:
        img The image to be changed.
    Returns:
        Changed image
    """
    return erosion(dilation(img))

def closing(img):
    """Perform Closing on an image.
    Args:
        img The image to be changed.
    Returns:
        Changed image
    """
    return dilation(erosion(img))

if __name__ == "__main__":
    main()
