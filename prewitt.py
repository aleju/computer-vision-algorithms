"""Apply Prewitt filter to an image to extract the x/y gradients."""
from __future__ import division, print_function
from scipy import signal
import numpy as np
import random
from skimage import data
from skimage import filters as skifilters
import util
np.random.seed(42)
random.seed(42)

def main():
    """Load image, apply filter, plot."""
    img = data.camera()

    # just like sobel, but no -2/+2 in the middle
    prewitt_y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    prewitt_x = np.rot90(prewitt_y) # rotates counter-clockwise

    img_sx = signal.correlate(img, prewitt_x, mode="same")
    img_sy = signal.correlate(img, prewitt_y, mode="same")
    g_magnitude = np.sqrt(img_sx**2 + img_sy**2)

    ground_truth = skifilters.prewitt(data.camera())

    util.plot_images_grayscale(
        [img, img_sx, img_sy, g_magnitude, ground_truth],
        ["Image", "Prewitt (x)", "Prewitt (y)", "Prewitt (both/magnitude)",
         "Prewitt (Ground Truth)"]
    )

if __name__ == "__main__":
    main()
