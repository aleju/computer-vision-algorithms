from __future__ import division, print_function
from scipy import signal, ndimage
from scipy.ndimage import filters as filters
import numpy as np
import random
from skimage import data
from skimage import filters as skifilters
from skimage import util as skiutil
from skimage import feature
from skimage import draw
import util
import math
import matplotlib.pyplot as plt
np.random.seed(42)
random.seed(42)

def main():
    im = data.text()

    height, width = im.shape
    nb_pixels = height * width

    gray_min = np.min(im)
    gray_max = np.max(im)
    im_hist_stretched = (im - gray_min) * (255 / (gray_max - gray_min))
    im_hist_stretched = im_hist_stretched.astype(np.uint8)

    im_cumulative = np.zeros(im.shape, dtype=np.uint8)
    hist_cumulative = [0] * 256
    running_sum = 0
    hist, edges = np.histogram(im, 256, (0, 255))
    for i in range(256):
        count = hist[i]
        running_sum += count
        hist_cumulative[i] = running_sum / nb_pixels

    for i in range(256):
        im_cumulative[im == i] = int(256 * hist_cumulative[i])

    plot_images_grayscale(
        [im, im_hist_stretched, im_cumulative],
        ["Image", "Histogram stretched to 0-255", "Cumulative Histogram Equalization"]
    )

def plot_images_grayscale(images, titles, no_axis=False):
    fig = plt.figure()
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(2,len(images),i+1)
        ax.set_title(title)
        if no_axis:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.imshow(image, cmap=plt.cm.gray)

    for i, image in enumerate(images):
        #hist = np.bincount(image.flatten())
        hist, edges = np.histogram(image, 256, (0, 255))

        ax = fig.add_subplot(2,len(images),len(images)+i+1)
        if no_axis:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.bar(range(256), hist)

    plt.show()

if __name__ == "__main__":
    main()
