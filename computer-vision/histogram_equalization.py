"""Apply histogram equalization techniques to an image."""
from __future__ import division, print_function
import numpy as np
import random
from skimage import data
import matplotlib.pyplot as plt
np.random.seed(42)
random.seed(42)

def main():
    """Load image, apply histogram stretching and cumulative histogram
    equalization. Plot the results."""
    img = data.text()

    height, width = img.shape
    nb_pixels = height * width

    # apply histogram stretching
    gray_min = np.min(img)
    gray_max = np.max(img)
    img_hist_stretched = (img - gray_min) * (255 / (gray_max - gray_min))
    img_hist_stretched = img_hist_stretched.astype(np.uint8)

    # apply cumulative histogram equalization
    img_cumulative = np.zeros(img.shape, dtype=np.uint8)
    hist_cumulative = [0] * 256
    running_sum = 0
    hist, edges = np.histogram(img, 256, (0, 255))
    for i in range(256):
        count = hist[i]
        running_sum += count
        hist_cumulative[i] = running_sum / nb_pixels

    for i in range(256):
        img_cumulative[img == i] = int(256 * hist_cumulative[i])

    # plot
    plot_images_grayscale(
        [img, img_hist_stretched, img_cumulative],
        ["Image", "Histogram stretched to 0-255", "Cumulative Histogram Equalization"]
    )

def plot_images_grayscale(images, titles, no_axis=False):
    """Plot images with their histograms.
    Args:
        images List of images
        titles List of titles of images
        no_axis Whether to show the x/y axis"""
    fig = plt.figure()
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(2, len(images), i+1)
        ax.set_title(title)
        if no_axis:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.imshow(image, cmap=plt.cm.gray)

    for i, image in enumerate(images):
        hist, edges = np.histogram(image, 256, (0, 255))

        ax = fig.add_subplot(2, len(images), len(images)+i+1)
        if no_axis:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.bar(range(256), hist)

    plt.show()

if __name__ == "__main__":
    main()
