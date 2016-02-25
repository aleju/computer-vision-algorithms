"""Apply rank order filters to an image."""
from __future__ import division, print_function
import numpy as np
import random
from skimage import data
import util
np.random.seed(42)
random.seed(42)

def main():
    """Load image, apply filters and plot the results."""
    img = data.moon()
    util.plot_images_grayscale(
        [img, median(img), morphological_edge(img),
         erosion(img), dilation(img),
         opening(img), closing(img)],
        ["Image", "Median", "Morphological Edge", "Erosion", "Dilation", "Opening", "Closing"]
    )

def median(img):
    """Apply a median rank order filter to an image.
    Picks for each pixel the median pixel value in the neighborhood around
    the pixel.
    Args:
        img    The input image
    Returns:
        Modified image"""
    return rank_order(img, np.median)

def erosion(img):
    """Apply an erosion rank order filter to an image.
    Picks for each pixel the lowest value in the neighborhood around
    the pixel.
    Args:
        img    The input image
    Returns:
        Modified image"""
    return rank_order(img, lambda neighborhood: neighborhood[0])

def dilation(img):
    """Apply a dilation rank order filter to an image.
    Picks for each pixel the highest pixel value in the neighborhood around
    the pixel.
    Args:
        img    The input image
    Returns:
        Modified image"""
    return rank_order(img, lambda neighborhood: neighborhood[-1])

def morphological_edge(img):
    """Apply a edge detection rank order filter to an image.
    Changes each pixel to the difference of the highest and lowest value in the
    neighborhood around the pixel.
    Args:
        img    The input image
    Returns:
        Modified image"""
    return rank_order(img, lambda neighborhood: neighborhood[-1] - neighborhood[0])

def opening(img):
    """Apply an opening rank order filter to an image.
    Applies first an erosion and then a dilation. Useful to get rid of smaller
    shapes/artefacts.
    Args:
        img    The input image
    Returns:
        Modified image"""
    return dilation(erosion(img))

def closing(img):
    """Apply a closing rank order filter to an image.
    Applies first a dilation and then an erosion. Useful to fill small
    holes/gaps.
    Args:
        img    The input image
    Returns:
        Modified image"""
    return erosion(dilation(img))

def rank_order(img, callback):
    """General rank order filter.
    Applies a callback to each pixel in the image. The callback receives the
    sorted pixel values in the neighborhood around the pixel and has to return
    a new pixel value.
    Args:
        img    The input image
        callback    The callback function to apply to each pixel neihborhood
    Returns:
        Modified image"""
    result = np.zeros(img.shape)
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            y_start = y-1 if y > 0 else 0
            y_end = y+1 if y < height else height-1
            x_start = x-1 if x > 0 else 0
            x_end = x+1 if x < width else width-1
            neighborhood = img[y_start:y_end+1, x_start:x_end+1].flatten()
            neighborhood = np.array(sorted(list(neighborhood)))
            result[y, x] = callback(neighborhood)
    return result

if __name__ == "__main__":
    main()
