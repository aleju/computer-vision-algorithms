"""Apply Hough Transform to an image."""
from __future__ import division, print_function
import numpy as np
import random
from skimage import draw
from scipy import signal
import util
import math
np.random.seed(42)
random.seed(42)

NB_QUANTIZATION_STEPS = 128

def main():
    """Create a noisy line image, recover the line via hough transform and
    plot an unnoise image with that line."""
    # generate example image (noisy lines)
    img = np.zeros((128, 128))
    for y in range(12, 120):
        img[y, y] = 1
    for y in range(40, 75):
        img[y, 12] = 1
    for x in range(16, 64):
        img[int(10 + x*0.2), x] = 1
    img = (img * 100) + np.random.binomial(80, 0.5, (img.shape))
    height, width = img.shape

    accumulator, local_maxima, img_hough = hough(img, 5)

    util.plot_images_grayscale(
        [img, accumulator, local_maxima, img_hough],
        ["Image", "Accumulator content", "Local Maxima", "Line from Hough Transform"]
    )

def grad_magnitude(img):
    """Calculate the gradient magnitude of an image.
    Args:
        img The image
    Returns:
        gradient image"""
    img = img / 255.0
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    sobel_x = np.rot90(sobel_y) # rotates counter-clockwise

    # apply x/y sobel filter to get x/y gradients
    imgx = signal.correlate(img, sobel_x, mode="same")
    imgy = signal.correlate(img, sobel_y, mode="same")
    imgmag = np.sqrt(imgx**2 + imgy**2)

    return imgmag

def hough(img, nb_lines):
    """Applies the Hough Transformation to an image.
    Args:
        img The image
        nb_lines The number of lines to search for.
    Returns:
        Accumulator image,
        Local maxima in accumulator image,
        image with detected lines"""
    height, width = img.shape
    magnitude = grad_magnitude(img)
    mag_avg = np.average(magnitude)

    max_d = math.sqrt(height**2 + width**2)
    min_d = -max_d
    alphas = np.linspace(0, np.pi, NB_QUANTIZATION_STEPS)
    distances = np.linspace(min_d, max_d, NB_QUANTIZATION_STEPS)
    accumulator = np.zeros((NB_QUANTIZATION_STEPS, NB_QUANTIZATION_STEPS))
    for y in range(1, height-1):
        for x in range(1, width-1):
            if magnitude[y, x] > mag_avg:
                for alpha_idx, alpha in enumerate(alphas):
                    distance = x * math.cos(alpha) + y * math.sin(alpha)
                    distance_idx = util.quantize_idx(distance, distances)
                    accumulator[alpha_idx, distance_idx] += 1

    img_hough = np.zeros((height, width, 3))
    img_hough[:, :, 0] = np.copy(img)
    img_hough[:, :, 1] = np.copy(img)
    img_hough[:, :, 2] = np.copy(img)

    local_maxima = find_local_maxima(accumulator)
    peaks_idx = get_peak_indices(local_maxima, nb_lines)
    for value, (alpha_idx, distance_idx) in peaks_idx:
        peak_alpha_rad = alphas[alpha_idx]
        peak_distance = distances[distance_idx]

        x0 = 0
        x1 = width - 1
        y0 = (peak_distance - 0 * np.cos(peak_alpha_rad)) / (np.sin(peak_alpha_rad) + 1e-8)
        y1 = (peak_distance - (width-1) * np.cos(peak_alpha_rad)) / (np.sin(peak_alpha_rad) + 1e-8)

        y0 = np.clip(y0, 0, height-1)
        y1 = np.clip(y1, 0, height-1)

        rr, cc = draw.line(int(y0), int(x0), int(y1), int(x1))
        img_hough[rr, cc, 0] = 1
        img_hough[rr, cc, 1] = 0
        img_hough[rr, cc, 2] = 0

    return accumulator, local_maxima, img_hough

def find_local_maxima(arr, size=5):
    """Finds the local maxima in an image.
    Args:
        arr The image
        size Neighborhood size (3 => 3x3)
    Returns:
        Local maxima image"""
    ssize = int((size-1)/2)
    arr = np.copy(arr)
    peaks = np.zeros(arr.shape)
    h, w = arr.shape
    for y in range(ssize, h-ssize):
        for x in range(ssize, w-ssize):
            val = arr[y, x]
            if val > 0:
                neighborhood = np.copy(arr[y-ssize:y+ssize+1, x-ssize:x+ssize+1])
                neighborhood[ssize, ssize] = 0
                if val > np.max(neighborhood):
                    peaks[y, x] = val
    return peaks

def get_peak_indices(arr, n):
    """Finds the indices of the n highest values in an array.
    Args:
        arr Array to analyze.
        n Number of values.
    Returns:
        List of (value, (dim1 idx, dim2 idx, ...))"""
    indices = arr.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, arr.shape) for i in indices)
    return [(arr[i], i) for i in indices]

def hough_old(img):
    # fill hough accumulator (2nd plotted image)
    # based on pseudocode from https://de.wikipedia.org/wiki/Hough-Transformation
    max_d = math.sqrt(height**2 + width**2)
    min_d = -max_d
    accumulator = np.zeros((180, max_d-min_d))
    for y in range(height):
        for x in range(width):
            for alpha in range(0, 180):
                distance = x * math.cos(np.deg2rad(alpha)) + y * math.sin(np.deg2rad(alpha))
                accumulator[alpha, distance] += 1

    peak_idx = np.argmax(accumulator)
    peak_idx_tuple = np.unravel_index(peak_idx, accumulator.shape)
    peak_alpha, peak_d = peak_idx_tuple
    peak_alpha_rad = np.deg2rad(peak_alpha)

    # from http://scikit-image.org/docs/dev/auto_examples/plot_line_hough_transform.html
    x0 = 0
    x1 = width - 1
    y0 = (peak_d - 0 * np.cos(peak_alpha_rad)) / np.sin(peak_alpha_rad)
    y1 = (peak_d - width * np.cos(peak_alpha_rad)) / np.sin(peak_alpha_rad)
    # ---

    # draw line
    img_hough = np.zeros(img.shape)
    rr, cc = draw.line(int(y0), int(x0), int(y1), int(x1))
    img_hough[rr, cc] = 1

    return accumulator, img_hough

if __name__ == "__main__":
    main()
