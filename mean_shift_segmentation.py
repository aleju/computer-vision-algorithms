"""Segment an image using mean shift clustering."""
from __future__ import division, print_function
import numpy as np
import random
from skimage import data
from skimage import color
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy import misc
import matplotlib.pyplot as plt
np.random.seed(42)
random.seed(42)

def main():
    """Load image, collect pixels, cluster, create segment images, plot."""
    # load image
    img_rgb = data.coffee()
    img_rgb = misc.imresize(img_rgb, (256, 256)) / 255.0
    img = color.rgb2hsv(img_rgb)
    height, width, channels = img.shape
    print("Image shape is: ", img.shape)

    # collect pixels as tuples of (r, g, b, y, x)
    print("Collecting pixels...")
    pixels = []
    for y in range(height):
        for x in range(width):
            pixel = img[y, x, ...]
            pixels.append([pixel[0], pixel[1], pixel[2], (y/height)*2.0, (x/width)*2.0])
    pixels = np.array(pixels)
    print("Found %d pixels to cluster" % (len(pixels)))

    # cluster the pixels using mean shift
    print("Clustering...")
    bandwidth = estimate_bandwidth(pixels, quantile=0.05, n_samples=500)
    clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = clusterer.fit_predict(pixels)

    # process labels generated during clustering
    labels_unique = set(labels)
    labels_counts = [(lu, len([l for l in labels if l == lu])) for lu in labels_unique]
    labels_unique = sorted(list(labels_unique), key=lambda l: labels_counts[l], reverse=True)
    nb_clusters = len(labels_unique)
    print("Found %d clusters" % (nb_clusters))
    print(labels.shape)

    print("Creating images of segments...")
    img_segments = [np.copy(img_rgb)*0.25 for label in labels_unique]

    for y in range(height):
        for x in range(width):
            pixel_idx = (y*width) + x
            label = labels[pixel_idx]
            img_segments[label][y, x, 0] = 1.0

    print("Plotting...")
    images = [img_rgb]
    titles = ["Image"]
    for i in range(min(8, nb_clusters)):
        images.append(img_segments[i])
        titles.append("Segment %d" % (i))

    plot_images(images, titles)

def plot_images(images, titles, no_axis=False):
    """Plot images with titles.
    Args:
        images List of images to plot (color or grayscale).
        titles List of titles of the images.
        no_axis Whether to show x/y axis."""
    fig = plt.figure()
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(1, len(images), i+1)
        ax.set_title(title)
        if no_axis:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        if len(image.shape) == 2:
            plt.imshow(image, cmap=plt.cm.gray)
        else:
            plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()
