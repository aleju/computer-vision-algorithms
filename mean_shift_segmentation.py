from __future__ import division, print_function
from scipy import signal, ndimage
from scipy.ndimage import filters as filters
import numpy as np
import random
from skimage import data
from skimage import filters as skifilters
from skimage import util as skiutil
from skimage import feature
from skimage import color
import util
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
np.random.seed(42)
random.seed(42)

def main():
    im_rgb = data.coffee()
    im_rgb = misc.imresize(im_rgb, (256, 256)) / 255.0
    im = color.rgb2hsv(im_rgb)
    height, width, channels = im.shape
    print("Image shape is: ", im.shape)

    print("Collecting pixels...")
    pixels = []
    for y in range(height):
        for x in range(width):
            pixel = im[y, x, ...]
            pixels.append([pixel[0], pixel[1], pixel[2], (y/height)*2.0, (x/width)*2.0])
            #pixels.append([pixel[0], (y/height)*2, (x/width)*2])
    pixels = np.array(pixels)
    print("Found %d pixels to cluster" % (len(pixels)))

    print("Clustering...")
    bandwidth = estimate_bandwidth(pixels, quantile=0.05, n_samples=500)
    clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = clusterer.fit_predict(pixels)
    labels_unique = set(labels)
    labels_counts = [(lu, len([l for l in labels if l == lu])) for lu in labels_unique]
    #labels_counts = sorted(labels_counts, key=lambda t: t[1], reverse=True)
    labels_unique = sorted(list(labels_unique), key=lambda l: labels_counts[l], reverse=True)
    nb_clusters = len(labels_unique)
    print("Found %d clusters" % (nb_clusters))
    print(labels.shape)

    print("Creating images of segments...")
    #im_segments = [np.zeros((height, width)) for label in labels_unique]
    im_segments = [np.copy(im_rgb)*0.25 for label in labels_unique]

    for y in range(height):
        for x in range(width):
            pixel_idx = (y*width) + x
            #print(pixel_idx, height, width)
            label = labels[pixel_idx]
            #im_segments[label][y, x] = 255
            im_segments[label][y, x, 0] = 1.0

    print("Plotting...")
    images = [im_rgb]
    titles = ["Image"]
    for i in range(min(8, nb_clusters)):
        images.append(im_segments[i])
        titles.append("Segment %d" % (i))

    plot_images(images, titles)

def plot_images(images, titles, no_axis=False):
    fig = plt.figure()
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(1,len(images),i+1)
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
