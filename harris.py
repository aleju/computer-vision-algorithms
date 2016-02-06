from __future__ import division, print_function
from scipy import signal, ndimage
from scipy.ndimage import filters as filters
import numpy as np
import random
from skimage import data
from skimage import feature
from skimage import util
import matplotlib.pyplot as plt
import matplotlib.cm as cm
np.random.seed(42)
random.seed(42)

def main():
    im = data.checkerboard()
    score_window = harris_window(im, 7)
    score_gauss = harris_gauss(im)
    plot_images([im, score_window, score_gauss, feature.corner_harris(im)], ["Image", "Harris-Score (window)", "Harris-Score (gauss)", "Harris-Score (ground truth)"])

def harris_gauss(im, sigma=1, k=0.05):
    im = util.img_as_float(im)
    imy, imx = np.gradient(im)

    imxy = imx * imy
    imxx = imx ** 2
    imyy = imy ** 2

    # compute parts of harris matrix
    A11 = ndimage.gaussian_filter(imxx, sigma=1, mode="constant")
    A12 = ndimage.gaussian_filter(imxy, sigma=1, mode="constant")
    A21 = A12
    A22 = ndimage.gaussian_filter(imyy, sigma=1, mode="constant")

    # compute score per pixel
    detA = A11 * A22 - A12 * A21
    traceA = A11 + A22
    score = detA - k * traceA ** 2
    #score = 2*detA / (traceA + 1e-6)

    return score

def harris_window(im, windowSize, k=0.05):
    #gauss = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    #I = signal.correlate(I, gauss, mode="same") / np.sum(gauss)
    #print(I)
    #I = ndimage.gaussian_filter(I, sigma=sigma)

    #Ix = dx(I)
    #Iy = dy(I)

    im = util.img_as_float(im)
    imy, imx = np.gradient(im)
    #skIx, skIy = _compute_derivatives(I)
    #Iy = util.img_as_float(skIy)
    #Ix = util.img_as_float(skIx)

    imxy = imx * imy
    imxx = imx ** 2
    imyy = imy ** 2

    window = np.ones((windowSize, windowSize))

    # compute parts of harris matrix
    A11 = signal.correlate(imxx, window, mode="same") / windowSize
    A12 = signal.correlate(imxy, window, mode="same") / windowSize
    A21 = A12
    A22 = signal.correlate(imyy, window, mode="same") / windowSize

    #A11 = ndimage.gaussian_filter(Ixx, sigma=1, mode="constant")
    #A12 = ndimage.gaussian_filter(Ixy, sigma=1, mode="constant")
    #A21 = A12
    #A22 = ndimage.gaussian_filter(Iyy, sigma=1, mode="constant")
    #print("A11:", A11)

    #A11, A12, A22 = feature.structure_tensor(I, sigma)
    #A21 = A12

    # compute score per pixel
    detA = A11 * A22 - A12 * A21
    traceA = A11 + A22
    #score = np.divide(detA, traceA) # if element in traceA is zero, numpy returns 0
    #score[np.isnan(score)] = 0
    score = detA - k * traceA ** 2
    #score = 2*detA / (traceA + 1e-6)

    #score = (score / np.max(score)) * 255.0
    #print("score:", score)


    #plot_images([I, Ix, Iy, score, feature.corner_harris(I)], ["I", "Ix", "Iy", "Harris-Score", "Harris-Score Ground Truth"])
    #plot_images([I, Ix, Iy, score, feature.corner_harris(I)])
    #plot_images([score])
    return score

"""
def _compute_derivatives(image, mode='constant', cval=0):
    imy = ndimage.sobel(image, axis=0, mode=mode, cval=cval)
    imx = ndimage.sobel(image, axis=1, mode=mode, cval=cval)

    return imx, imy

def dx(I):
    #return signal.correlate(I, np.array([-1, 1]), mode="same") / 2

    #return filters.correlate1d(I, np.array([-1, 1]), axis=1)
    Ix = np.copy(I)
    Ix[:, 1:-1] = Ix[:, 2:] - Ix[:, :-2]
    Ix[:, -1] = Ix[:, -1] - Ix[:, -2]
    return Ix

def dy(I):
    #from scipy.ndimage import filters as filters
    #return filters.correlate1d(I, np.array([-1, 1]), axis=0)
    Itmp = dx(np.rot90(I))
    return np.rot90(Itmp, 3)
"""

def plot_images(images, titles):
    fig = plt.figure()
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(1,len(images),i+1)
        ax.set_title(title)
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        plt.imshow(image, cmap=cm.Greys_r)
    plt.show()

if __name__ == "__main__":
    main()
