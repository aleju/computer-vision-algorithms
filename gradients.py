from __future__ import division, print_function
from scipy import signal, ndimage
import numpy as np
import random
from skimage import data
from skimage import util as skiutil
import util
np.random.seed(42)
random.seed(42)

def main():
    im = data.camera()
    imy_np, imx_np = np.gradient(im)
    imx_ski, imy_ski = _compute_derivatives(im)

    # dx
    util.plot_images_grayscale(
        [im, dx_symmetric(im), dx_forward(im), dx_backward(im), imx_np, imx_ski],
        ["Image", "dx (symmetric)", "dx (forward)", "dx (backward)", "Ground Truth (numpy)", "Ground Truth (scikit-image)"]
    )

    # dy
    util.plot_images_grayscale(
        [im, dy_symmetric(im), dy_forward(im), dy_backward(im), imy_np, imy_ski],
        ["Image", "dy (symmetric)", "dy (forward)", "dy (backward)", "Ground Truth (numpy)", "Ground Truth (scikit-image)"]
    )

# df_dx = f(y,x+1) - f(y,x-1)
def dx_symmetric(im):
    #return signal.correlate(I, np.array([-1, 1]), mode="same") / 2

    #return filters.correlate1d(I, np.array([-1, 1]), axis=1)
    imx = np.copy(im)
    imx[:, 1:-1] = imx[:, 2:] - imx[:, :-2]
    imx[:, 0] = 0
    imx[:, -1] = 0
    return imx

# df_dx = f(y,x+1) - f(y,x)
def dx_forward(im):
    imx = np.copy(im)
    imx[:, :-1] = imx[:, 1:] - imx[:, :-1]
    imx[:, -1] = 0
    return imx

# df_dx = f(y,x) - f(y,x-1)
def dx_backward(im):
    imx = np.copy(im)
    imx[:, 1:] = imx[:, 1:] - imx[:, :-1]
    imx[:, 0] = 0
    return imx

def dy_symmetric(im):
    return np.rot90(dx_symmetric(np.rot90(im)), 3)

def dy_forward(im):
    return np.rot90(dx_forward(np.rot90(im)), 3)

def dy_backward(im):
    return np.rot90(dx_backward(np.rot90(im)), 3)

# from skimage
def _compute_derivatives(image, mode='constant', cval=0):
    imy = ndimage.sobel(image, axis=0, mode=mode, cval=cval)
    imx = ndimage.sobel(image, axis=1, mode=mode, cval=cval)

    return imx, imy

if __name__ == "__main__":
    main()
