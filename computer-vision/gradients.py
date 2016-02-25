"""Compute derivatives of an image with respect to x and y."""
from __future__ import division, print_function
from scipy import ndimage
import numpy as np
import random
from skimage import data
import util
np.random.seed(42)
random.seed(42)

def main():
    """Load image, compute derivatives, plot."""
    img = data.camera()
    imgy_np, imgx_np = np.gradient(img)
    imgx_ski, imgy_ski = _compute_derivatives(img)

    # dx
    util.plot_images_grayscale(
        [img, dx_symmetric(img), dx_forward(img), dx_backward(img), imgx_np, imgx_ski],
        ["Image", "dx (symmetric)", "dx (forward)", "dx (backward)",
         "Ground Truth (numpy)", "Ground Truth (scikit-image)"]
    )

    # dy
    util.plot_images_grayscale(
        [img, dy_symmetric(img), dy_forward(img), dy_backward(img), imgy_np, imgy_ski],
        ["Image", "dy (symmetric)", "dy (forward)", "dy (backward)",
         "Ground Truth (numpy)", "Ground Truth (scikit-image)"]
    )

def dx_symmetric(img):
    """Calculate the derivative with respect to x of an image.
    Symmetric formula: df_dx = f(y,x+1) - f(y,x-1)
    Args:
        img The image
    Returns:
        x-derivate of the image as a new image"""
    imgx = np.copy(img)
    imgx[:, 1:-1] = imgx[:, 2:] - imgx[:, :-2]
    imgx[:, 0] = 0
    imgx[:, -1] = 0
    return imgx

def dx_forward(img):
    """Calculate the derivative with respect to x of an image.
    Forward formula: df_dx = f(y,x+1) - f(y,x)
    Args:
        img The image
    Returns:
        x-derivate of the image as a new image"""
    imgx = np.copy(img)
    imgx[:, :-1] = imgx[:, 1:] - imgx[:, :-1]
    imgx[:, -1] = 0
    return imgx

def dx_backward(img):
    """Calculate the derivative with respect to x of an image.
    Backward formula: df_dx = f(y,x) - f(y,x-1)
    Args:
        img The image
    Returns:
        x-derivate of the image as a new image"""
    imgx = np.copy(img)
    imgx[:, 1:] = imgx[:, 1:] - imgx[:, :-1]
    imgx[:, 0] = 0
    return imgx

def dy_symmetric(img):
    """Calculate the derivative with respect to y of an image.
    Symmetric formula: df_dy = f(y+1,x) - f(y-1,x)
    Args:
        img The image
    Returns:
        y-derivate of the image as a new image"""
    return np.rot90(dx_symmetric(np.rot90(img)), 3)

def dy_forward(img):
    """Calculate the derivative with respect to y of an image.
    Symmetric formula: df_dy = f(y+1,x) - f(y,x)
    Args:
        img The image
    Returns:
        y-derivate of the image as a new image"""
    return np.rot90(dx_forward(np.rot90(img)), 3)

def dy_backward(img):
    """Calculate the derivative with respect to y of an image.
    Backward formula: df_dy = f(y,x) - f(y-1,x)
    Args:
        img The image
    Returns:
        y-derivate of the image as a new image"""
    return np.rot90(dx_backward(np.rot90(img)), 3)

def _compute_derivatives(image, mode='constant', cval=0):
    """Compute derivatives the way that scikit-image does it (for comparison).
    This method is fully copied from the repository."""
    imgy = ndimage.sobel(image, axis=0, mode=mode, cval=cval)
    imgx = ndimage.sobel(image, axis=1, mode=mode, cval=cval)

    return imgx, imgy

if __name__ == "__main__":
    main()
