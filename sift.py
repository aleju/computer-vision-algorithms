"""Apply a simplified version of the SIFT keypoint locator to an image."""
from __future__ import division, print_function
from scipy.ndimage import filters as filters
import numpy as np
import random
from skimage import data
from skimage import util as skiutil
from skimage import draw
import util
import math
import matplotlib.pyplot as plt
np.random.seed(42)
random.seed(42)

def main():
    """Load image, calculate Difference of Gaussians (DoG), find keypoints,
    draw them on the image, plot the results."""
    # load image
    img = skiutil.img_as_float(data.camera())

    # calculate Difference of Gaussians
    dog = generate_dog(img, 3, 4)

    # find keypoints
    keypoints = find_keypoints(img, dog)
    print("Found %d keypoints" % len(keypoints))

    # draw keypoints
    img = draw_keypoints(img, keypoints, 0.005)

    # plot
    plot_images([img], ["Image with 0.5% of all keypoints"])

def generate_dog(img, nb_octaves, nb_per_octave=4):
    """Generate the difference of gaussians of an image.
    Args:
        img    The input image
        nb_octaves   Number of octaves (groups of images with similar smoothing/sigmas)
        nb_per_octave Number of images in one octave (with increasing smoothing/sigmas)
    Returns:
        List of (difference image, sigma value)
    """
    spaces = []
    sigma_start = 1.6
    k_start = math.sqrt(2)
    for i in range(nb_octaves):
        sigma = sigma_start * (2 ** i)
        last_gauss = None
        for j in range(nb_per_octave+1):
            k = k_start ** (j+1)
            gauss = filters.gaussian_filter(img, k*sigma)
            if last_gauss is not None:
                diff = gauss - last_gauss
                spaces.append((diff, k*sigma))
            last_gauss = gauss
    return spaces

def find_keypoints(img, dog):
    """Find keypoints in an image using DoG-images.
    Args:
        img The image in which to find keypoints.
        dog The Difference of Gaussians images.
    Returns:
        List of ((y, x), orientation, id of DoG-image, sigma value of scale, "min"/"max")
    """
    extrema = []
    for scale_idx, (scale, scale_size) in enumerate(dog):
        print("Finding keypoints in scale with id %d, sigma %.2f" % (scale_idx, scale_size))

        # upwords neighbor
        nup = dog[scale_idx+1][0] if scale_idx+1 < len(dog) else None
        # downwards neighbor
        ndown = dog[scale_idx-1][0] if scale_idx > 0 else None

        grad_y, grad_x = np.gradient(scale)
        grad_orientation_rad = np.arctan2(grad_y, grad_x)

        height, width = scale.shape
        for y in range(1, height-1):
            for x in range(1, width-1):
                # extract 3x3 patch around the current pixel
                patch_img = img[y-1:y+2, x-1:x+2]

                # add only keypoints to the end results which are if areas
                # of higher contrast (to reduce the number of bad keypoints)
                if not is_low_contrast(patch_img):
                    center_val = scale[y+1, x+1] # value of the center pixel in the patch
                    orientation = np.degrees(grad_orientation_rad[y+1, x+1])

                    # collect neighboring pixels in neighboring scales
                    neighbors = [scale[y-1, x], scale[y, x+1], scale[y+1, x], scale[y, x-1]]
                    neighbors.extend(
                        [scale[y-1, x-1], scale[y-1, x+1],
                         scale[y+1, x+1], scale[y+1, x-1]]
                    )
                    if nup is not None:
                        neighbors.extend(
                            [nup[y-1, x], nup[y, x+1],
                             nup[y+1, x], nup[y, x-1]]
                        )
                    if ndown is not None:
                        neighbors.extend(
                            [ndown[y-1, x], ndown[y, x+1],
                             ndown[y+1, x], ndown[y, x-1]]
                        )

                    # only add keypoints which are the maximum or minimum among
                    # their collected neighbors (i.e. get local maxima/minima)
                    # (y+1, x+1) to get the center of the extracted patch
                    if center_val >= max(neighbors):
                        extrema.append(((y+1, x+1), orientation, scale_idx, scale_size, "max"))
                    elif center_val <= min(neighbors):
                        extrema.append(((y+1, x+1), orientation, scale_idx, scale_size, "min"))
    return extrema

def is_low_contrast(patch):
    """Estimate whether a patch is probably from a low contrast area in the image.
    Uses max() - min() instead of average/median, as that is faster.
    Args:
        patch    The patch in the image
    Returns:
        boolean"""
    return np.max(patch) - np.min(patch) < 0.1

def draw_keypoints(img, keypoints, draw_prob):
    """Draws for each keypoint a circle (roughly matching the sigma of the scale)
    with a line for the orientation.
    Args:
        img    The image to which to add the keypoints (gets copied)
        keypoints The keypoints to draw
        draw_prob Probability of drawing a keypoint (the lower the less keypoints are drawn)
    Returns:
        Image with keypoints"""
    height, width = img.shape
    img = np.copy(img)

    # convert from grayscale image to RGB so that keypoints can be drawn in red
    img = img[:, :, np.newaxis]
    img = np.repeat(img, 3, axis=2)

    for (y, x), orientation, scale_idx, scale_size, kp_type in keypoints:
        if draw_prob < 1.0 and random.random() <= draw_prob:
            # draw the circle
            radius = int(scale_size)
            rr, cc = draw.circle_perimeter(y, x, radius, shape=img.shape)
            img[rr, cc, 0] = 1.0
            img[rr, cc, 1:] = 0

            # draw orientation
            orientation = util.quantize(orientation, [-135, -90, -45, 0, 45, 90, 135, 180])
            x_start = x
            y_start = y
            if orientation == 0:
                x_end = x + radius
                y_end = y
            elif orientation == 45:
                x_end = x + radius*(1/math.sqrt(2))
                y_end = y + radius*(1/math.sqrt(2))
            elif orientation == 90:
                x_end = x
                y_end = y + radius
            elif orientation == 135:
                x_end = x - radius*(1/math.sqrt(2))
                y_end = y - radius*(1/math.sqrt(2))
            elif orientation == 180:
                x_end = x - radius
                y_end = y
            elif orientation == -135:
                x_end = x - radius*(1/math.sqrt(2))
                y_end = y - radius*(1/math.sqrt(2))
            elif orientation == -90:
                x_end = x
                y_end = y - radius
            elif orientation == -45:
                x_end = x + radius*(1/math.sqrt(2))
                y_end = y - radius*(1/math.sqrt(2))
            x_end = np.clip(x_end, 0, width-1)
            y_end = np.clip(y_end, 0, height-1)
            rr, cc = draw.line(int(y_start), int(x_start), int(y_end), int(x_end))
            img[rr, cc, 0] = 1.0
            img[rr, cc, 1:] = 0
    img = np.clip(img, 0, 1.0)

    return img

def plot_images(images, titles, no_axis=False):
    """Plot RGB images.
    Args:
        images    List of RGB images
        titles    List of titles for each image
        no_axis   Whether to show x/y axis"""
    fig = plt.figure()
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(1, len(images), i+1)
        ax.set_title(title)
        if no_axis:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()
