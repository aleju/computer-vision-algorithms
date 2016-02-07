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
import bisect
np.random.seed(42)
random.seed(42)

def main():
    im = skiutil.img_as_float(data.camera())
    dog = generate_dog(im, 4, 4)
    keypoints = find_keypoints(im, dog)
    print("Keypoints:", len(keypoints))

    im = draw_keypoints(im, keypoints)

    plots = [im]
    #plots.extend(dog)
    titles = ["" for plot in plots]
    plot_images(
        plots,
        titles
    )

def generate_dog(im, nb_spaces, nb_per_sigma=4):
    spaces = []
    sigma_start = 1.6
    k_start = math.sqrt(2)
    for i in range(nb_spaces):
        sigma = sigma_start * (2 ** i)
        last_gauss = None
        for j in range(nb_per_sigma+1):
            k = k_start ** (j+1)
            gauss = filters.gaussian_filter(im, k*sigma)
            if last_gauss is not None:
                diff = gauss - last_gauss
                #diff = ndimage.gaussian_laplace(im, k+sigma)
                #print(diff)
                spaces.append((diff, k*sigma))
                #from scipy import misc
                #misc.imshow(diff)
            last_gauss = gauss
    return spaces

def find_keypoints(im, dog):
    ps = 1 # patch size in px (to left/right/top/bottom) => 1 = 3x3 patches, 2 => 5x5
    psf = 1 + ps*2 # full patch size (both sides + 1px for center)
    #im_pad = np.pad(im, ((ps, ps), (ps, ps)), mode="constant")
    extrema = []
    for scale_idx, (scale, scale_size) in enumerate(dog):
        print("finding keypoints in scale ", scale_idx)
        #scale = feature.corner_harris(scale)
        #scale = np.pad(scale, ((ps, ps), (ps, ps)), mode="constant")

        neighbor_up = dog[scale_idx+1][0] if scale_idx+1 < len(dog) else None
        neighbor_down = dog[scale_idx-1][0] if scale_idx > 0 else None
        #if neighbor_up is not None:
        #    neighbor_up = np.pad(neighbor_up, ((ps, ps), (ps, ps)), mode="constant")
        #if neighbor_down is not None:
        #    neighbor_down = np.pad(neighbor_down, ((ps, ps), (ps, ps)), mode="constant")

        grad_y, grad_x = np.gradient(scale)
        grad_orientation_rad = np.arctan2(grad_y, grad_x)

        height, width = scale.shape
        for y in range(1, height-1):
            for x in range(1, width-1):
                patch_im = im[y:y+psf+1, x:x+psf+1]
                if not is_low_contrast(patch_im):
                    #patch = scale[y:y+psf+1, x:x+psf+1]
                    #center_val = patch[ps, ps]
                    center_val = scale[y, x]
                    orientation = np.degrees(grad_orientation_rad[y, x])

                    #neighbors = list(patch.flatten())
                    neighbors = [scale[y-1, x], scale[y, x+1], scale[y+1, x], scale[y, x-1]]
                    neighbors.extend([scale[y-1, x-1], scale[y-1, x+1], scale[y+1, x+1], scale[y+1, x-1]])
                    if neighbor_up is not None:
                        nu = neighbor_up
                        neighbors.extend([nu[y-1, x], nu[y, x+1], nu[y+1, x], nu[y, x-1]])
                    if neighbor_down is not None:
                        nd = neighbor_down
                        neighbors.extend([nd[y-1, x], nd[y, x+1], nd[y+1, x], nd[y, x-1]])

                    #print(center_val, neighbors)
                    if center_val > max(neighbors):
                        extrema.append(((y-ps, x-ps), orientation, scale_idx, scale_size, "max"))
                    else:
                        if center_val < min(neighbors):
                            extrema.append(((y-ps, x-ps), orientation, scale_idx, scale_size, "min"))
    return extrema

def is_low_contrast(patch):
    #return np.var(patch) < 50
    #return np.var(patch) < 0.2
    return np.max(patch) - np.min(patch) < 0.1

def draw_keypoints(im, keypoints):
    height, width = im.shape
    im = np.copy(im)
    im = im[:,:,np.newaxis]
    im = np.repeat(im, 3, axis=2)

    print(im.shape)
    for (y, x), orientation, scale_idx, scale_size, kp_type in keypoints:
        if random.random() > 0.0:
            radius = int(scale_size)
            rr, cc = draw.circle_perimeter(y, x, radius, shape=im.shape)
            im[rr, cc, 0] = 1.0
            im[rr, cc, 1:] = 0

            orientation = quantize(orientation, [-135, -90, -45, 0, 45, 90, 135, 180])
            #quantization_steps = [-135, -90, -45, 0, 45, 90, 135, 180]
            #step = bisect.bisect(quantization_steps, orientation)
            #print(step)
            #orientation = quantization_steps[step]
            #print(orientation)

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
            im[rr, cc, 0] = 1.0
            im[rr, cc, 1:] = 0

    return im

def plot_images(images, titles, no_axis=False):
    fig = plt.figure()
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(1,len(images),i+1)
        ax.set_title(title)
        if no_axis:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        #plt.imshow(image, cmap=cm.Greys_r)
        plt.imshow(image)
    plt.show()

def quantize(val, to_values):
    best_match = None
    best_match_diff = None
    for other_val in to_values:
        diff = abs(other_val - val)
        if best_match is None or diff < best_match_diff:
            best_match = other_val
            best_match_diff = diff
    return best_match

if __name__ == "__main__":
    main()
