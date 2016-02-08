from __future__ import division, print_function
from scipy import signal, ndimage
from scipy.ndimage import filters as filters
import numpy as np
import random
from skimage import data
from skimage import filters as skifilters
from skimage import util as skiutil
from skimage import feature
import util
from sklearn.decomposition import PCA
import os
from scipy import misc
np.random.seed(42)
random.seed(42)

NB_COMPONENTS = 8
SCALE = 32

def main():
    images_filepaths = get_files_in_directory("images/faces/")
    images = [ndimage.imread(fp, flatten=True) for fp in images_filepaths]
    images = [misc.imresize(image, (SCALE, SCALE)) for image in images]
    images_vecs = np.array([image.flatten() for image in images])
    pca = PCA(NB_COMPONENTS)
    images_vecs_transformed = pca.fit_transform(images_vecs)
    images_vecs_reversed = pca.inverse_transform(images_vecs_transformed)
    pcs = pca.components_.reshape((NB_COMPONENTS, SCALE, SCALE))

    plots_imgs = [images[0], images_vecs_reversed[0].reshape((SCALE, SCALE))]
    plots_titles = ["Image 0", "Image 0\n(reconstructed)"]
    for i in range(NB_COMPONENTS):
        plots_imgs.append(pcs[i])
        plots_titles.append("Eigenface %d" % (i))

    util.plot_images_grayscale(plots_imgs, plots_titles)

def get_files_in_directory(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

if __name__ == "__main__":
    main()
