"""Generate Eigenfaces from a set of training images."""
from __future__ import division, print_function
from scipy import ndimage
import numpy as np
import random
import util
from sklearn.decomposition import PCA
import os
from scipy import misc
np.random.seed(42)
random.seed(42)

NB_COMPONENTS = 8
SCALE = 32

def main():
    """Load example faces, extract principal components, create Eigenfaces from
    PCs, plot results."""
    # load images, resize them, convert to 1D-vectors
    images_filepaths = get_images_in_directory("images/faces/")
    images = [ndimage.imread(fp, flatten=True) for fp in images_filepaths]
    images = [misc.imresize(image, (SCALE, SCALE)) for image in images]
    images_vecs = np.array([image.flatten() for image in images])

    # train PCA, embed image vectors, reverse the embedding (lossy)
    pca = PCA(NB_COMPONENTS)
    images_vecs_transformed = pca.fit_transform(images_vecs)
    images_vecs_reversed = pca.inverse_transform(images_vecs_transformed)

    # Extract Eigenfaces. The Eigenfaces are the principal components.
    pcs = pca.components_.reshape((NB_COMPONENTS, SCALE, SCALE))

    # plot (First example image, recovered first example image, first 8 PCs)
    plots_imgs = [images[0], images_vecs_reversed[0].reshape((SCALE, SCALE))]
    plots_titles = ["Image 0", "Image 0\n(reconstructed)"]
    for i in range(NB_COMPONENTS):
        plots_imgs.append(pcs[i])
        plots_titles.append("Eigenface %d" % (i))

    util.plot_images_grayscale(plots_imgs, plots_titles)

def get_images_in_directory(directory, ext="jpg"):
    """Read the filepaths to all images in a directory.
    Args:
        directory Filepath to the directory.
        ext File extension of the images.
    Returns:
        List of filepaths
    """
    filepaths = []
    for fname in os.listdir(directory):
        if fname.endswith(ext) and os.path.isfile(os.path.join(directory, fname)):
            filepaths.append(os.path.join(directory, fname))
    return filepaths

if __name__ == "__main__":
    main()
