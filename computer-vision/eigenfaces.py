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

    # ------------
    # Custom Implementation of PCA
    # ------------
    pcs, images_vecs_transformed, images_vecs_reversed = custom_pca(images_vecs, NB_COMPONENTS)
    pcs = pcs.reshape((NB_COMPONENTS, SCALE, SCALE))

    # plot (First example image, recovered first example image, first 8 PCs)
    plots_imgs = [images[0], images_vecs_reversed[0].reshape((SCALE, SCALE))]
    plots_titles = ["Image 0", "Image 0\n(reconstructed)"]
    for i in range(NB_COMPONENTS):
        plots_imgs.append(pcs[i])
        plots_titles.append("Eigenface %d" % (i))

    util.plot_images_grayscale(plots_imgs, plots_titles)


    # ------------
    # Using the PCA implementation from scikit
    # ------------
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

def custom_pca(images_vecs, nb_components):
    """Custom implementation of PCA for images.
    Args:
        images_vecs    The images to transform.
        nb_components  Number of principal components.
    Returns:
        Principal Components of shape (NB_COMPONENTS, height*width),
        Transformed images of shape (nb_images, NB_COMPONENTS),
        Reverse transformed/reconstructed images of shape (nb_images, height*width)
    """
    imgs = np.copy(images_vecs)
    imgs = np.transpose(imgs)
    mean = np.average(imgs, axis=1)
    mean = mean[:, np.newaxis]
    A = imgs - np.tile(mean, (1, imgs.shape[1]))
    # Compute eigenvectors of A^TA instead of AA^T (covariance matrix) as
    # that is faster.
    L = np.dot(np.transpose(A), A) # A^TA
    eigenvalues, eigenvectors = np.linalg.eig(L)

    # recover eigenvectors of AA^T (covariance matrix)
    U = np.dot(A, eigenvectors)

    # reduce to requested number of eigenvectors
    U = np.transpose(U)
    nb_components = min(len(eigenvectors), nb_components)
    U = U[0:nb_components, :]

    # project faces to face space
    imgs_transformed = np.dot(U, A)
    imgs_transformed = np.transpose(imgs_transformed)

    # reconstruct faces
    imgs_reversed = np.dot(np.transpose(U), np.transpose(imgs_transformed))
    imgs_reversed = np.transpose(imgs_reversed)

    return U, imgs_transformed, imgs_reversed

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
