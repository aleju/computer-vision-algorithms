# About

Custom implementations of common machine learning algorithms.
Purely for training purposes and not intended for productive use.

# Files

All files can simply be executed with `python filename.py` (i.e. no command line arguments).

**Computer Vision**

* **binary_dilation_erosion.py** - Dilation, Erosion, Opening and Closing for binary images.
* **canny.py** - Canny edge detector.
* **crosscorrelation_convolution.py** - Implementation of crosscorrelation and convolution to apply filters to images.
* **eigenfaces.py** - Apply a PCA (from sklearn) to human faces, plot the principal components (aka eigenfaces).
* **gauss.py** - Gauss filter for smoothing of images.
* **gradients.py** - Calculate x and y derivatives of an image using symmetric, backward and forward gradient techniques.
* **graph_image_segmentation.py** - Split an image to segments using a simple graph based technique with internal and external subgraph differences.
* **harris.py** - Calculate the Harris edge detector score of pixels in an image.
* **histogram_equalization.py** - Normalize the intensity histogram of an image using histogram stretching and cumulative histogram equalization.
* **hough.py** - Find a line in a noisy image using a Hough Transformation.
* **mean_shift_segmentation.py** - Split an image to segments using mean shift clustering.
* **otsu.py** - Binarize an image using Otsu's method.
* **prewitt.py** - Apply a Prewitt filter to an image to calculate the x/y gradients.
* **rank_order.py** - Apply rank order filters to an image for (non-binary) erosion/dilation/closing/opening, median filtering and morphological edge detection.
* **sift.py** - Simplified implementation of the SIFT keypoint locator. Does not contain the descriptor nor sophisticated keypoint filtering (using hessian und principal curvatures). Also does not resize scales with higher sigmas.
* **sobel.py** - Apply a Sobel filter to an image for smoothed gradient calculation.
* **template_matching.py** - Find an example template image in a larger image.

**Classifiers**

* **gaussian_mixture_1d_em.py** - Train a mixture model of 1d gaussians using the EM algorithm.

# Requirements

Python 2.7, scipy, numpy, sklearn, scikit-image
