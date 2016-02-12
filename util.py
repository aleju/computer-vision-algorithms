"""Utility functions for the CV algorithms."""
import matplotlib.pyplot as plt

def plot_images_grayscale(images, titles, no_axis=False):
    """Plot a list of grayscale images in one window.
    Args:
        images    List of images (2D arrays)
        titles    List of titles for each image
        no_axis   Whether to show x/y axis"""
    fig = plt.figure()
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(1,len(images),i+1)
        ax.set_title(title)
        if no_axis:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.imshow(image, cmap=plt.cm.gray)
    plt.show()

def quantize(val, to_values):
    """Quantizes a value to a set of allowed values.
    Compares to each value among the allowed values and thus is rather slow.
    Example:
        quantize(4.6, [3.5, 4, 4.5, 5, 5.5, 6]) -> 4.5

    Args:
        val    The value to quantize.
        to_values The list of allowed values."""
    best_match = None
    best_match_diff = None
    for other_val in to_values:
        diff = abs(other_val - val)
        if best_match is None or diff < best_match_diff:
            best_match = other_val
            best_match_diff = diff
    return best_match
