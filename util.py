import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_images_grayscale(images, titles, no_axis=False):
    fig = plt.figure()
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(1,len(images),i+1)
        ax.set_title(title)
        if no_axis:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        #plt.imshow(image, cmap=cm.Greys_r)
        plt.imshow(image, cmap=plt.cm.gray)
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
