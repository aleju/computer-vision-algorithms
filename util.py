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
