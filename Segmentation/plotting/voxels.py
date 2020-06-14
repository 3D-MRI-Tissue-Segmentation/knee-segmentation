import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import tensorflow as tf

def plot_volume(volume, show=False):
    if len(volume.shape) == 3:
        voxel = volume[:, :, :] > 0
    else:
        voxel = volume[:, :, :, 0] > 0
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    print("Beginning voxel representation")
    print("...please wait, it's going to take a while...")
    ax.voxels(voxel) #, facecolors=volume, linewidth=0.5)
    print("done")

    if show:
        plt.show()
    else:
        return fig

def plot_slice(vol_slice, show=False):
    fig = plt.figure()
    plt.imshow(vol_slice, cmap="gray")
    if show:
        plt.show()
    else:
        return fig

def plot_to_image(figure):
    """ code from https://www.tensorflow.org/tensorboard/image_summaries """
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
