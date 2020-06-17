import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import tensorflow as tf

def plot_volume(volume, show=False):
    voxel = volume[:, :, :, 0] > 0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel, facecolors=volume, linewidth=0.5)
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

def plot_through_slices(batch_idx, x_crop, y_crop, mean_pred, writer):
    for i in range(160):
        x_slice = tf.slice(x_crop, [batch_idx, i, 0, 0, 0], [1, 1, -1, -1, -1])
        y_slice = tf.slice(y_crop, [batch_idx, i, 0, 0, 0], [1, 1, -1, -1, -1])
        m_slice = tf.slice(mean_pred, [batch_idx, i, 0, 0, 0], [1, 1, -1, -1, -1])
        m_slice = tf.math.round(m_slice)

        img = tf.concat((x_slice, y_slice, m_slice), axis=-2)
        img = tf.reshape(img, (img.shape[1:]))
        with writer.as_default():
            tf.summary.image(f"Whole Validation - All Slices", img, step=i)
