""" Necessary functions for visualizing, eg. making gifs, plots, or saving
    numpy volumes for plotly graph 
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Segmentation.train.reshape import get_mid_slice, get_mid_vol

# ## train model loop
def visualise_sample(x,
                     y,
                     pred,
                     num_to_visualise,
                     slice_writer,
                     vol_writer,
                     use_2d,
                     epoch,
                     multi_class,
                     predict_slice,
                     is_training):
                     
    img = get_mid_slice(x.values[0], y.values[0], pred.values[0], multi_class)
    session_type = "Train" if is_training else "Validation"
    with slice_writer.as_default():
        tf.summary.image(f"{session_type} - Slice", img, step=epoch)
    if epoch % visual_save_freq == 0:
        if not predict_slice:
            img = get_mid_vol(y.values[0], pred.values[0], multi_class, check_empty=True)
            if img is None:
                num_to_visualise += 1
            else:
                with vol_writer.as_default():
                    tf.summary.image(f"{session_type} - Volume", img, step=epoch)
    return num_to_visualise


## VNet: vnet_train from og dev_rl

def plot_imgs(images_arr, img_plt_names, plt_supertitle, save_fig_name, color_map="gray"):
    """ Plot images via imshow with titles.
        Input array images_arr shape determines subplots.
        Input array of images should have a corresponding array or list of plott names. """
    rows = np.shape(images_arr)[0]
    cols = np.shape(images_arr)[1]

    f, axes = plt.subplots(rows, cols)

    for i in rows:
        for j in cols:
            axes[i, j].imshow(images_arr[i, j], cmap=color_map)
            axes[i, j].set_title(img_plt_names[i * cols + j], cmap=color_map)

    for a in axes:
        for ax in a:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    f.tight_layout(rect=[0, 0.01, 1, 0.93])
    f.suptitle(plt_supertitle)
    plt.savefig(save_fig_name)
    plt.close('all')




