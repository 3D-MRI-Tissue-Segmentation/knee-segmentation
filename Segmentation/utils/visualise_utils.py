""" Necessary functions for visualizing, eg. making gifs, plots, or saving
    numpy volumes for plotly graph 
"""
import numpy as np
import matplotlib.pyplot as plt



#### VNet: vnet_train 

def plot_imgs(images_arr, img_plt_names, plt_supertitle, save_fig_name, color_map="gray"):
    """ Plot images via imshow with titles.
        Input array images_arr shape determines subplots.
        Input array of images should have a corresponding array or list of plott names. """
    rows = np.shape(images_arr)[0]
    cols = np.shape(images_arr)[1]

    f, axes = plt.subplots(rows, cols)

    for i in rows:
        for j in cols:
            axes[i, j].imshow(images_arr[i,j], cmap=color_map)
            axes[i, j].set_title(img_plt_names[i*cols+j], cmap=color_map)


    for a in axes:
        for ax in a:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    f.tight_layout(rect=[0, 0.01, 1, 0.93])
    f.suptitle(plt_supertitle)
    plt.savefig(save_fig_name)
    plt.close('all')




