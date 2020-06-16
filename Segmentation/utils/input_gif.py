#to run as 'py -m Segmentation.utils.input_gif' for import reasons

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import ArtistAnimation

from datetime import datetime
import os
import sys

from Segmentation.utils.data_loader import read_tfrecord
from Segmentation.utils.evaluation_utils import pred_evolution_gif

def create_single_input_gif(which_volume,
                            clean=False):

    valid_ds = read_tfrecord(tfrecords_dir='gs://oai-challenge-dataset/tfrecords/valid/',
                            batch_size=160,
                            buffer_size=500,
                            augmentation=None,
                            multi_class=True,
                            is_training=False,
                            use_bfloat16=False,
                            use_RGB=False)

    for idx, data in enumerate(valid_ds):
        if idx == which_volume:
            x, _ = data
            x = np.array(x)

            print("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"\t\tCollected data for volume {idx}")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

            print('Input image data type: {}, shape: {}'.format(type(x), x.shape))
            print('reducing image size')
            x = np.squeeze(x, axis=-1)
            print('Input image data type: {}, shape: {}\n\n'.format(type(x), x.shape))

            fig, ax = plt.subplots()

            gif_frames = []
            for i in range(x.shape[0]):
                print(f"Analysing slice {i+1}")
                im = ax.imshow(x[i,:,:], cmap='gray', animated=True, aspect='auto')
                if not clean:
                    text = ax.text(0.5,1.05,f'Slice {i+1}', 
                                size=plt.rcParams["axes.titlesize"],
                                ha="center", transform=ax.transAxes)
                    gif_frames.append([im, text])
                else:
                    ax.axis('off')
                    gif_frames.append([im])

            break

    pred_evolution_gif(fig, gif_frames, save_dir='results/input_volume_gif2.gif')

# def create_collage_input_gif(which_volume,
#                              clean=False):

if __name__ == '__main__':
    # print('\n\n\n\n\n')
    # for p in sys.path:
    #     print(p)
    # print('\n\n\n\n\n')
    create_single_input_gif(1, clean=True)

    

