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
        if idx+1 == which_volume:
            x, _ = data
            x = np.array(x)

            print("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"\t\tCollected data for volume {idx+1}")
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


def create_collage_input_gif(volume_numbers):
    
    valid_ds = read_tfrecord(tfrecords_dir='gs://oai-challenge-dataset/tfrecords/valid/',
                            batch_size=160,
                            buffer_size=500,
                            augmentation=None,
                            multi_class=True,
                            is_training=False,
                            use_bfloat16=False,
                            use_RGB=False)

    subplot_dimension = int(volume_numbers**0.5)
    fig, axes = plt.subplots(subplot_dimension, subplot_dimension)
    fig.set_facecolor('black')
    gif_frames = [[] for _ in range(160)]
    r, c = 0, 0

    for idx, data in enumerate(valid_ds):
        if idx+1 <= volume_numbers:
            if c == subplot_dimension:
                c = 0
                r = r+1
            x, _ = data
            x = np.array(x)

            print("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"\t\tCollected data for volume {idx+1}")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

            print('Input image data type: {}, shape: {}'.format(type(x), x.shape))
            print('reducing image size')
            x = np.squeeze(x, axis=-1)
            print('Input image data type: {}, shape: {}\n\n'.format(type(x), x.shape))

            for i in range(x.shape[0]):
                print(f"Analysing slice {i+1} of Volume {idx+1}")
                im = axes[r,c].imshow(x[i,:,:], cmap='gray', animated=True, aspect='auto')
                axes[r,c].axis('off')
                gif_frames[i].append(im)

            c = c+1

        else:
            break

    pred_evolution_gif(fig, gif_frames, save_dir='results/input_volume_gif_collage.gif')

# def gif_collage(figures,
#                 gifs_lists,
#                 interval=300,
#                 save_dir='',
#                 save=True,
#                 show=False):

#     print("\n\n\n\n=================")
#     print("checking for ffmpeg...")
#     if not os.path.isfile('./../../../opt/conda/bin/ffmpeg'):
#         print("please 'pip install ffmpeg' to create gif")
#         print("gif not created")
        
#     else:
#         print("ffmpeg found")
#         print("creating gif collage ...\n")
#         gif = ArtistAnimation(fig, frames_list, interval, repeat=True) # create gif

#         if save:
#             plt.tight_layout()
#             plt.gca().set_axis_off()
#             plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
#                                 hspace = 0, wspace = 0)
#             plt.margins(0,0)
#             plt.gca().xaxis.set_major_locator(plt.NullLocator())
#             plt.gca().yaxis.set_major_locator(plt.NullLocator())
#             if save_dir == '':
#                 time = datetime.now().strftime("%Y%m%d-%H%M%S")
#                 save_dir = 'results/gif'+ time + '.gif'

#             plt.rcParams['animation.ffmpeg_path'] = r'//opt//conda//bin//ffmpeg'  # set directory of ffmpeg binary file
#             Writer = animation.writers['ffmpeg']
#             ffmwriter = Writer(fps=1000//interval, metadata=dict(artist='Me'), bitrate=1800) #set the save writer
#             gif.save('results/temp_video.mp4', writer=ffmwriter)

#             codeBASH = f"ffmpeg -i 'results/temp_video.mp4' -loop 0 {save_dir}" #convert mp4 to gif
#             os.system(codeBASH)
#             os.remove("results/temp_video.mp4")

#             plt.close('all')

#         if show:
#             plt.show()
#             plt.close('all')
        
#         print("\n\n=================")
#         print('done\n\n')

if __name__ == '__main__':
    # print('\n\n\n\n\n')
    # for p in sys.path:
    #     print(p)
    # print('\n\n\n\n\n')
    # create_single_input_gif(1, clean=True)
    create_collage_input_gif(16)

    

