import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import ArtistAnimation

import glob
from google.cloud import storage
from pathlib import Path
import os
import datetime

from Segmentation.plotting.voxels import plot_volume
# from Segmentation.utils.data_loader import read_tfrecord_2d
from Segmentation.utils.training_utils import visualise_binary, visualise_multi_class
from Segmentation.utils.evaluation_metrics import get_confusion_matrix, plot_confusion_matrix, iou_loss_eval, dice_coef_eval
from Segmentation.utils.metrics import dice_coef, IoU

def get_depth(conc):
    depth = 0
    for batch in conc:
        depth += batch.shape[0]
    return depth


def get_bucket_weights(bucket_name, logdir, tpu_name, visual_file, weights_dir):
    """ Load the checkpoints in the specified log directory """

    ######################
    """ Add the visualisation code here """
    print("+========================================================")
    print('bucket_name',bucket_name)
    print("\n\nThe directories are:")
    print('weights_dir == "checkpoint"',weights_dir == "checkpoint")
    print('weights_dir',weights_dir)
    ######################

    session_name = weights_dir.split('/')[3]
    session_name = os.path.join(session_name, tpu_name, visual_file)
    # Pietro's: session_name = os.path.join(weights_dir, tpu_name, visual_file)

    # Get names within folder in gcloud
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)
    session_content = []
    print('session_name',session_name)
    for blob in blobs:
        if session_name in blob.name:
            session_content.append(blob.name)

    session_weights = []
    for item in session_content:
        if ('_weights' in item) and ('.ckpt.index' in item):
            session_weights.append(item)

    ######################
    for s in session_weights:
        print(s) #print all the checkpoint directories
    print("--")
    ######################

    return session_weights

# def plot_and_eval_3D(model,
                    #  logdir,
                    #  visual_file,
                    #  tpu_name,
                    #  bucket_name,
                    #  weights_dir,
                    #  multi_class,
                    #  save_freq,
                    #  dataset,
                    #  model_args):

    # """ plotly: Generates a numpy volume for every #save_freq number of weights
    #     and saves it in local results/pred/*visual_file* and results/y/*visual_file*

    #     Once numpy's are generated, run the following in console to get an embeddable html file:
    #         python3 Visualization/plotly_3d_voxels/run_plotly.py -dir_l FOLDER_TO_Y_SAMPLES
    #          -dir_r FOLDER_TO_PREDICTIONS


    # """

    # session_weights = get_bucket_weights(bucket_name, logdir, tpu_name, visual_file, weights_dir)

    # # Only use part of dataset
    # idx_vol= 0 # how many numpies have been save
    # target = 160
    
    # for i, chkpt in enumerate(session_weights):
        
    #     should_save_np = np.mod((i+1), save_freq) == 0
        
    #     ######################
    #     # print('should_save_np',should_save_np)
    #     # print('checkpoint enum i',i)
    #     # print('save_freq set to ',save_freq)
    #     ######################

    #     if not should_save_np:      # skip this checkpoint weight
    #         print("Skipping weight", chkpt)
    #         continue


    #     name = chkpt.split('/')[-1]
    #     name = name.split('.inde')[0]
    #     trained_model = model(*model_args)
    #     trained_model.load_weights('gs://' + os.path.join(bucket_name,
    #                                                     'checkpoints',
    #                                                     tpu_name,
    #                                                     visual_file,
    #                                                     name)).expect_partial()



    #     # sample_x = []    # x for current 160,288,288 vol
    #     sample_pred = []  # prediction for current 160,288,288 vol
    #     sample_y = []    # y for current 160,288,288 vol

    #     which_volume = 2
    #     for idx, ds in enumerate(dataset):

    #         ######################
    #         print('Current chkpt name',name)
    #         print(f"the index is {idx}")
    #         ######################


    #         x, y = ds
    #         batch_size = x.shape[0]

    #         if batch_size == 160:
    #             if not (int(idx) == int(which_volume)):
    #                 continue

    #         x = np.array(x)
    #         y = np.array(y)
        
    #         pred = trained_model.predict(x)

    #         ######################
    #         # print("Current batch size set to {}. Target depth is {}".format(batch_size, target))
    #         # print('Input image data type: {}, shape: {}'.format(type(x), x.shape))
    #         # print('Ground truth data type: {}, shape: {}'.format(type(y), y.shape))
    #         # print('Prediction data type: {}, shape: {}'.format(type(pred), pred.shape))
    #         # print("=================")
    #         ######################

    #         if (get_depth(sample_pred) + batch_size) < target:  # check if next batch will fit in volume (160)
    #             sample_pred.append(pred)
    #             del pred
    #             sample_y.append(y)
    #             del y
    #         else:
    #             remaining = target - get_depth(sample_pred)
    #             sample_pred.append(pred[:remaining])
    #             sample_y.append(y[:remaining])
    #             pred_vol = np.concatenate(sample_pred)
    #             del sample_pred
    #             y_vol = np.concatenate(sample_y)
    #             del sample_y
    #             sample_pred = [pred[remaining:]]
    #             sample_y = [y[remaining:]]

    #             del pred
    #             del y

    #             ######################
    #             # print("===============")
    #             # print("pred done")
    #             # print(pred_vol.shape)
    #             # print(y_vol.shape)
    #             # print("===============")
    #             # print('multi_class', multi_class)
    #             ######################

    #             if multi_class:  # or np.shape(pred_vol)[-1] not
    #                 pred_vol = np.argmax(pred_vol, axis=-1)
    #                 y_vol = np.argmax(y_vol, axis=-1)

    #                 ######################
    #                 # print('np.shape(pred_vol)', np.shape(pred_vol))
    #                 # print('np.shape(y_vol)',np.shape(y_vol))
    #                 ######################

    #             # Save volume as numpy file for plotlyyy
    #             fig_dir = "results"
    #             name_pred_npy = os.path.join(fig_dir, "pred", (visual_file + "_" + name))
    #             name_y_npy = os.path.join(fig_dir, "ground_truth", (visual_file + "_" + str(which_volume).zfill(3)))
                
    #             ######################
    #             # print("npy save pred as ", name_pred_npy)
    #             # print("npy save y as ", name_y_npy)
    #             # print("Currently on vol ", idx_vol)
    #             ######################


    #             # Get middle xx slices cuz 288x288x160 too big
    #             roi = int(80 / 2)
    #             d1,d2,d3 = np.shape(pred_vol)[0:3]
    #             d1, d2, d3 = int(np.floor(d1/2)), int(np.floor(d2/2)), int(np.floor(d3/2))
    #             pred_vol = pred_vol[(d1-roi):(d1+roi),(d2-roi):(d2+roi), (d3-roi):(d3+roi)]
    #             d1,d2,d3 = np.shape(y_vol)[0:3]
    #             d1, d2, d3 = int(np.floor(d1/2)), int(np.floor(d2/2)), int(np.floor(d3/2))
    #             y_vol = y_vol[(d1-roi):(d1+roi),(d2-roi):(d2+roi), (d3-roi):(d3+roi)]

    #             ######################
    #             print('y_vol.shape', np.shape(y_vol))
    #             ######################

    #             np.save(name_pred_npy,pred_vol)
    #             np.save(name_y_npy,y_vol)
    #             idx_vol += 1
    #             ######################
    #             print("Total voxels saved, pred:", np.sum(pred_vol), "y:", np.sum(y_vol))
    #             ######################
    #             del pred_vol
    #             del y_vol

    #             break

# def epoch_gif(model,
            #   logdir,
            #   tfrecords_dir,
            #   aug_strategy,
            #   visual_file,
            #   tpu_name,
            #   bucket_name,
            #   weights_dir,
            #   multi_class,
            #   model_args,
            #   which_slice,
            #   which_volume=1,
            #   epoch_limit=1000,
            #   gif_dir='',
            #   gif_cmap='gray',
            #   clean=False):

    # #load the database
    # valid_ds = read_tfrecord_2d(tfrecords_dir=tfrecords_dir, #'gs://oai-challenge-dataset/tfrecords/valid/',
    #                         batch_size=160,
    #                         buffer_size=500,
    #                         augmentation=aug_strategy,
    #                         multi_class=multi_class,
    #                         is_training=False,
    #                         use_bfloat16=False,
    #                         use_RGB=False)

    # # load the checkpoints in the specified log directory
    # session_weights = get_bucket_weights(bucket_name, logdir, tpu_name, visual_file, weights_dir)

    # #figure for gif
    # fig, ax = plt.subplots()
    # images_gif = []

    # for chkpt in session_weights:
    #     name = chkpt.split('/')[-1]
    #     name = name.split('.inde')[0]

    #     if int(name.split('.')[1]) <= epoch_limit:

    #         print("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #         print(f"\t\tLoading weights from {name.split('.')[1]} epoch")
    #         print(f"\t\t  {name}")
    #         print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    #         trained_model = model(*model_args)
    #         trained_model.load_weights('gs://' + os.path.join(bucket_name,
    #                                                           weights_dir,
    #                                                           tpu_name,
    #                                                           visual_file,
    #                                                           name)).expect_partial()

    #         for idx, ds in enumerate(valid_ds):

    #             if idx+1 == which_volume:
    #                 x, _ = ds
    #                 x_slice = np.expand_dims(x[which_slice-1], axis=0)
    #                 print('Input image data type: {}, shape: {}\n'.format(type(x_slice), x_slice.shape))

    #                 print('predicting slice {}'.format(which_slice))
    #                 predicted_slice = trained_model.predict(x_slice)
    #                 if multi_class:
    #                     predicted_slice = np.argmax(predicted_slice, axis=-1)
    #                 else:
    #                     predicted_slice = np.squeeze(predicted_slice, axis=-1)

    #                 print('slice predicted\n')

    #                 print("adding prediction to the queue")
    #                 im = ax.imshow(predicted_slice[0], cmap=gif_cmap, animated=True)
    #                 if not clean:
    #                     text = ax.text(0.5,1.05,f"Epoch {int(name.split('.')[1])}", 
    #                                 size=plt.rcParams["axes.titlesize"],
    #                                 ha="center", transform=ax.transAxes)
    #                     images_gif.append([im, text])
    #                 else:
    #                     ax.axis('off')
    #                     images_gif.append([im])
    #                 print("prediction added\n")

    #                 break

    #     else:
    #         break

    # pred_evolution_gif(fig, images_gif, save_dir=gif_dir, save=True, no_margins=clean)

# def volume_gif(model,
            #    logdir,
            #    tfrecords_dir,
            #    aug_strategy,
            #    visual_file,
            #    tpu_name,
            #    bucket_name,
            #    weights_dir,
            #    multi_class,
            #    model_args,
            #    which_epoch,
            #    which_volume=1,
            #    gif_dir='',
            #    gif_cmap='gray',
            #    clean=False):

    # #load the database
    # valid_ds = read_tfrecord_2d(tfrecords_dir=tfrecords_dir, #'gs://oai-challenge-dataset/tfrecords/valid/',
    #                         batch_size=160,
    #                         buffer_size=500,
    #                         augmentation=aug_strategy,
    #                         multi_class=multi_class,
    #                         is_training=False,
    #                         use_bfloat16=False,
    #                         use_RGB=False)

    # # load the checkpoints in the specified log directory
    # session_weights = get_bucket_weights(bucket_name, logdir, tpu_name, visual_file, weights_dir)

    # #figure for gif
    # fig, ax = plt.subplots()
    # images_gif = []

    # for chkpt in session_weights:
    #     name = chkpt.split('/')[-1]
    #     name = name.split('.inde')[0]

    #     if int(name.split('.')[1]) == which_epoch:

    #         print("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #         print(f"\t\tLoading weights from {name.split('.')[1]} epoch")
    #         print(f"\t\t  {name}")
    #         print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    #         trained_model = model(*model_args)
    #         trained_model.load_weights('gs://' + os.path.join(bucket_name,
    #                                                           weights_dir,
    #                                                           tpu_name,
    #                                                           visual_file,
    #                                                           name)).expect_partial()

    #         for idx, ds in enumerate(valid_ds):

    #             if idx+1 == which_volume:
    #                 x, _ = ds
    #                 x = np.array(x)
    #                 print('Input image data type: {}, shape: {}\n'.format(type(x), x.shape))

    #                 print('predicting volume {}'.format(which_volume))
    #                 pred_vol = trained_model.predict(x)
    #                 if multi_class:
    #                     pred_vol = np.argmax(pred_vol, axis=-1)
    #                 else:
    #                     pred_vol = np.squeeze(pred_vol, axis=-1)
    #                 print('volume predicted\n')

    #                 for i in range(x.shape[0]):
    #                     print(f"Analysing slice {i+1}")
    #                     im = ax.imshow(pred_vol[i,:,:], cmap='gray', animated=True, aspect='auto')
    #                     if not clean:
    #                         text = ax.text(0.5,1.05,f'Slice {i+1}', 
    #                                     size=plt.rcParams["axes.titlesize"],
    #                                     ha="center", transform=ax.transAxes)
    #                         images_gif.append([im, text])
    #                     else:
    #                         ax.axis('off')
    #                         images_gif.append([im])

    #                 break
            
    #         break

    # pred_evolution_gif(fig, images_gif, save_dir=gif_dir, save=True, no_margins=clean)

# def volume_comparison_gif(model,
                        #   logdir,
                        #   tfrecords_dir,
                        #   visual_file,
                        #   tpu_name,
                        #   bucket_name,
                        #   weights_dir,
                        #   multi_class,
                        #   model_args,
                        #   which_epoch,
                        #   which_volume=1,
                        #   gif_dir='',
                        #   gif_cmap='gray',
                        #   clean=False):

    # #load the database
    # valid_ds = read_tfrecord_2d(tfrecords_dir=tfrecords_dir, #'gs://oai-challenge-dataset/tfrecords/valid/',
    #                         batch_size=160,
    #                         buffer_size=500,
    #                         augmentation=None,
    #                         multi_class=multi_class,
    #                         is_training=False,
    #                         use_bfloat16=False,
    #                         use_RGB=False)

    # # load the checkpoints in the specified log directory
    # session_weights = get_bucket_weights(bucket_name, logdir, tpu_name, visual_file, weights_dir)

    # #figure for gif
    # fig, axes = plt.subplots(1, 3)
    # images_gif = []

    # for chkpt in session_weights:
    #     name = chkpt.split('/')[-1]
    #     name = name.split('.inde')[0]

    #     if int(name.split('.')[1]) == which_epoch:

    #         print("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #         print(f"\t\tLoading weights from {name.split('.')[1]} epoch")
    #         print(f"\t\t  {name}")
    #         print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    #         trained_model = model(*model_args)
    #         trained_model.load_weights('gs://' + os.path.join(bucket_name,
    #                                                           weights_dir,
    #                                                           tpu_name,
    #                                                           visual_file,
    #                                                           name)).expect_partial()

    #         for idx, ds in enumerate(valid_ds):

    #             if idx+1 == which_volume:
    #                 x, y = ds
    #                 x = np.array(x)
    #                 x = np.squeeze(x, axis=-1)

    #                 print('predicting volume {}'.format(which_volume))
    #                 pred_vol = trained_model.predict(x)
    #                 if multi_class:
    #                     pred_vol = np.argmax(pred_vol, axis=-1)
    #                     y = np.argmax(y, axis=-1)
    #                 print('volume predicted\n')

    #                 print('input image data type: {}, shape: {}'.format(type(x), x.shape))
    #                 print('label image data type: {}, shape: {}'.format(type(y), y.shape))
    #                 print('prediction image data type: {}, shape: {}\n'.format(type(pred), pred.shape))

    #                 for i in range(x.shape[0]):
    #                     print(f"Analysing slice {i+1}")
    #                     x_im = axes[0].imshow(x[i,:,:], cmap='gray', animated=True, aspect='auto')
    #                     y_im = axes[1].imshow(y[i,:,:], cmap='gray', animated=True, aspect='auto')
    #                     pred_im = axes[2].imshow(pred_vol[i,:,:], cmap='gray', animated=True, aspect='auto')
    #                     if not clean:
    #                         text = ax.text(0.5,1.05,f'Slice {i+1}', 
    #                                     size=plt.rcParams["axes.titlesize"],
    #                                     ha="center", transform=ax.transAxes)
    #                         images_gif.append([im, text])
    #                     else:
    #                         ax.axis('off')
    #                         images_gif.append([im])

    #                 break
            
    #         break

    # pred_evolution_gif(fig, images_gif, save_dir=gif_dir, save=True, no_margins=False)


def pred_evolution_gif(fig,
                       frames_list,
                       interval=300,
                       save_dir='',
                       save=True,
                       no_margins=True,
                       show=False):

    print("\n\n\n\n=================")
    print("checking for ffmpeg...")
    if not os.path.isfile('./../../../opt/conda/bin/ffmpeg'):
        print("please 'pip install ffmpeg' to create gif")
        print("gif not created")
        
    else:
        print("ffmpeg found")
        print("creating the gif ...\n")
        gif = ArtistAnimation(fig, frames_list, interval, repeat=True) # create gif

        if save:
            if no_margins:
                plt.tight_layout()
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                                    hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

            if save_dir == '':
                time = datetime.now().strftime("%Y%m%d-%H%M%S")
                save_dir = 'results/gif'+ time + '.gif'

            plt.rcParams['animation.ffmpeg_path'] = r'//opt//conda//bin//ffmpeg'  # set directory of ffmpeg binary file
            Writer = animation.writers['ffmpeg']
            ffmwriter = Writer(fps=1000//interval, metadata=dict(artist='Me'), bitrate=1800) #set the save writer
            gif.save('results/temp_video.mp4', writer=ffmwriter)

            codeBASH = f"ffmpeg -i 'results/temp_video.mp4' -loop 0 {save_dir}" #convert mp4 to gif
            os.system(codeBASH)
            os.remove("results/temp_video.mp4")

            plt.close('all')

        if show:
            plt.show()
            plt.close('all')
        
        print("\n\n=================")
        print('done\n\n')

# def take_slice(model,
            #    logdir,
            #    tfrecords_dir,
            #    aug_strategy,
            #    visual_file,
            #    tpu_name,
            #    bucket_name,
            #    weights_dir,
            #    multi_as_binary,
            #    multi_class,
            #    model_args,
            #    which_epoch,
            #    which_slice,
            #    which_volume=1,
            #    save_dir='',
            #    cmap='gray',
            #    clean=False):

    # #load the database
    # valid_ds = read_tfrecord_2d(tfrecords_dir=tfrecords_dir, #'gs://oai-challenge-dataset/tfrecords/valid/',
    #                         batch_size=160,
    #                         buffer_size=500,
    #                         augmentation=aug_strategy,
    #                         multi_class=multi_class,
    #                         is_training=False,
    #                         use_bfloat16=False,
    #                         use_RGB=False)

    # # load the checkpoints in the specified log directory
    # session_weights = get_bucket_weights(bucket_name, logdir, tpu_name, visual_file, weights_dir)

    # #figure for gif
    # fig, axes = plt.subplots(1, 3)
    # images_gif = []

    # for chkpt in session_weights:
    #     name = chkpt.split('/')[-1]
    #     name = name.split('.inde')[0]

    #     if int(name.split('.')[1]) == which_epoch:

    #         print("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #         print(f"\t\tLoading weights from {name.split('.')[1]} epoch")
    #         print(f"\t\t  {name}")
    #         print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    #         trained_model = model(*model_args)
    #         trained_model.load_weights('gs://' + os.path.join(bucket_name,
    #                                                           weights_dir,
    #                                                           tpu_name,
    #                                                           visual_file,
    #                                                           name)).expect_partial()

    #         for idx, ds in enumerate(valid_ds):

    #             if idx+1 == which_volume:
    #                 x, y = ds
    #                 x_slice = np.expand_dims(x[which_slice-1], axis=0)
    #                 y_slice = y[which_slice-1]

    #                 print('predicting slice {}'.format(which_slice))
    #                 pred_slice = trained_model.predict(x_slice)
    #                 print('prediction image data type: {}, shape: {}\n'.format(type(pred_slice), pred_slice.shape))
    #                 if multi_class:
    #                     pred_slice = np.argmax(pred_slice, axis=-1)
    #                     y_slice = np.argmax(y_slice, axis=-1)
    #                     if multi_as_binary:
    #                         pred_slice[pred_slice>0] = 1
    #                         y_slice[y_slice>0] = 1
    #                 else:
    #                     pred_slice = np.squeeze(pred_slice, axis=-1)
    #                     y_slice = np.squeeze(y_slice, axis=-1)
    #                 print('slice predicted\n')

    #                 print('input image data type: {}, shape: {}'.format(type(x), x.shape))
    #                 print('label image data type: {}, shape: {}'.format(type(y), y.shape))
    #                 print('prediction image data type: {}, shape: {}\n'.format(type(pred_slice), pred_slice.shape))

    #                 print("Creating input image")
    #                 x_s = np.squeeze(x[which_slice-1], axis=-1)
    #                 fig_x = plt.figure()
    #                 ax_x = fig_x.add_subplot(1, 1, 1)
    #                 ax_x.imshow(x_s, cmap='gray')
                    
    #                 print("Creating label image")
    #                 fig_y = plt.figure()
    #                 ax_y = fig_y.add_subplot(1, 1, 1)
    #                 ax_y.imshow(y_slice, cmap='gray')
                    
    #                 print("Creating prediction image")
    #                 fig_pred = plt.figure()
    #                 ax_pred = fig_pred.add_subplot(1, 1, 1)
    #                 ax_pred.imshow(pred_slice[0], cmap='gray')

    #                 #Removing outside frame
    #                 if clean:
    #                     ax_x.axis('off')
    #                     ax_y.axis('off')
    #                     ax_pred.axis('off')
                        

    #                 print("Saving images")
    #                 save_dir_x = save_dir + '_x.png'
    #                 save_dir_y = save_dir + '_y.png'
    #                 save_dir_pred = save_dir + '_pred.png'
    #                 fig_x.savefig(save_dir_x)
    #                 fig_y.savefig(save_dir_y)
    #                 fig_pred.savefig(save_dir_pred)

    #                 break
            
    #         break

# def confusion_matrix(trained_model,
                    #  weights_dir,
                    #  fig_dir,
                    #  dataset,
                    #  validation_steps,
                    #  multi_class,
                    #  model_architecture,
                    #  callbacks,
                    #  num_classes=7
                    #  ):

    # trained_model.load_weights(weights_dir).expect_partial()
    # trained_model.evaluate(dataset, steps=validation_steps, callbacks=callbacks)


    # f = weights_dir.split('/')[-1]
    # # Excluding parenthese before f too
    # if weights_dir.endswith(f):
    #     writer_dir = weights_dir[:-(len(f)+1)]
    # writer_dir = os.path.join(writer_dir, 'eval')
    # # os.makedirs(writer_dir)
    # eval_metric_writer = tf.summary.create_file_writer(writer_dir)


    # if multi_class:
    #     cm = np.zeros((num_classes, num_classes))
    #     classes = ["Background",
    #                "Femoral",
    #                "Medial Tibial",
    #                "Lateral Tibial",
    #                "Patellar",
    #                "Lateral Meniscus",
    #                "Medial Meniscus"]
    # else:
    #     cm = np.zeros((2, 2))
    #     classes = ["Background",
    #                "Cartilage"]

    # for step, (image, label) in enumerate(dataset):
    #     print(step)
    #     pred = trained_model.predict(image)
    #     visualise_multi_class(label, pred)
    #     cm = cm + get_confusion_matrix(label, pred, classes=list(range(0, num_classes)))

    #     if multi_class:
    #         iou = iou_loss_eval(label, pred)
    #         dice = dice_coef_eval(label, pred)
    #     else:
    #         iou = iou_loss(label, pred)
    #         dice = dice_coef(label, pred)

    #     with eval_metric_writer.as_default():
    #         tf.summary.scalar('iou eval validation', iou, step=step)
    #         tf.summary.scalar('dice eval validation', dice, step=step)
        

            

    #     if step > validation_steps - 1:
    #         break

    # fig_file = model_architecture + '_matrix.png'
    # fig_dir = os.path.join(fig_dir, fig_file)
    # plot_confusion_matrix(cm, fig_dir, classes=classes)




########## Confusion Matrix ##########
def initialize_cm(multi_class, num_classes=7):
    if multi_class:
        cm = np.zeros((num_classes, num_classes))
        classes = ["Background",
                   "Femoral",
                   "Medial Tibial",
                   "Lateral Tibial",
                   "Patellar",
                   "Lateral Meniscus",
                   "Medial Meniscus"]
    else:
        cm = np.zeros((2, 2))
        classes = ["Background",
                   "Cartilage"]

    return cm, classes


def update_cm(cm, num_classes=7):
    cm = cm + get_confusion_matrix(label, pred, classes=list(range(0, num_classes)))
    return cm

def save_cm(cm, model_architecture, fig_dir, classes):
    fig_file = model_architecture + '_matrix.png'
    fig_dir = os.path.join(fig_dir, fig_file)
    plot_confusion_matrix(cm, fig_dir, classes=classes)
##########



########## Gif ##########
def initialize_gif():   
    #figure for gif
    fig, axes = plt.subplots(1, 3)
    images_gif = []
    return fig, axes, images_gif

def update_gif_slice(x, y, trained_model,
               aug_strategy,
               multi_as_binary, multi_class,
               which_epoch, which_slice, which_volume=1,
               save_dir='',
               gif_cmap='gray',
               clean=False):


    x_slice = np.expand_dims(x[which_slice-1], axis=0)
    y_slice = y[which_slice-1]

    print('predicting slice {}'.format(which_slice))
    pred_slice = trained_model.predict(x_slice)
    print('prediction image data type: {}, shape: {}\n'.format(type(pred_slice), pred_slice.shape))
    if multi_class:
        pred_slice = np.argmax(pred_slice, axis=-1)
        y_slice = np.argmax(y_slice, axis=-1)
        if multi_as_binary:
            pred_slice[pred_slice>0] = 1
            y_slice[y_slice>0] = 1
    else:
        pred_slice = np.squeeze(pred_slice, axis=-1)
        y_slice = np.squeeze(y_slice, axis=-1)

    ###############
    print('slice predicted\n')
    print('input image data type: {}, shape: {}'.format(type(x), x.shape))
    print('label image data type: {}, shape: {}'.format(type(y), y.shape))
    print('prediction image data type: {}, shape: {}\n'.format(type(pred_slice), pred_slice.shape))
    ###############

    print("Creating input image")
    x_s = np.squeeze(x[which_slice-1], axis=-1)
    fig_x = plt.figure()
    ax_x = fig_x.add_subplot(1, 1, 1)
    ax_x.imshow(x_s, cmap='gray')
    
    print("Creating label image")
    fig_y = plt.figure()
    ax_y = fig_y.add_subplot(1, 1, 1)
    ax_y.imshow(y_slice, cmap='gray')
    
    print("Creating prediction image")
    fig_pred = plt.figure()
    ax_pred = fig_pred.add_subplot(1, 1, 1)
    ax_pred.imshow(pred_slice[0], cmap='gray')

    #Removing outside frame
    if clean:
        ax_x.axis('off')
        ax_y.axis('off')
        ax_pred.axis('off')
        

    print("Saving images")
    save_dir_x = save_dir + '_x.png'
    save_dir_y = save_dir + '_y.png'
    save_dir_pred = save_dir + '_pred.png'
    fig_x.savefig(save_dir_x)
    fig_y.savefig(save_dir_y)
    fig_pred.savefig(save_dir_pred)


def update_volume_comp_gif(x,y, images_gif, trained_model,
                          multi_class,
                          which_epoch,
                          which_volume=1,
                          gif_dir='',
                          gif_cmap='gray',
                          clean=False):

    x = np.array(x)
    x = np.squeeze(x, axis=-1)

    print('predicting volume {}'.format(which_volume))
    pred_vol = trained_model.predict(x)
    if multi_class:
        pred_vol = np.argmax(pred_vol, axis=-1)
        y = np.argmax(y, axis=-1)
    print('volume predicted\n')

    print('input image data type: {}, shape: {}'.format(type(x), x.shape))
    print('label image data type: {}, shape: {}'.format(type(y), y.shape))
    print('prediction image data type: {}, shape: {}\n'.format(type(pred), pred.shape))

    for i in range(x.shape[0]):
        print(f"Analysing slice {i+1}")
        x_im = axes[0].imshow(x[i,:,:], cmap='gray', animated=True, aspect='auto')
        y_im = axes[1].imshow(y[i,:,:], cmap='gray', animated=True, aspect='auto')
        pred_im = axes[2].imshow(pred_vol[i,:,:], cmap='gray', animated=True, aspect='auto')
        if not clean:
            text = ax.text(0.5,1.05,f'Slice {i+1}', 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes)
            images_gif.append([im, text])
        else:
            ax.axis('off')
            images_gif.append([im])

    return images_gif


def update_epoch_gif(x, trained_model, aug_strategy,
              multi_class, which_slice, which_volume=1,
              epoch_limit=1000,
              gif_dir='',
              gif_cmap='gray',
              clean=False):

    images_gif = []

    x_slice = np.expand_dims(x[which_slice-1], axis=0)
    print('Input image data type: {}, shape: {}\n'.format(type(x_slice), x_slice.shape))

    print('predicting slice {}'.format(which_slice))
    predicted_slice = trained_model.predict(x_slice)
    if multi_class:
        predicted_slice = np.argmax(predicted_slice, axis=-1)
    else:
        predicted_slice = np.squeeze(predicted_slice, axis=-1)

    print('slice predicted\n')

    print("adding prediction to the queue")
    im = ax.imshow(predicted_slice[0], cmap=gif_cmap, animated=True)
    if not clean:
        text = ax.text(0.5,1.05,f"Epoch {int(name.split('.')[1])}", 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes)
        images_gif.append([im, text])
    else:
        ax.axis('off')
        images_gif.append([im])
    print("prediction added\n")

    return images_gif
##########



########## Plotly npys ##########
def update_volume_npy(y, pred, target, 
                     sample_pred, sample_y, 
                     visual_file, name, 
                     which_volume, multi_class):
    batch_size = y.shape[0]
    y = np.array(y)
    pred = np.array(pred)


    if (get_depth(sample_pred) + batch_size) < target:  # check if next batch will fit in volume (160)
        sample_pred.append(pred)
        del pred
        sample_y.append(y)
        del y
    else:
        remaining = target - get_depth(sample_pred)
        sample_pred.append(pred[:remaining])
        sample_y.append(y[:remaining])
        pred_vol = np.concatenate(sample_pred)
        del sample_pred
        y_vol = np.concatenate(sample_y)
        del sample_y
        sample_pred = [pred[remaining:]]
        sample_y = [y[remaining:]]

        del pred
        del y

        ######################
        # print("===============")
        # print("pred done")
        # print(pred_vol.shape)
        # print(y_vol.shape)
        # print("===============")
        # print('multi_class', multi_class)
        ######################

        if multi_class:  # or np.shape(pred_vol)[-1] not
            pred_vol = np.argmax(pred_vol, axis=-1)
            y_vol = np.argmax(y_vol, axis=-1)

            ######################
            # print('np.shape(pred_vol)', np.shape(pred_vol))
            # print('np.shape(y_vol)',np.shape(y_vol))
            ######################

        # Save volume as numpy file for plotlyyy
        fig_dir = "results"
        name_pred_npy = os.path.join(fig_dir, "pred", (visual_file + "_" + name))
        name_y_npy = os.path.join(fig_dir, "ground_truth", (visual_file + "_vol" + str(which_volume).zfill(3)))
        
        ######################
        # print("npy save pred as ", name_pred_npy)
        # print("npy save y as ", name_y_npy)
        # print("Currently on vol ", idx_vol)
        ######################


        # Get middle xx slices cuz 288x288x160 too big
        roi = int(80 / 2)
        d1,d2,d3 = np.shape(pred_vol)[0:3]
        d1, d2, d3 = int(np.floor(d1/2)), int(np.floor(d2/2)), int(np.floor(d3/2))
        pred_vol = pred_vol[(d1-roi):(d1+roi),(d2-roi):(d2+roi), (d3-roi):(d3+roi)]
        d1,d2,d3 = np.shape(y_vol)[0:3]
        d1, d2, d3 = int(np.floor(d1/2)), int(np.floor(d2/2)), int(np.floor(d3/2))
        y_vol = y_vol[(d1-roi):(d1+roi),(d2-roi):(d2+roi), (d3-roi):(d3+roi)]

        ######################
        print('y_vol.shape', np.shape(y_vol))
        ######################

        np.save(name_pred_npy,pred_vol)
        np.save(name_y_npy,y_vol)
        ######################
        print("Total voxels saved, pred:", np.sum(pred_vol), "y:", np.sum(y_vol))
        ######################

        sample_pred = []
        sample_y = []
        del pred_vol
        del y_vol

    return sample_pred, sample_y

##########





def eval_loop(dataset, validation_steps, aug_strategy,
                     bucket_name, logdir, tpu_name, visual_file, weights_dir, 
                     fig_dir, 
                     which_volume, which_epoch, which_slice, 
                     multi_as_binary,
                     trained_model, model_architecture, 
                     callbacks,
                     num_classes=7
                     ):

    """ Evaluate model and visualize as needed """

    multi_class = num_classes > 1
    gif_dir=''

    # load the checkpoints in the specified log directory
    session_weights = get_bucket_weights(bucket_name, logdir, tpu_name, visual_file, weights_dir)
    last_epoch = len(session_weights)

    # trained_model.load_weights(weights_dir).expect_partial()
    # trained_model.evaluate(dataset, steps=validation_steps, callbacks=callbacks)

    


    # Callbacks (as in og conf matrix function)
    f = weights_dir.split('/')[-1]
    # Excluding parenthese before f too
    if weights_dir.endswith(f):
        writer_dir = weights_dir[:-(len(f)+1)]
    writer_dir = os.path.join(writer_dir, 'eval')
    eval_metric_writer = tf.summary.create_file_writer(writer_dir)


    # Init visuals
    cm, classes = initialize_cm(multi_class, num_classes)
    fig, axes, images_gif = initialize_gif()
    target = 160 # how many slices in 1 vol
    sample_pred = []  # prediction for current 160,288,288 vol
    sample_y = []    # y for current 160,288,288 vol




    for chkpt in session_weights:
        ### Skip to last chkpt if you only want evaluation

        name = chkpt.split('/')[-1]
        name = name.split('.inde')[0]
        epoch = name.split('.')[1]

        #########################
        print("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"\t\tLoading weights from {epoch} epoch")
        print(f"\t\t  {name}")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        #########################

        trained_model.load_weights('gs://' + os.path.join(bucket_name,
                                                            weights_dir,
                                                            tpu_name,
                                                            visual_file,
                                                            name)).expect_partial()
        if epoch==last_epoch:
            trained_model.evaluate(dataset, steps=validation_steps, callbacks=callbacks)    


        # Initializing volume saving
        sample_pred = []  # prediction for current 160,288,288 vol
        sample_y = []    # y for current 160,288,288 vol

        for step, (x, label) in enumerate(dataset):
            print('step',step)
            pred = trained_model.predict(x)
            

            # Update visuals
            cm = update_cm(cm, num_classes)
            visualise_multi_class(label, pred)

            if step+1 == which_volume:
                update_gif_slice(x, label, trained_model,
                                aug_strategy,
                                multi_as_binary, multi_class,
                                which_epoch, which_slice)

                images_gif = update_volume_comp_gif(x,label, images_gif, trained_model,
                          multi_class,
                          which_epoch,
                          gif_dir=gif_dir)

                images_gif = update_epoch_gif(x, trained_model, aug_strategy,
                                            multi_class, which_slice, 
                                            gif_dir=gif_dir)
            
                sample_pred, sample_y = update_volume_npy(label, pred, target, 
                                                        sample_pred, sample_y, 
                                                        visual_file, name, 
                                                        which_volume, multi_class)
                    


            # if multi_class:
            #     iou = iou_loss_eval(label, pred)
            #     dice = dice_coef_eval(label, pred)
            # else:
            #     iou = iou_loss(label, pred)
            #     dice = dice_coef(label, pred)
            iou = iou_loss_eval(label, pred) if multi_class else iou_loss(label, pred)
            dice = dice_coef_eval(label, pred) if multi_class else dice_coef(label, pred)

            with eval_metric_writer.as_default():
                tf.summary.scalar('iou eval validation', iou, step=step)
                tf.summary.scalar('dice eval validation', dice, step=step)

        # Save visuals
        save_cm(cm, model_architecture, fig_dir, classes)
    pred_evolution_gif(fig, images_gif, save_dir=gif_dir, save=True, no_margins=False)

