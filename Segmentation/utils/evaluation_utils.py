import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import ArtistAnimation

import glob
from google.cloud import storage
from pathlib import Path
import os
from datetime import datetime

from Segmentation.utils.losses import dice_coef
from Segmentation.plotting.voxels import plot_volume
from Segmentation.utils.data_loader import read_tfrecord
from Segmentation.utils.training_utils import visualise_binary, visualise_multi_class
from Segmentation.utils.evaluation_metrics import get_confusion_matrix, plot_confusion_matrix, iou_loss_eval, dice_coef_eval
from Segmentation.utils.losses import dice_coef, iou_loss

def get_depth(conc):
    depth = 0
    for batch in conc:
        depth += batch.shape[0]
    return depth

def plot_and_eval_3D(model,
                     logdir,
                     visual_file,
                     tpu_name,
                     bucket_name,
                     weights_dir,
                     is_multi_class,
                     save_freq,
                     dataset,
                     model_args):

    """ plotly: Generates a numpy volume for every #save_freq number of weights
        and saves it in local results/pred/*visual_file* and results/y/*visual_file*
    """

    # load the checkpoints in the specified log directory
    train_hist_dir = os.path.join(logdir, tpu_name)
    train_hist_dir = os.path.join(train_hist_dir, visual_file)

    ######################
    """ Add the visualisation code here """
    print("\n\nTraining history directory: {}".format(train_hist_dir))
    print("+========================================================")
    print("\n\nThe directories are:")
    print('weights_dir == "checkpoint"',weights_dir == "checkpoint")
    print('weights_dir',weights_dir)
    ######################

    session_name = os.path.join(weights_dir, tpu_name, visual_file)

    # Get names within folder in gcloud
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)
    session_content = []
    tf_records_content = []
    for blob in blobs:
        if session_name in blob.name:
            session_content.append(blob.name)
        if os.path.join('tfrecords', 'valid') in blob.name:
            tf_records_content.append(blob.name)

    session_weights = []
    for item in session_content:
        if ('_weights' in item) and ('.ckpt.index' in item):
            session_weights.append(item)

    ######################
    for s in session_weights:
        print(s) #print all the checkpoint directories
    print("--")
    ######################

    # Only use part of dataset
    idx_vol= 0 # how many numpies have been save
    target = 160
    
    for i, chkpt in enumerate(session_weights):
        
        should_save_np = np.mod(i, save_freq) == 0
        
        ######################
        print('should_save_np',should_save_np)
        print('checkpoint enum i',i)
        print('save_freq set to ',save_freq)
        ######################

        if not should_save_np:      # skip this checkpoint weight
            print("skipping ", chkpt)
            continue


        name = chkpt.split('/')[-1]
        name = name.split('.inde')[0]
        trained_model.load_weights('gs://' + os.path.join(bucket_name,
                                                          weights_dir,
                                                          tpu_name,
                                                          visual_file,
                                                          name)).expect_partial()



        # sample_x = []    # x for current 160,288,288 vol
        sample_pred = []  # prediction for current 160,288,288 vol
        sample_y = []    # y for current 160,288,288 vol


        for idx, ds in enumerate(dataset):

            ######################
            print(f"the index is {idx}")
            print('Current chkpt name',name)
            ######################

            x, y = ds
            batch_size = x.shape[0]
            x = np.array(x)
            y = np.array(y)
        
            pred = trained_model.predict(x)

            ######################
            print("Current batch size set to {}. Target depth is {}".format(batch_size, target))
            print('Input image data type: {}, shape: {}'.format(type(x), x.shape))
            print('Ground truth data type: {}, shape: {}'.format(type(y), y.shape))
            print('Prediction data type: {}, shape: {}'.format(type(pred), pred.shape))
            print("=================")
            ######################

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
                print("===============")
                print("pred done")
                print(pred_vol.shape)
                print(y_vol.shape)
                print("===============")
                print('is_multi_class', is_multi_class)
                ######################

                if is_multi_class:  # or np.shape(pred_vol)[-1] not
                    pred_vol = np.argmax(pred_vol, axis=-1)
                    y_vol = np.argmax(y_vol, axis=-1)

                    ######################
                    print('np.shape(pred_vol)', np.shape(pred_vol))
                    print('np.shape(y_vol)',np.shape(y_vol))
                    ######################

                # Save volume as numpy file for plotlyyy
                fig_dir = "results"
                name_pred_npy = os.path.join(fig_dir, "pred", (visual_file + "_" + name + "_" +str(idx_vol).zfill(3)))
                name_y_npy = os.path.join(fig_dir, "ground_truth", (visual_file + "_" + name + "_" + str(idx_vol).zfill(3)))
                
                ######################
                print("npy save pred as ", name_pred_npy)
                print("npy save y as ", name_y_npy)
                print("Currently on vol ", idx_vol)
                ######################


                # Get middle xx slices cuz 288x288x160 too big
                roi = int(50 / 2)
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
                idx_vol += 1
                del pred_vol
                del y_vol

                ######################
                print("breaking after saving vol ", idx, "for ", name)
                ######################
                break



def epoch_gif(model,
              logdir,
              tfrecords_dir,
              visual_file,
              tpu_name,
              bucket_name,
              weights_dir,
              is_multi_class,
              model_args,
              which_slice,
              which_volume=1,
              epoch_limit=1000,
              gif_dir='',
              gif_cmap='gray',
              clean=False):

    #load the database
    valid_ds = read_tfrecord(tfrecords_dir=tfrecords_dir, #'gs://oai-challenge-dataset/tfrecords/valid/',
                            batch_size=160,
                            buffer_size=500,
                            augmentation=None,
                            multi_class=is_multi_class,
                            is_training=False,
                            use_bfloat16=False,
                            use_RGB=False)

    # load the checkpoints in the specified log directory
    train_hist_dir = os.path.join(logdir, tpu_name)
    train_hist_dir = os.path.join(train_hist_dir, visual_file)

    print("\n\nTraining history directory: {}".format(train_hist_dir))
    print("+========================================================")
    print("\n\nThe directories are:")

    storage_client = storage.Client()
    session_name = os.path.join(weights_dir, tpu_name, visual_file)

    blobs = storage_client.list_blobs(bucket_name)
    session_content = []
    for blob in blobs:
        if session_name in blob.name:
            session_content.append(blob.name)

    session_weights = []
    for item in session_content:
        if ('_weights' in item) and ('.ckpt.index' in item):
            session_weights.append(item)

    for s in session_weights:
        print(s) #print all the checkpoint directories
    print("--")

    #figure for gif
    fig, ax = plt.subplots()
    images_gif = []

    for chkpt in session_weights:
        name = chkpt.split('/')[-1]
        name = name.split('.inde')[0]

        if int(name.split('.')[1]) <= epoch_limit:

            print("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"\t\tLoading weights from {name.split('.')[1]} epoch")
            print(f"\t\t  {name}")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

            trained_model = model(*model_args)
            trained_model.load_weights('gs://' + os.path.join(bucket_name,
                                                              weights_dir,
                                                              tpu_name,
                                                              visual_file,
                                                              name)).expect_partial()

            for idx, ds in enumerate(valid_ds):

                if idx+1 == which_volume:
                    x, _ = ds
                    # x = np.array(x)
                    x_slice = np.expand_dims(x[which_slice-1], axis=0)
                    print('Input image data type: {}, shape: {}\n'.format(type(x), x.shape))

                    print('predicting slice {}'.format(which_slice))
                    # pred_vol = trained_model.predict(x)
                    predicted_slice = trained_model.predict(x_slice)
                    if is_multi_class:
                        # pred_vol = np.argmax(pred_vol, axis=-1)
                        predicted_slice = np.argmax(predicted_slice, axis=-1)
                    print('slice predicted\n')

                    # im = ax.imshow(pred_vol[which_slice-1,:,:], cmap=gif_cmap, animated=True)
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

                    break

        else:
            break

    pred_evolution_gif(fig, images_gif, save_dir=gif_dir, save=True, no_margins=clean)

def volume_gif(model,
               logdir,
               tfrecords_dir,
               visual_file,
               tpu_name,
               bucket_name,
               weights_dir,
               is_multi_class,
               model_args,
               which_epoch,
               which_volume=1,
               gif_dir='',
               gif_cmap='gray',
               clean=False):

    #load the database
    valid_ds = read_tfrecord(tfrecords_dir=tfrecords_dir, #'gs://oai-challenge-dataset/tfrecords/valid/',
                            batch_size=160,
                            buffer_size=500,
                            augmentation=None,
                            multi_class=is_multi_class,
                            is_training=False,
                            use_bfloat16=False,
                            use_RGB=False)

    # load the checkpoints in the specified log directory
    train_hist_dir = os.path.join(logdir, tpu_name)
    train_hist_dir = os.path.join(train_hist_dir, visual_file)

    print("\n\nTraining history directory: {}".format(train_hist_dir))
    print("+========================================================")
    print("\n\nThe directories are:")

    storage_client = storage.Client()
    session_name = os.path.join(weights_dir, tpu_name, visual_file)

    blobs = storage_client.list_blobs(bucket_name)
    session_content = []
    for blob in blobs:
        if session_name in blob.name:
            session_content.append(blob.name)

    session_weights = []
    for item in session_content:
        if ('_weights' in item) and ('.ckpt.index' in item):
            session_weights.append(item)

    for s in session_weights:
        print(s) #print all the checkpoint directories
    print("--")

    #figure for gif
    fig, ax = plt.subplots()
    images_gif = []

    for chkpt in session_weights:
        name = chkpt.split('/')[-1]
        name = name.split('.inde')[0]

        if int(name.split('.')[1]) == which_epoch:

            print("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"\t\tLoading weights from {name.split('.')[1]} epoch")
            print(f"\t\t  {name}")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

            trained_model = model(*model_args)
            trained_model.load_weights('gs://' + os.path.join(bucket_name,
                                                              weights_dir,
                                                              tpu_name,
                                                              visual_file,
                                                              name)).expect_partial()

            for idx, ds in enumerate(valid_ds):

                if idx+1 == which_volume:
                    x, _ = ds
                    x = np.array(x)
                    print('Input image data type: {}, shape: {}\n'.format(type(x), x.shape))

                    print('predicting volume {}'.format(which_volume))
                    pred_vol = trained_model.predict(x)
                    if is_multi_class:
                        pred_vol = np.argmax(pred_vol, axis=-1)
                    print('volume predicted\n')

                    for i in range(x.shape[0]):
                        print(f"Analysing slice {i+1}")
                        im = ax.imshow(pred_vol[i,:,:], cmap='gray', animated=True, aspect='auto')
                        if not clean:
                            text = ax.text(0.5,1.05,f'Slice {i+1}', 
                                        size=plt.rcParams["axes.titlesize"],
                                        ha="center", transform=ax.transAxes)
                            images_gif.append([im, text])
                        else:
                            ax.axis('off')
                            images_gif.append([im])

                    break
            
            break

    pred_evolution_gif(fig, images_gif, save_dir=gif_dir, save=True, no_margins=clean)




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

def confusion_matrix(trained_model,
                     weights_dir,
                     fig_dir,
                     dataset,
                     validation_steps,
                     multi_class,
                     model_architecture,
                     callbacks,
                     num_classes=7
                     ):

    trained_model.load_weights(weights_dir).expect_partial()
    trained_model.evaluate(dataset, steps=validation_steps, callbacks=callbacks)


    f = weights_dir.split('/')[-1]
    # Excluding parenthese before f too
    if weights_dir.endswith(f):
        writer_dir = weights_dir[:-(len(f)+1)]
    writer_dir = os.path.join(writer_dir, 'eval')
    # os.makedirs(writer_dir)
    eval_metric_writer = tf.summary.create_file_writer(writer_dir)


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

    for step, (image, label) in enumerate(dataset):
        print(step)
        pred = trained_model.predict(image)
        cm = cm + get_confusion_matrix(label, pred, classes=list(range(0, num_classes)))

        if multi_class:
            iou = iou_loss_eval(label, pred)
            dice = dice_coef_eval(label, pred)
        else:
            iou = iou_loss(label, pred)
            dice = dice_coef(label, pred)

        with eval_metric_writer.as_default():
            tf.summary.scalar('iou validation', iou, step=step)
            tf.summary.scalar('dice validation', dice, step=step)
        

            

        if step > validation_steps - 1:
            break

    fig_file = model_architecture + '_matrix.png'
    fig_dir = os.path.join(fig_dir, fig_file)
    plot_confusion_matrix(cm, fig_dir, classes=classes)
