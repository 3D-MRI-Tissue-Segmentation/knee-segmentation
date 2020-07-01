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





# def plot_and_eval_3D(model,
                    #  logdir,
                    #  visual_file,
                    #  tpu_name,
                    #  bucket_name,
                    #  weights_dir,
                    #  is_multi_class,
                    #  save_freq,
                    #  dataset,
                    #  model_args):

    # """ plotly: Generates a numpy volume for every #save_freq number of weights
    #     and saves it in local results/pred/*visual_file* and results/y/*visual_file*

    #     Once numpy's are generated, run the following in console to get an embeddable html file:
    #         python3 Visualization/plotly_3d_voxels/run_plotly.py -dir_l FOLDER_TO_Y_SAMPLES
    #          -dir_r FOLDER_TO_PREDICTIONS


    # """

    # session_weights = get_all_weights(bucket_name, logdir, tpu_name, visual_file, weights_dir)

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
    #             # print('is_multi_class', is_multi_class)
    #             ######################

    #             if is_multi_class:  # or np.shape(pred_vol)[-1] not
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
            #   is_multi_class,
            #   model_args,
            #   which_slice,
            #   which_volume=1,
            #   epoch_limit=1000,
            #   gif_dir='',
            #   gif_cmap='gray',
            #   clean=False):

    # #load the database
    # valid_ds = read_tfrecord(tfrecords_dir=tfrecords_dir, #'gs://oai-challenge-dataset/tfrecords/valid/',
    #                         batch_size=160,
    #                         buffer_size=500,
    #                         augmentation=aug_strategy,
    #                         multi_class=is_multi_class,
    #                         is_training=False,
    #                         use_bfloat16=False,
    #                         use_RGB=False)

    # # load the checkpoints in the specified log directory
    # session_weights = get_all_weights(bucket_name, logdir, tpu_name, visual_file, weights_dir)

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
    #                 if is_multi_class:
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
            #    is_multi_class,
            #    model_args,
            #    which_epoch,
            #    which_volume=1,
            #    gif_dir='',
            #    gif_cmap='gray',
            #    clean=False):

    # #load the database
    # valid_ds = read_tfrecord(tfrecords_dir=tfrecords_dir, #'gs://oai-challenge-dataset/tfrecords/valid/',
    #                         batch_size=160,
    #                         buffer_size=500,
    #                         augmentation=aug_strategy,
    #                         multi_class=is_multi_class,
    #                         is_training=False,
    #                         use_bfloat16=False,
    #                         use_RGB=False)

    # # load the checkpoints in the specified log directory
    # session_weights = get_all_weights(bucket_name, logdir, tpu_name, visual_file, weights_dir)

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
    #                 if is_multi_class:
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
                        #   is_multi_class,
                        #   model_args,
                        #   which_epoch,
                        #   which_volume=1,
                        #   gif_dir='',
                        #   gif_cmap='gray',
                        #   clean=False):

    # #load the database
    # valid_ds = read_tfrecord(tfrecords_dir=tfrecords_dir, #'gs://oai-challenge-dataset/tfrecords/valid/',
    #                         batch_size=160,
    #                         buffer_size=500,
    #                         augmentation=None,
    #                         multi_class=is_multi_class,
    #                         is_training=False,
    #                         use_bfloat16=False,
    #                         use_RGB=False)

    # # load the checkpoints in the specified log directory
    # session_weights = get_all_weights(bucket_name, logdir, tpu_name, visual_file, weights_dir)

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
    #                 if is_multi_class:
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
