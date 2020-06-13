import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import glob
from google.cloud import storage
from pathlib import Path
import os

from Segmentation.utils.losses import dice_coef

def plot_and_eval_3D(trained_model,
                     logdir,
                     visual_file,
                     tpu_name,
                     bucket_name,
                     weights_dir,
                     dataset):

    # load the checkpoints in the specified log directory
    train_hist_dir = os.path.join(logdir, tpu_name)
    train_hist_dir = os.path.join(train_hist_dir, visual_file)
    checkpoints = Path(train_hist_dir).glob('*')

    """ Add the visualisation code here """
    print("Training history directory: {}".format(train_hist_dir))
    print("+========================================================")
    print(f"Does the selected path exist: {Path(train_hist_dir).is_dir()}")
    print(f"The glob object is: {checkpoints}")
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
        print(s)
    print("--")

    for chkpt in session_weights:
        name = chkpt.split('/')[-1]
        name = name.split('.inde')[0]
        trained_model.load_weights('gs://' + os.path.join(bucket_name,
                                                          weights_dir,
                                                          tpu_name,
                                                          visual_file,
                                                          name)).expect_partial()

        sample_x = []    # x for current 160,288,288 vol
        sample_pred = []  # prediction for current 160,288,288 vol
        sample_y = []    # y for current 160,288,288 vol

        for idx, ds in enumerate(dataset):
            x, y = ds
            batch_size = x.shape[0]
            target = 160
            print("Current batch size set to {}. Target depth is {}".format(batch_size, target))

            x = np.array(x)
            y = np.array(y)

            pred = trained_model.predict(x)
            print('Input image data type: {}, shape: {}'.format(type(x), x.shape))
            print('Ground truth data type: {}, shape: {}'.format(type(y), y.shape))
            print('Prediction data type: {}, shape: {}'.format(type(pred), pred.shape))
            print("=================")

            if (get_depth(sample_pred) + batch_size) < target:  # check if next batch will fit in volume (160)
                sample_pred.append(pred)
                sample_y.append(y)
            else:
                remaining = target - get_depth(sample_pred)
                sample_pred.append(pred[:remaining])
                sample_y.append(y[:remaining])
                pred_vol = np.concatenate(sample_pred)
                pred_y = np.concatenate(sample_pred)
                sample_pred = [pred[remaining:]]
                sample_y = [y[remaining:]]

                print("===============")
                print("pred done")
                print(pred_vol.shape)
                print(pred_y.shape)
                print("===============")

                pred_vol_dice = dice_coef(y_vol, pred_vol)
                dices.append(pred_vol_dice)

                print("DICE:", pred_vol_dice)

                print("VOLUME DICE:", dice_coef(y_vol, pred_vol))

                pred_vol = pred_vol[50:110, 114:174, 114:174, 0]
                pred_vol = np.stack((pred_vol,) * 3, axis=-1)

                fig = plot_volume(pred_vol)
                plt.savefig(f"results/hello-hello")
                plt.close('all')

                break

            print("=================")

            if idx == 4:
                break
            # # we need to then merge into each (288,288,160) volume. Validation data should be in order

        break
