import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import glob
from google.cloud import storage
from pathlib import Path
import os

from Segmentation.utils.losses import dice_coef
from Segmentation.plotting.voxels import plot_volume
from Segmentation.utils.training_utils import visualise_binary, visualise_multi_class
from Segmentation.utils.evaluation_metrics import get_confusion_matrix, plot_confusion_matrix

def get_depth(conc):
    depth = 0
    for batch in conc:
        depth += batch.shape[0]
    return depth

def plot_and_eval_3D(trained_model,
                     logdir,
                     visual_file,
                     tpu_name,
                     bucket_name,
                     weights_dir,
                     is_multi_class,
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

        pred_vols = []
        y_vols = []

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
                y_vol = np.concatenate(sample_pred)
                sample_pred = [pred[remaining:]]
                sample_y = [y[remaining:]]

                pred_vols.append(pred_vol)
                y_vols.append(y_vol)

                print("===============")
                print("pred done")
                print(pred_vol.shape)
                print(y_vol.shape)
                print("===============")

                pred_vol_dice = dice_coef(y_vol, pred_vol)

                print("DICE:", pred_vol_dice)

                # pred_vol = pred_vol[50:110, 114:174, 114:174, 0]
                # pred_vol = np.stack((pred_vol,) * 3, axis=-1)

                # fig = plot_volume(pred_vol)
                # plt.savefig(f"results/hello-hello")
                # plt.close('all')

                if idx == 2:
                    print("Number of vols:", len(pred_vols), len(y_vols))
                    batch_pred_vols = np.concatenate(pred_vols)
                    batch_y_vols = np.concatenate(y_vols)

                    print("BATCH pred SIZE:", batch_pred_vols.shape)
                    print("BATCH y SIZE:", batch_y_vols.shape)

                    print("DICE BATCH:", dice_coef(batch_y_vols, batch_pred_vols))

                    break

                if is_multi_class:  # or np.shape(pred_vol)[-1] not
                    pred_vol = np.argmax(pred_vol, axis=-1)

                # Figure saving
                fig_dir = "results"
                fig = plot_volume(pred_vol)
                plt.savefig(f"results/hello-hello")
                plt.close('all')

                # Save volume as numpy file for plotlyyy
                vol_name_npy = os.path.join(fig_dir, (visual_file + "_" + idx))
                np.save(pred_vol, vol_name_npy)
                print("npy saved as ", vol_name_npy)

            print("=================")

            if idx == 4:
                break
            # # we need to then merge into each (288,288,160) volume. Validation data should be in order

        break

def confusion_matrix(trained_model,
                     weights_dir,
                     fig_dir,
                     dataset,
                     validation_steps,
                     multi_class,
                     model_architecture,
                     num_classes=7):

    trained_model.load_weights(weights_dir).expect_partial()
    trained_model.evaluate(dataset, steps=validation_steps)

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
        if multi_class:
            visualise_multi_class(label, pred)
        else:
            visualise_binary(label, pred, fig_dir)
        cm = cm + get_confusion_matrix(label, pred, classes=list(range(0, num_classes)))

        if step > validation_steps - 1:
            break

    fig_file = model_architecture + '_matrix.png'
    fig_dir = os.path.join(fig_dir, fig_file)
    plot_confusion_matrix(cm, fig_dir, classes=classes)
