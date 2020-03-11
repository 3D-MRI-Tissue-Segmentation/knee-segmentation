import numpy as np
import math
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import os
import time


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def train(model, n_classes=1, batch_size=1, sample_shape=(128, 128, 128), epochs=10,
          save_model=True, validate=True, train_name="",
          reduce_lr=False, start_lr=5e-4, dataset_load_method=None,
          shuffle_order=True, normalise_input=True, remove_outliers=True,
          transform_angle=False, transform_position="normal",
          skip_empty=True, examples_per_load=1,
          **model_kwargs):
    start_time = time.perf_counter()
    get_position, get_slice = False, False
    if model == "tiny":
        from Segmentation.model.vnet_tiny import VNet_Tiny
        vnet = VNet_Tiny(1, n_classes, **model_kwargs)
    elif model == "small":
        from Segmentation.model.vnet_small import VNet_Small
        vnet = VNet_Small(1, n_classes, **model_kwargs)
    elif model == "small_relative":
        from Segmentation.model.vnet_small_relative import VNet_Small_Relative
        vnet = VNet_Small_Relative(1, n_classes, **model_kwargs)
        get_position = True
    elif model == "slice":
        from Segmentation.model.vnet_slice import VNet_Slice
        vnet = VNet_Slice(1, n_classes, **model_kwargs)
        get_slice = True
    else:
        raise NotImplementedError()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    now = datetime.now()
    now_time = now.strftime('%Y_%m_%d-%H_%M_%S')

    if not os.path.exists(f'checkpoints/train_session_{now_time}_{model}'):
        os.makedirs(f'checkpoints/train_session_{now_time}_{model}')

    setup_gpu()

    steps = None
    vsteps = None
    if dataset_load_method == "tf":
        get_slice = False
        if model == "slice":
            get_slice = True

        from Segmentation.utils.data_loader_3d_tf import get_dataset
        tdataset, steps = get_dataset(file_path="t",
                                      sample_shape=sample_shape,
                                      remove_outliers=remove_outliers,
                                      transform_angle=False,
                                      transform_position=transform_position,
                                      get_slice=get_slice,
                                      get_position=get_position,
                                      skip_empty=skip_empty)
        vdataset, vsteps = get_dataset(file_path="v",
                                       sample_shape=sample_shape,
                                       remove_outliers=remove_outliers,
                                       transform_angle=False,
                                       transform_position=False,
                                       get_slice=get_slice,
                                       get_position=get_position,
                                       skip_empty=skip_empty)
        tdataset = tdataset.repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        vdataset = vdataset.repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    elif dataset_load_method is None:
        get_slice = False
        if model == "slice":
            get_slice = True

        from Segmentation.utils.data_loader_3d import VolumeGenerator
        tdataset = VolumeGenerator(batch_size, sample_shape, "t",
                                   shuffle_order=shuffle_order,
                                   normalise_input=normalise_input,
                                   remove_outliers=remove_outliers,
                                   transform_angle=False,
                                   transform_position=transform_position,
                                   get_slice=get_slice,
                                   get_position=get_position,
                                   skip_empty=skip_empty,
                                   examples_per_load=examples_per_load)
        vdataset = VolumeGenerator(batch_size, sample_shape, "v",
                                   shuffle_order=shuffle_order,
                                   normalise_input=normalise_input,
                                   remove_outliers=remove_outliers,
                                   transform_angle=False,
                                   transform_position=False,
                                   get_slice=get_slice,
                                   get_position=get_position,
                                   skip_empty=skip_empty,
                                   examples_per_load=examples_per_load)

    from Segmentation.utils.losses import bce_dice_loss, dice_loss
    from Segmentation.utils.losses import tversky_loss, precision, recall

    loss_name = ""
    if n_classes == 1:
        loss_func = bce_dice_loss
        loss_name = "crossentropy and dice loss"
    else:
        loss_func = tversky_loss
        loss_name = "tversky loss"
    vnet.compile(optimizer=Adam(lr=start_lr),
                 loss=loss_func,
                 metrics=['categorical_crossentropy', dice_loss, precision, recall],
                 experimental_run_tf_function=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'checkpoints/train_session_{now_time}_{model}/chkp/model_' + '{epoch}',
            verbose=1),
    ]

    if reduce_lr:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=10, min_lr=1e-7, verbose=1)
        )

    if validate:
        history_1 = vnet.fit(x=tdataset, validation_data=vdataset,
                             callbacks=callbacks, epochs=epochs, verbose=1,
                             steps_per_epoch=steps, validation_steps=vsteps)
    else:
        history_1 = vnet.fit(x=tdataset, callbacks=callbacks, epochs=epochs, verbose=1)

    time_taken = time.perf_counter() - start_time
    roll_period = 5
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(history_1.history['loss'], label="loss")
    ax1.plot(running_mean(history_1.history['loss'], roll_period), label="loss roll")
    if validate:
        ax1.plot(history_1.history['val_loss'], label="val loss")
        ax1.plot(running_mean(history_1.history['val_loss'], roll_period), label="val loss roll")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel(loss_name)
    ax1.legend()
    ax2.plot(history_1.history['categorical_crossentropy'], label="catcross")
    ax2.plot(running_mean(history_1.history['categorical_crossentropy'], roll_period), label="catcross roll")
    if validate:
        ax2.plot(history_1.history['val_categorical_crossentropy'], label="val catcross")
        ax2.plot(running_mean(history_1.history['val_categorical_crossentropy'], roll_period), label="val catcross roll")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("crossentropy")
    ax2.legend()
    f.suptitle(f"{model}: {train_name}, {time_taken:.1f}")
    plt.savefig(f"checkpoints/train_session_{now_time}_{model}/train_result_{now_time}")
    return time_taken

if __name__ == "__main__":

    print("running eagerly:", tf.executing_eagerly())

    import time
    import sys
    sys.path.insert(0, os.getcwd())

    e = 250
    examples_per_load = 1
    batch_size = 3

    t0 = train("tiny", batch_size=batch_size, sample_shape=(288, 288, 160), epochs=e, examples_per_load=examples_per_load,
               train_name="(288,288,160)")

    t1 = train("small", batch_size=batch_size, sample_shape=(288, 288, 160), epochs=e, examples_per_load=examples_per_load,
               train_name="(288,288,160) 2 layers")

    t2 = train("small", batch_size=batch_size, sample_shape=(240, 240, 160), epochs=e, examples_per_load=examples_per_load,
               train_name="(240,240,160) 3 layers", num_conv_layers=3)

    t3 = train("small", batch_size=batch_size, sample_shape=(240, 240, 160), epochs=e, examples_per_load=examples_per_load,
               train_name="(240,240,160) 4 layers", num_conv_layers=4)

    t4 = train("small_relative", batch_size=batch_size, sample_shape=(240, 240, 160), epochs=e, examples_per_load=examples_per_load,
               train_name="(240,240,160) (add)", action="add")

    t5 = train("small_relative", batch_size=3, sample_shape=(288, 288, 160), epochs=e, examples_per_load=1,
               train_name="(288,288,160) (add)", action="add")

    t6 = train("slice", batch_size=batch_size, sample_shape=(384, 384, 7), epochs=e, examples_per_load=examples_per_load,
               train_name="(384,384,7) lr=5e-4, k=(3,3,3)", kernel_size=(3, 3, 3))

    t7 = train("slice", batch_size=batch_size, sample_shape=(384, 384, 7), epochs=e, examples_per_load=examples_per_load,
               train_name="(384,384,7) lr=5e-4, k=(3,3,1)", kernel_size=(3, 3, 1))
