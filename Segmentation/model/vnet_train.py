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
          save_model=True, validate=True, train_name="", custom_train_loop=False, train_debug=False,
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
                                   examples_per_load=examples_per_load,
                                   train_debug=train_debug)
        vdataset = VolumeGenerator(batch_size, sample_shape, "v",
                                   shuffle_order=shuffle_order,
                                   normalise_input=normalise_input,
                                   remove_outliers=remove_outliers,
                                   transform_angle=False,
                                   transform_position=False,
                                   get_slice=get_slice,
                                   get_position=get_position,
                                   skip_empty=skip_empty,
                                   examples_per_load=examples_per_load,
                                   train_debug=train_debug)

    from Segmentation.utils.losses import bce_dice_loss, dice_loss
    from Segmentation.utils.losses import tversky_loss, precision, recall

    loss_name = ""
    if n_classes == 1:
        loss_func = dice_loss
        loss_name = 'dice loss'
        from tensorflow.keras.losses import binary_crossentropy
        met_loss = binary_crossentropy
        met_loss_name = 'binary_crossentropy'
    else:
        loss_func = tversky_loss
        loss_name = "tversky loss"
        from tensorflow.keras.losses import categorical_crossentropy
        met_loss = categorical_crossentropy
        met_loss_name = 'categorical_crossentropy'

    loss_hist = []
    loss_val_hist = []

    metrics = [met_loss, precision, recall]

    metric_hist = [None] * len(metrics)
    metric_val_hist = [None] * len(metrics)
    for i in range(1):  # we only care about 1 metrics
        metric_hist[i] = []
        metric_val_hist[i] = []

    save_models_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'checkpoints/train_session_{now_time}_{model}/chkp/model_' + '{epoch}',
        verbose=1)

    if custom_train_loop:
        optimizer = Adam(lr=start_lr)
        for epoch in range(epochs):
            epoch_time = time.perf_counter()
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_val_loss_avg = tf.keras.metrics.Mean()
            epoch_met_loss_avg = tf.keras.metrics.Mean()
            epoch_val_met_loss_avg = tf.keras.metrics.Mean()

            for x, y in tdataset:

                with tf.GradientTape() as tape:
                    y_ = vnet(x, training=True)
                    loss_value = loss_func(y_true=y, y_pred=y_)
                grads = tape.gradient(loss_value, vnet.trainable_variables)

                optimizer.apply_gradients(zip(grads, vnet.trainable_variables))
                epoch_loss_avg(loss_value)

                met_loss_value = met_loss(y_true=y, y_pred=y_)
                epoch_met_loss_avg(met_loss_value)

            slice_idx = int(y.shape[3] / 2)

            store_x = x[0, :, :, slice_idx, 0]
            store_y = y[0, :, :, slice_idx, 0]
            store_y_pred = y_[0, :, :, slice_idx, 0]

            for x, y in vdataset:
                y_ = vnet(x, training=False)
                val_loss_value = loss_func(y_true=y, y_pred=y_)
                epoch_val_loss_avg(val_loss_value)

                val_met_loss_value = met_loss(y_true=y, y_pred=y_)
                epoch_val_met_loss_avg(val_met_loss_value)

            store_x_val = x[0, :, :, slice_idx, 0]
            store_y_val = y[0, :, :, slice_idx, 0]
            store_y_pred_val = y_[0, :, :, slice_idx, 0]

            print(f"{time.perf_counter() - epoch_time:.0f} s epoch: {epoch}, loss: {epoch_loss_avg.result(): .4f}, loss val: {epoch_val_loss_avg.result(): .4f}, bce: {epoch_met_loss_avg.result(): .4f}, bce val: {epoch_val_met_loss_avg.result(): .4f}")

            loss_hist.append(epoch_loss_avg.result())
            loss_val_hist.append(epoch_val_loss_avg.result())

            metric_hist[0].append(epoch_met_loss_avg.result())
            metric_val_hist[0].append(epoch_val_met_loss_avg.result())

            f, axes = plt.subplots(2, 3)

            axes[0, 0].imshow(store_x)
            axes[0, 0].set_title("train raw image")
            axes[0, 1].imshow(store_y)
            axes[0, 1].set_title("train y")
            axes[0, 2].imshow(store_y_pred)
            axes[0, 2].set_title("train y pred")
            axes[1, 0].imshow(store_x_val)
            axes[1, 0].set_title("val raw image")
            axes[1, 1].imshow(store_y_val)
            axes[1, 1].set_title("val y")
            axes[1, 2].imshow(store_y_pred_val)
            axes[1, 2].set_title("val y pred")
            f.tight_layout(rect=[0, 0.05, 1, 0.95])
            f.suptitle(f"{model}: {train_name}, epoch: {epoch}")
            plt.savefig(f"checkpoints/train_session_{now_time}_{model}/train_{epoch}_{now_time}")
            print(f"plot saved: {time.perf_counter() - epoch_time:.0f}")

            vnet.save_weights(f"checkpoints/train_session_{now_time}_{model}/chkp/ckpt_{epoch}_{now_time}.cktp")
    else:

        vnet.compile(optimizer=Adam(lr=start_lr),
                     loss=loss_func,
                     metrics=metrics,
                     experimental_run_tf_function=True)

        callbacks = [
            save_models_callback
        ]

        if reduce_lr:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                     patience=10, min_lr=1e-7, verbose=1)
            )

        if validate:
            h = vnet.fit(x=tdataset, validation_data=vdataset,
                         callbacks=callbacks, epochs=epochs, verbose=1,
                         steps_per_epoch=steps, validation_steps=vsteps)
            loss_val_hist = h.history['val_loss']
            metric_val_hist[0] = h.history[f'val_{met_loss_name}']
        else:
            h = vnet.fit(x=tdataset, callbacks=callbacks, epochs=epochs, verbose=1)

        loss_hist = h.history['loss']
        metric_hist[0] = h.history[f'{met_loss_name}']

    time_taken = time.perf_counter() - start_time
    roll_period = 5
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(loss_hist, label="loss")
    ax1.plot(running_mean(loss_hist, roll_period), label="loss roll")
    if validate:
        ax1.plot(loss_val_hist, label="val loss")
        ax1.plot(running_mean(loss_val_hist, roll_period), label="val loss roll")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel(loss_name)
    ax1.legend()
    ax2.plot(metric_hist[0], label=met_loss_name)
    ax2.plot(running_mean(metric_hist[0], roll_period), label=f"{met_loss_name} roll")
    if validate:
        ax2.plot(metric_val_hist[0], label=f"val {met_loss_name}")
        ax2.plot(running_mean(metric_val_hist[0], roll_period), label=f"val {met_loss_name} roll")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("cross entropy")
    ax2.legend()
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.suptitle(f"{model}: {train_name}, {time_taken:.1f}")

    plt.savefig(f"checkpoints/train_session_{now_time}_{model}/train_result_{now_time}")
    return time_taken


if __name__ == "__main__":

    print("running eagerly:", tf.executing_eagerly())

    import sys
    sys.path.insert(0, os.getcwd())

    e = 10
    examples_per_load = 1
    batch_size = 2

    toy = train("tiny", batch_size=batch_size, sample_shape=(4, 4, 4), epochs=e,
                examples_per_load=examples_per_load,
                train_name="toy (4,4,4)", custom_train_loop=True, train_debug=True)

"""
    t0 = train("tiny", batch_size=batch_size, sample_shape=(200, 200, 160), epochs=e,
               examples_per_load=examples_per_load,
               train_name="(200,200,160)", custom_train_loop=True)

    t1 = train("small", batch_size=batch_size, sample_shape=(240, 240, 160), epochs=e,
               examples_per_load=examples_per_load,
               train_name="(240,240,160) 2 layers", custom_train_loop=True)

    # t2 = train("small", batch_size=batch_size, sample_shape=(200, 200, 160), epochs=e,
    #            examples_per_load=examples_per_load,
    #            train_name="(200,200,160)", custom_train_loop=True)

    # t3 = train("small", batch_size=batch_size, sample_shape=(240, 240, 160), epochs=e,
    #            examples_per_load=examples_per_load,
    #            train_name="(240,240,160) 4 layers", num_conv_layers=4)

    # t4 = train("small_relative", batch_size=batch_size, sample_shape=(240, 240, 160), epochs=e,
    #            examples_per_load=examples_per_load,
    #            train_name="(240,240,160) (add)", action="add")

    t5 = train("small_relative", batch_size=3, sample_shape=(288, 288, 160), epochs=e,
               examples_per_load=1,
               train_name="(288,288,160) (add)", action="add", custom_train_loop=True)

    t6 = train("slice", batch_size=batch_size, sample_shape=(384, 384, 7), epochs=e,
               examples_per_load=examples_per_load,
               train_name="(384,384,7) lr=5e-4, k=(3,3,3)", kernel_size=(3, 3, 3), custom_train_loop=True)

    t7 = train("slice", batch_size=batch_size, sample_shape=(384, 384, 7), epochs=e,
               examples_per_load=examples_per_load,
               train_name="(384,384,7) lr=5e-4, k=(3,3,1)", kernel_size=(3, 3, 1), custom_train_loop=True)
"""