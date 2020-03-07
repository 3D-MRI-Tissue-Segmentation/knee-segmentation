import numpy as np
import math
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
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


def train(model, n_classes=1, batch_size=1, shape=(128, 128, 128), epochs=1000,
          validate=True, merge_connect=True, norm=True, save_model=True,
          slice_index=None, examples_per_load=1, max_angle=None, train_name="", **model_kwargs):
    add_pos = False
    if model == "tiny":
        from Segmentation.model.vnet_tiny import VNet_Tiny
        vnet = VNet_Tiny(1, n_classes, merge_connections=merge_connect, **model_kwargs)
        assert slice_index is None, "Tiny requires slice index to be none"
    elif model == "small":
        from Segmentation.model.vnet_small import VNet_Small
        vnet = VNet_Small(1, n_classes, merge_connections=merge_connect, **model_kwargs)
        assert slice_index is None, "Small requires slice index to be none"
    elif model == "small_relative":
        from Segmentation.model.vnet_small_relative import VNet_Small_Relative
        vnet = VNet_Small_Relative(1, n_classes, merge_connections=merge_connect, **model_kwargs)
        assert slice_index is None, "Small relative requires slice index to be none"
        add_pos = True
    elif model == "slice":
        from Segmentation.model.vnet_slice import VNet_Slice
        vnet = VNet_Slice(1, n_classes, merge_connections=merge_connect, **model_kwargs)
    else:
        raise NotImplementedError()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    now = datetime.now()
    now_time = now.strftime('%Y_%m_%d-%H_%M_%S')

    if not os.path.exists(f'checkpoints/train_session_{now_time}_{model}'):
        os.makedirs(f'checkpoints/train_session_{now_time}_{model}')

    setup_gpu()

    from Segmentation.utils.data_loader_3d import VolumeGenerator
    train_gen = VolumeGenerator(batch_size, shape,
                                norm=norm, add_pos=add_pos,
                                slice_index=slice_index,
                                examples_per_load=examples_per_load, max_angle=max_angle)
    valid_gen = VolumeGenerator(batch_size, shape,
                                file_path="./Data/valid/", data_type='valid',
                                norm=norm, add_pos=add_pos,
                                slice_index=slice_index,
                                examples_per_load=examples_per_load)

    from Segmentation.utils.losses import dsc, dice_loss, tversky_loss, bce_dice_loss, focal_tversky, precision, recall, bce_precise_dice_loss

    if n_classes == 1:
        loss_func = dice_loss
    else:
        loss_func = tversky_loss

    vnet.compile(optimizer=Adam(lr=1e-5),
                 loss=loss_func,
                 metrics=['categorical_crossentropy', precision, recall],
                 experimental_run_tf_function=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'checkpoints/train_session_{now_time}_{model}/chkp/model_' + '{epoch}',
            verbose=1),
    ]
    roll_period = 5

    if validate:
        history_1 = vnet.fit(x=train_gen, validation_data=valid_gen, callbacks=callbacks, epochs=epochs, verbose=1)
    else:
        history_1 = vnet.fit(x=train_gen, callbacks=callbacks, epochs=epochs, verbose=1)

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(history_1.history['loss'], label="loss")
    ax1.plot(running_mean(history_1.history['loss'], roll_period), label="loss roll")
    if validate:
        ax1.plot(history_1.history['val_loss'], label="val loss")
        ax1.plot(running_mean(history_1.history['val_loss'], roll_period), label="val loss roll")
    ax1.legend()
    ax2.plot(history_1.history['categorical_crossentropy'], label="catcross")
    ax2.plot(running_mean(history_1.history['categorical_crossentropy'], roll_period), label="catcross roll")
    if validate:
        ax2.plot(history_1.history['val_categorical_crossentropy'], label="val catcross")
        ax2.plot(running_mean(history_1.history['val_categorical_crossentropy'], roll_period), label="val catcross roll")
    ax2.legend()
    f.suptitle(f"{model}: {train_name}")
    plt.savefig(f"checkpoints/train_session_{now_time}_{model}/train_result_{now_time}")


if __name__ == "__main__":

    print(tf.executing_eagerly())

    import sys, os
    sys.path.insert(0, os.getcwd())

    e = 200

    train("tiny", shape=(28,28,28), epochs=e, examples_per_load=3, train_name="(28,28,28) without rotate")
    # train("tiny", shape=(28,28,28), epochs=e, examples_per_load=3, max_angle=2, train_name="(28,28,28) with rotate")

    train("small", shape=(96,96,96), epochs=e, examples_per_load=3, train_name="(96,96,96) without rotate")
    train("small", shape=(128,128,128), epochs=e, examples_per_load=3, train_name="(128,128,128) without rotate")
    # train("small", shape=(96,96,96), epochs=e, max_angle=2, train_name="(96,96,96) without rotate")

    train("small_relative", shape=(96,96,96), epochs=e, examples_per_load=3, train_name="(96,96,96) without rotate (multiply)", action='multiply')
    train("small_relative", shape=(128,128,128), epochs=e, examples_per_load=3, train_name="(128,128,128) without rotate (multiply)", action='multiply')
    train("small_relative", shape=(128,128,128), epochs=e, examples_per_load=3, train_name="(128,128,128) without rotate (add)", action='add')
    train("small_relative", shape=(128,128,128), epochs=e, examples_per_load=3, norm=False, train_name="(128,128,128) without rotate (multiply, no norm)", action='multiply')
    train("small_relative", shape=(128,128,128), epochs=e, examples_per_load=3, merge_connect=False, train_name="(128,128,128) without rotate (multiply, no merge)", action='multiply')
    # train("small_relative", shape=(96,96,96), epochs=e, max_angle=2, train_name="(96,96,96) without rotate")

    train("slice", epochs=e, shape=(384, 384, 3), slice_index=2, examples_per_load=10, train_name="slice (384,384,3)")
    train("slice", epochs=e, shape=(384, 384, 5), slice_index=3, examples_per_load=10, train_name="slice (384,384,5)")
    train("slice", epochs=e, shape=(384, 384, 7), slice_index=4, examples_per_load=10, train_name="slice (384,384,7)")
    train("slice", epochs=e, shape=(384, 384, 9), slice_index=5, examples_per_load=10, train_name="slice (384,384,9)")
