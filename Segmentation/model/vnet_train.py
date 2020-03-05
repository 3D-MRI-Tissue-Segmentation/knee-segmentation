import numpy as np
import math

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

if __name__ == "__main__":
    n_classes = 1
    batch_size = 1
    shape = (128, 128, 128)
    epochs = 1000
    validate = True

    import sys
    from os import getcwd
    sys.path.insert(0, getcwd())

    import tensorflow as tf
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

    from Segmentation.utils.data_loader_3d import VolumeGenerator

    train_gen = VolumeGenerator(batch_size, shape, norm=True, add_pos=True)
    valid_gen = VolumeGenerator(batch_size, shape,
                                file_path="./Data/valid/", data_type='valid', norm=True, add_pos=True)

    from Segmentation.model.vnet_tiny import VNet_Tiny
    from Segmentation.model.vnet_small import VNet_Small
    from Segmentation.model.vnet_small_relative import VNet_Small_Relative
    vnet = VNet_Small_Relative(1, n_classes, merge_connections=True)
    # vnet = VNet_Tiny(1, n_classes, merge_connections=True)
    # vnet = VNet_Small(1, n_classes, merge_connections=True)

    from tensorflow.keras.optimizers import Adam
    from Segmentation.utils.losses import dsc, dice_loss, tversky_loss, bce_dice_loss, focal_tversky, precision, recall, bce_precise_dice_loss

    if n_classes == 1:
        loss_func = dice_loss
    else:
        loss_func = tversky_loss

    vnet.compile(optimizer=Adam(lr=5e-5),
                 loss=loss_func,
                 metrics=['categorical_crossentropy', precision, recall],
                 experimental_run_tf_function=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/mymodel_{epoch}',
            verbose=1),
    ]
    roll_period = 5

    if validate:
        history_1 = vnet.fit(x=train_gen, validation_data=valid_gen, callbacks=callbacks, epochs=epochs, verbose=1)
    else:
        history_1 = vnet.fit(x=train_gen, callbacks=callbacks, epochs=epochs, verbose=1)

    import matplotlib.pyplot as plt
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
    plt.show()

    from datetime import datetime
    now = datetime.now()
    now_time = now.strftime('%Y_%m_%d-%H_%M_%S')
    plt.savefig(f"train_result_{now_time}")
