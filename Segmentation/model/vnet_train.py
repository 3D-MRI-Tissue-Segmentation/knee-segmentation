import numpy as np

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

if __name__ == "__main__":
    n_classes = 1
    batch_size = 3
    shape = (128, 128, 128)

    import sys
    from os import getcwd
    sys.path.insert(0, getcwd())

    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    from Segmentation.utils.data_loader_3d import VolumeGenerator

    train_gen = VolumeGenerator(batch_size, shape, add_pos=True)
    valid_gen = VolumeGenerator(batch_size, shape, add_pos=True,
                                file_path="./Data/valid/", data_type='valid')

    from Segmentation.model.vnet_tiny import VNet_Tiny
    from Segmentation.model.vnet_small import VNet_Small
    from Segmentation.model.vnet_small_relative import VNet_Small_Relative
    vnet = VNet_Small_Relative(1, n_classes, merge_connections=True)

    from tensorflow.keras.optimizers import Adam, RMSprop
    from Segmentation.utils.losses import dsc, dice_loss, tversky_loss, bce_dice_loss, focal_tversky, precision, recall, bce_precise_dice_loss

    if n_classes == 1:
        loss_func = dice_loss
    else:
        loss_func = tversky_loss

    vnet.compile(optimizer=Adam(5e-4),
                 loss=loss_func,
                 metrics=['categorical_crossentropy', precision, recall],
                 experimental_run_tf_function=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/mymodel_{epoch}',
            verbose=1)
    ]

    roll_period = 5

    history = vnet.fit(x=train_gen, validation_data=valid_gen, callbacks=callbacks, epochs=1000, verbose=1)

    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(history.history['loss'], label="loss")
    ax1.plot(running_mean(history.history['loss'], roll_period), label="loss roll")
    ax1.plot(history.history['val_loss'], label="val loss")
    ax1.plot(running_mean(history.history['val_loss'], roll_period), label="val loss roll")
    ax1.legend()
    ax2.plot(history.history['categorical_crossentropy'], label="catcross")
    ax2.plot(running_mean(history.history['categorical_crossentropy'], roll_period), label="catcross roll")
    ax2.plot(history.history['val_categorical_crossentropy'], label="val catcross")
    ax2.plot(running_mean(history.history['val_categorical_crossentropy'], roll_period), label="val catcross roll")
    ax2.legend()
    plt.show()
