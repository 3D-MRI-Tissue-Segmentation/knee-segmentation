if __name__ == "__main__":
    import sys
    from os import getcwd
    sys.path.insert(0, getcwd())

    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    from Segmentation.utils.data_loader_3d import VolumeGenerator

    train_gen = VolumeGenerator(8, (96, 96, 96))

    from Segmentation.model.vnet_tiny import VNet_Tiny
    from Segmentation.model.vnet_small import VNet_Small
    vnet = VNet_Small(1, 6, merge_connections=True)

    from tensorflow.keras.optimizers import Adam
    from Segmentation.utils.training_utils import tversky_loss, dice_loss, dice_coef_loss, tversky
    loss_func = tversky

    metrics = ['categorical_crossentropy']
    vnet.compile(optimizer=Adam(0.00001),
                 loss=loss_func,
                 metrics=metrics,
                 experimental_run_tf_function=True)

    history = vnet.fit(x=train_gen, epochs=200, verbose=1)
    loss_history = history.history['loss']
