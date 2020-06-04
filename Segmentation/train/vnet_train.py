import sys
import os
from glob import glob
import datetime
import tensorflow as tf
import numpy as np
from time import time


def load_datasets(batch_size, buffer_size,
                  tfrec_dir='./Data/tfrecords/',
                  multi_class=False, use_keras_fit=True):
    """
    Loads tf records datasets for 3D models.
    """
    from Segmentation.utils.data_loader import read_tfrecord, parse_fn_3d
    train_ds = read_tfrecord(tfrecords_dir=os.path.join(tfrec_dir, 'train_3d/'),
                             batch_size=batch_size,
                             buffer_size=buffer_size,
                             parse_fn=parse_fn_3d,
                             multi_class=multi_class,
                             is_training=True,
                             use_keras_fit=use_keras_fit)
    valid_ds = read_tfrecord(tfrecords_dir=os.path.join(tfrec_dir, 'valid_3d/'),
                             batch_size=batch_size,
                             buffer_size=buffer_size,
                             parse_fn=parse_fn_3d,
                             multi_class=multi_class,
                             is_training=False,
                             use_keras_fit=use_keras_fit)
    return train_ds, valid_ds


def build_model(num_channels, num_classes):
    """
    Builds standard vnet for 3D
    """
    from Segmentation.model.vnet import VNet
    model = VNet(num_channels, num_classes)
    return model


def train_model_keras(model, train_ds, valid_ds, epochs, batch_size,
                      loss_func, optimizer, tfrec_dir='./Data/tfrecords/'):
    """
    Trains 3D model with keras fit
    """

    model.compile(optimizer=optimizer,
                  loss=loss_func,
                  metrics=['binary_crossentropy', 'acc'])

    train_size = len(glob(os.path.join(tfrec_dir, 'train_3d/*')))
    valid_size = len(glob(os.path.join(tfrec_dir, 'valid_3d/*')))

    log_dir = "logs/vnet/keras/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)

    history = model.fit(train_ds,
                        steps_per_epoch=train_size // batch_size,
                        epochs=epochs,
                        validation_data=valid_ds,
                        validation_steps=valid_size // batch_size,
                        callbacks=[tensorboard_callback],
                        verbose=1)


def train_model_loop(model, train_ds, valid_ds, epochs, batch_size,
                     loss_func, optimizer,
                     tfrec_dir='./Data/tfrecords/', num_to_visualise=0):
    """
    Trains 3D model with custom tf loop
    """

    train_size = len(glob(os.path.join(tfrec_dir, 'train_3d/*')))
    valid_size = len(glob(os.path.join(tfrec_dir, 'valid_3d/*')))

    log_dir = "logs/vnet/gradient_tape/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_summary_writer = tf.summary.create_file_writer(log_dir + '/train')
    test_summary_writer = tf.summary.create_file_writer(log_dir + '/validation')

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    from Segmentation.train.utils import train_step, test_step
    if num_to_visualise > 0:
        from Segmentation.plotting.voxels import plot_volume, plot_slice

    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}", end="")
        et0 = time()
        for idx, (x_train, y_train) in enumerate(train_ds):
            visualise = False
            if idx < num_to_visualise:
                visualise = True
            pred = train_step(model, loss_func, optimizer, x_train, y_train, train_loss, visualise=visualise)
            if visualise:
                print("x", x_train.shape)
                print("y", y_train.shape)
                print("p", pred[0].shape)
                r = 20
                print("p", pred[0, (80 - r):(80 + r), (30 - r):(30 + r), (30 - r):(30 + r)].shape)

                sample = pred[0, (80 - r):(80 + r), (30 - r):(30 + r), (30 - r):(30 + r)]
                sample = np.squeeze(sample, -1)
                sample = np.stack((sample,) * 3, axis=-1)

                true_sample = y_train[0, (20 - r):(20 + r), (30 - r):(30 + r), (30 - r):(30 + r)]
                true_sample = np.stack((true_sample,) * 3, axis=-1)

                plot_volume(true_sample, show=True)
                plot_volume(sample, show=True)

                plot_slice(y_train[0, 120, :, :])
                plot_slice(pred[0, 120, :, :, 0])

                print(np.sum(y_train))
                print(y_train.shape)
                print("image done")

        with train_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', train_loss.result(), step=e)

        for idx, (x_valid, y_valid) in enumerate(valid_ds):
            visualise = False
            if idx < num_to_visualise:
                visualise = True
            pred = test_step(model, loss_func, x_valid, y_valid, test_loss, visualise=visualise)
            if visualise:
                print("val")
                print("x", x_valid.shape)
                print("y", y_valid.shape)
                print("p", pred.shape)
                print("===============")
                #plot_volume(y_valid[0], show=True)
                #'plot_volume(pred[0])

        with test_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', test_loss.result(), step=e)
        print(f" - {time() - et0:.0f}s - loss: {train_loss.result():.05f} - val_loss: {test_loss.result():.05f}")
        train_loss.reset_states()
        test_loss.reset_states()


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())

    from Segmentation.train.utils import setup_gpu

    setup_gpu()

    use_keras_fit = False
    num_channels = 1
    num_classes = 1  # binary segmentation problem
    buffer_size = 4
    epochs = 5
    batch_size = 1
    lr = 1e-4
    tfrec_dir = './Data/tfrecords/'

    model = build_model(num_channels, num_classes)
    train_ds, valid_ds = load_datasets(batch_size, buffer_size, tfrec_dir, use_keras_fit=use_keras_fit)

    from Segmentation.utils.losses import dice_loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    t0 = time()

    if not use_keras_fit:
        train_model_loop(model, train_ds, valid_ds, epochs, batch_size,
                         dice_loss, optimizer, tfrec_dir, num_to_visualise=5)
    else:
        train_model_keras(model, train_ds, valid_ds, epochs, batch_size,
                          dice_loss, optimizer, tfrec_dir)

    print(f"{time() - t0:.02f}")
