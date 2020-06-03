import sys
import os
from glob import glob
import datetime
import tensorflow as tf
from time import time


def load_datasets(batch_size, buffer_size,
                  tfrec_dir='./Data/tfrecords/', multi_class=False):
    """
    Loads tf records datasets for 3D models.
    """
    from Segmentation.utils.data_loader import read_tfrecord, parse_fn_3d
    train_ds = read_tfrecord(tfrecords_dir=os.path.join(tfrec_dir, 'train_3d/'),
                             batch_size=batch_size,
                             buffer_size=buffer_size,
                             parse_fn=parse_fn_3d,
                             multi_class=multi_class,
                             is_training=True)
    valid_ds = read_tfrecord(tfrecords_dir=os.path.join(tfrec_dir, 'valid_3d/'),
                             batch_size=batch_size,
                             buffer_size=buffer_size,
                             parse_fn=parse_fn_3d,
                             multi_class=multi_class,
                             is_training=False)
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
                     loss_func, optimizer, tfrec_dir='./Data/tfrecords/'):
    """
    Trains 3D model with custom tf loop
    """

    train_size = len(glob(os.path.join(tfrec_dir, 'train_3d/*')))
    valid_size = len(glob(os.path.join(tfrec_dir, 'valid_3d/*')))

    log_dir = "logs/vnet/gradient_tape/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_summary_writer = tf.summary.create_file_writer(log_dir + '/train')
    test_summary_writer = tf.summary.create_file_writer(log_dir + '/test')

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    from Segmentation.train.utils import train_step, test_step

    for e in range(epochs):
        for (x_train, y_train) in train_ds:
            train_step(model, loss_func, optimizer, x_train, y_train, train_loss)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=e)

        for (x_valid, y_valid) in valid_ds:
            test_step(model, loss_func, x_valid, y_valid, test_loss)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=e)

        train_loss.reset_states()
        test_loss.reset_states()

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())

    from Segmentation.train.utils import setup_gpu

    setup_gpu()

    num_channels = 1
    num_classes = 1  # binary segmentation problem
    buffer_size = 4
    epochs = 3
    batch_size = 1
    lr = 1e-4
    tfrec_dir = './Data/tfrecords/'

    model = build_model(num_channels, num_classes)
    train_ds, valid_ds = load_datasets(batch_size, buffer_size, tfrec_dir)

    from Segmentation.utils.losses import dice_loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    t0 = time()

    # train_model_loop(model, train_ds, valid_ds, epochs, batch_size,
    #                  dice_loss, optimizer, tfrec_dir)

    train_model_keras(model, train_ds, valid_ds, epochs, batch_size,
                      dice_loss, optimizer, tfrec_dir)

    print(time() - t0)

    