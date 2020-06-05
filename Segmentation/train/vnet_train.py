import sys
import os
from glob import glob
import datetime
import tensorflow as tf
import numpy as np
from time import time

from Segmentation.train.utils import setup_gpu
from Segmentation.utils.data_loader import read_tfrecord_3d
from Segmentation.utils.losses import dice_loss
from Segmentation.plotting.voxels import plot_volume, plot_slice
from Segmentation.model.vnet import VNet


class Train:
    def __init__(self, epochs, batch_size, enable_function,
                 model, optimizer, loss_func,
                 tfrec_dir='./Data/tfrecords/', log_dir="logs/vnet/gradient_tape/"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.enable_function = enable_function
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.tfrec_dir = tfrec_dir
        self.log_dir = log_dir

    def train_step(self, x_train, y_train, visualise):
        with tf.GradientTape() as tape:
            predictions = self.model(x_train, training=True)
            loss = self.loss_func(y_train, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        if visualise:
            return loss, predictions
        return loss, None

    def test_step(self, x_test, y_test, visualise):
        predictions = self.model(x_test, training=False)
        loss = self.loss_func(y_test, predictions)
        if visualise:
            return loss, predictions
        return loss, None

    def train_model_loop(self, train_ds, valid_ds, strategy,
                         num_to_visualise=0):
        """ Trains 3D model with custom tf loop and MirrorStrategy
        """
        def run_train_strategy(x, y, visualise):
            total_step_loss, pred = strategy.run(self.train_step, args=(x, y, visualise, ))
            return strategy.reduce(
                tf.distribute.ReduceOp.SUM, total_step_loss, axis=None), pred

        def run_test_strategy(x, y, visualise):
            total_step_loss, pred = strategy.run(self.test_step, args=(x, y, visualise, ))
            return strategy.reduce(
                tf.distribute.ReduceOp.SUM, total_step_loss, axis=None), pred

        def distributed_train_epoch(train_ds, epoch, strategy, num_to_visualise, writer):
            total_loss, num_train_batch = 0.0, 0.0
            for x_train, y_train in train_ds:
                visualise = (num_train_batch < num_to_visualise)
                loss, pred = run_train_strategy(x_train, y_train, visualise)
                loss /= strategy.num_replicas_in_sync
                total_loss += loss
                if visualise:
                    y_slice = tf.slice(y_train.values[0], [0, 80, 0, 0, 0], [-1, 1, -1, -1, -1])
                    pred_slice = tf.slice(pred.values[0], [0, 80, 0, 0, 0], [-1, 1, -1, -1, -1])
                    y_slice = tf.reshape(y_slice, (y_slice.shape[1:]))
                    pred_slice = tf.reshape(pred_slice, (pred_slice.shape[1:]))
                    img = tf.concat((y_slice, pred_slice), axis=-2)
                    with writer.as_default():
                        tf.summary.image("Train", img, step=epoch)
                num_train_batch += 1
            return total_loss / num_train_batch

        def distributed_test_epoch(valid_ds, epoch, strategy, num_to_visualise, writer):
            total_loss, num_test_batch = 0.0, 0.0
            for x_valid, y_valid in valid_ds:
                visualise = (num_test_batch < num_to_visualise)
                loss, pred = run_test_strategy(x_valid, y_valid, visualise)
                loss /= strategy.num_replicas_in_sync
                total_loss += loss
                if visualise:
                    y_slice = tf.slice(y_valid.values[0], [0, 80, 0, 0, 0], [-1, 1, -1, -1, -1])
                    pred_slice = tf.slice(pred.values[0], [0, 80, 0, 0, 0], [-1, 1, -1, -1, -1])
                    y_slice = tf.reshape(y_slice, (y_slice.shape[1:]))
                    pred_slice = tf.reshape(pred_slice, (pred_slice.shape[1:]))
                    img = tf.concat((y_slice, pred_slice), axis=-2)
                    with writer.as_default():
                        tf.summary.image("Validation", img, step=epoch)
                    # working code for plotting a 3D volume
                    # y_subvol = tf.slice(y_valid.values[0], [0, 60, 124, 124, 0], [-1, 40, 40, 40, -1])
                    # y_subvol = tf.reshape(y_subvol, (y_subvol.shape[1:4]))
                    # y_subvol = tf.stack((y_subvol,) * 3, axis=-1)
                    # plot_volume(y_subvol, show=False)
                num_test_batch += 1
            return total_loss / num_test_batch

        if self.enable_function:
            run_train_strategy = tf.function(run_train_strategy)
            run_test_strategy = tf.function(run_test_strategy)

        log_dir_now = self.log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        train_summary_writer = tf.summary.create_file_writer(log_dir_now + '/train')
        test_summary_writer = tf.summary.create_file_writer(log_dir_now + '/validation')
        train_img_writer = tf.summary.create_file_writer(log_dir_now + '/train/img')
        test_img_writer = tf.summary.create_file_writer(log_dir_now + '/validation/img')

        for e in range(self.epochs):
            print(f"Epoch {e+1}/{self.epochs}", end="")
            et0 = time()

            train_loss = distributed_train_epoch(train_ds, e, strategy, num_to_visualise, train_img_writer)
            with train_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', train_loss, step=e)

            test_loss = distributed_test_epoch(valid_ds, e, strategy, num_to_visualise, test_img_writer)
            with test_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', test_loss, step=e)

            print(f" - {time() - et0:.0f}s - loss: {train_loss:.05f} - val_loss: {test_loss:.05f}")


def load_datasets(batch_size, buffer_size,
                  tfrec_dir='./Data/tfrecords/',
                  multi_class=False, crop_size=144):
    """
    Loads tf records datasets for 3D models.
    """
    train_ds = read_tfrecord_3d(tfrecords_dir=os.path.join(tfrec_dir, 'train_3d/'),
                                batch_size=batch_size,
                                buffer_size=buffer_size,
                                multi_class=multi_class,
                                is_training=True,
                                use_keras_fit=False, crop_size=crop_size)
    valid_ds = read_tfrecord_3d(tfrecords_dir=os.path.join(tfrec_dir, 'valid_3d/'),
                                batch_size=batch_size,
                                buffer_size=buffer_size,
                                multi_class=multi_class,
                                is_training=False,
                                use_keras_fit=False, crop_size=crop_size)
    return train_ds, valid_ds


def build_model(num_channels, num_classes, **kwargs):
    """
    Builds standard vnet for 3D
    """
    model = VNet(num_channels, num_classes, **kwargs)
    return model


def main(epochs = 3,
         batch_size = 2,
         lr = 1e-3, 
         num_to_visualise = 2,
         num_channels = 4,
         buffer_size = 2,
         enable_function=False,
         tfrec_dir='./Data/tfrecords/',
         multi_class=False,
         crop_size=144,
         **model_kwargs,
         ):
    t0 = time()

    num_classes = 7 if multi_class else 1
    train_ds, valid_ds = load_datasets(batch_size, buffer_size, tfrec_dir, multi_class, crop_size=crop_size)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model = build_model(num_channels, num_classes, **model_kwargs)

        trainer = Train(epochs, batch_size, enable_function,
                        model, optimizer, dice_loss)
        
        train_ds = strategy.experimental_distribute_dataset(train_ds)
        valid_ds = strategy.experimental_distribute_dataset(valid_ds)

    trainer.train_model_loop(train_ds, valid_ds, strategy, num_to_visualise)
    print(f"{time() - t0:.02f}")


if __name__ == "__main__":
    setup_gpu()
    main(epochs=5, lr=1e-4, dropout_rate=0.0, use_batchnorm=False, crop_size=128, enable_function=True)
    # main(epochs=50, lr=1e-4, dropout_rate=0.0, noise=1e-4)
