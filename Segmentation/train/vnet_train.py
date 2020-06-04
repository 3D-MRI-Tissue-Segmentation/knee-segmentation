import sys
import os
from glob import glob
import datetime
import tensorflow as tf
import numpy as np
from time import time

from Segmentation.train.utils import setup_gpu
from Segmentation.utils.data_loader import read_tfrecord, parse_fn_3d
from Segmentation.utils.losses import dice_loss
from Segmentation.plotting.voxels import plot_volume, plot_slice
from Segmentation.model.vnet import VNet


class Train:
    def __init__(self, epochs, batch_size, enable_function,
                 model, optimizer, loss_func, strategy, 
                 tfrec_dir='./Data/tfrecords/', log_dir="logs/vnet/gradient_tape/"):

        self.epochs = epochs
        self.batch_size = batch_size
        self.enable_function = enable_function
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.strategy = strategy
        self.tfrec_dir = tfrec_dir
        self.log_dir = log_dir


    def train_step(self, x_train, y_train):
        with tf.GradientTape() as tape:
            predictions = self.model(x_train, training=True)
            loss = self.loss_func(y_train, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return predictions, loss

    def test_step(self, x_test, y_test):
        predictions = self.model(x_test, training=False)
        loss = self.loss_func(y_test, predictions)
        return predictions, loss



    def train_model_loop(self, train_ds, valid_ds, strategy,
                         num_to_visualise=0):
        """
        Trains 3D model with custom tf loop
        """

        def distributed_train_epoch(train_ds, num_to_visualise):
            total_loss = 0.0
            num_train_batch = 0.0
            for idx, (x_train, y_train) in enumerate(train_ds):
                visualise = not (idx < num_to_visualise)
                pred, per_step_loss = self.strategy.run(self.train_step, args=(x_train, y_train,))
                total_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_step_loss, axis=None)
                if visualise:
                    print("visual")
                num_train_batch += 1
            return total_loss, num_train_batch

        def distributed_test_epoch(valid_ds, num_to_visualise):
            total_loss = 0.0
            num_test_batch = 0.0
            for idx, (x_valid, y_valid) in enumerate(valid_ds):
                visualise = not (idx < num_to_visualise)
                pred, per_step_loss = self.strategy.run(self.test_step, args=(x_valid, y_valid,))
                total_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_step_loss, axis=None)
                if visualise:
                    print("visual")
                num_test_batch += 1
            return total_loss, num_test_batch


        if self.enable_function:
            distributed_train_epoch = tf.function(self.train_step)
            distributed_test_epoch = tf.function(self.test_step)

        log_dir_now = self.log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        train_summary_writer = tf.summary.create_file_writer(log_dir_now + '/train')
        test_summary_writer = tf.summary.create_file_writer(log_dir_now + '/validation')

        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

        for e in range(self.epochs):
            print(f"Epoch {e+1}/{self.epochs}", end="")
            et0 = time()

            loss, num_train = distributed_train_epoch(train_ds, num_to_visualise)
            with train_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', loss, step=e)

            loss, num_valid = distributed_test_epoch(valid_ds, num_to_visualise)
            with test_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', loss.result(), step=e)

            print(f" - {time() - et0:.0f}s - loss: {train_loss.result():.05f} - val_loss: {test_loss.result():.05f}")



            # for idx, (x_train, y_train) in enumerate(train_ds):
            #     visualise = False
            #     if idx < num_to_visualise:
            #         visualise = True
            #     pred = train_step(model, loss_func, optimizer, x_train, y_train, train_loss, visualise=visualise)
            #     if visualise:
            #         print("x", x_train.shape)
            #         print("y", y_train.shape)
            #         print("p", pred[0].shape)
            #         r = 60
            #         print("p", pred[0, (80 - r):(80 + r), (144 - r):(144 + r), (144 - r):(144 + r)].shape)

            #         sample = pred[0, (80 - r):(80 + r), (144 - r):(144 + r), (144 - r):(144 + r)]
            #         sample = np.squeeze(sample, -1)
            #         sample = np.stack((sample,) * 3, axis=-1)

            #         true_sample = y_train[0, (80 - r):(80 + r), (144 - r):(144 + r), (144 - r):(144 + r)]
            #         true_sample = np.squeeze(true_sample, -1)
            #         true_sample = np.stack((true_sample,) * 3, axis=-1)

            #         print(true_sample.shape)
            #         print(sample.shape)

            #         plot_volume(true_sample, show=True)
            #         plot_volume(sample, show=True)

            #         print("y", y_train.shape)
            #         plot_slice(y_train[0, 80, :, :, 0])
            #         print("max", np.max(y_train))
            #         print("min", np.min(y_train))
            #         plot_slice(pred[0, 80, :, :, 0])
            #         print("-----------")
            #         print("sum", np.sum(y_train))
            #         print("y shape", y_train.shape)
            #         print("image done")

            # with train_summary_writer.as_default():
            #     tf.summary.scalar('epoch_loss', train_loss.result(), step=e)

            # for idx, (x_valid, y_valid) in enumerate(valid_ds):
            #     visualise = False
            #     if idx < num_to_visualise:
            #         visualise = True
            #     pred = test_step(model, loss_func, x_valid, y_valid, test_loss, visualise=visualise)
            #     if visualise:
            #         print("val")
            #         print("x", x_valid.shape)
            #         print("y", y_valid.shape)
            #         print("p", pred.shape)
            #         print("===============")
            #         # plot_volume(y_valid[0], show=True)
            #         # plot_volume(pred[0])

            # with test_summary_writer.as_default():
            #     tf.summary.scalar('epoch_loss', test_loss.result(), step=e)
            # print(f" - {time() - et0:.0f}s - loss: {train_loss.result():.05f} - val_loss: {test_loss.result():.05f}")
            # train_loss.reset_states()
            # test_loss.reset_states()


def load_datasets(batch_size, buffer_size,
                  tfrec_dir='./Data/tfrecords/',
                  multi_class=False):
    """
    Loads tf records datasets for 3D models.
    """
    train_ds = read_tfrecord(tfrecords_dir=os.path.join(tfrec_dir, 'train_3d/'),
                            batch_size=batch_size,
                            buffer_size=buffer_size,
                            parse_fn=parse_fn_3d,
                            multi_class=multi_class,
                            is_training=True,
                            use_keras_fit=False)
    valid_ds = read_tfrecord(tfrecords_dir=os.path.join(tfrec_dir, 'valid_3d/'),
                            batch_size=batch_size,
                            buffer_size=buffer_size,
                            parse_fn=parse_fn_3d,
                            multi_class=multi_class,
                            is_training=False,
                            use_keras_fit=False)
    return train_ds, valid_ds


def build_model(num_channels, num_classes):
    """
    Builds standard vnet for 3D
    """
    model = VNet(num_channels, num_classes)
    return model


def main(epochs = 75,
         batch_size = 1,
         lr = 1e-3, 
         num_to_visualise = 0,
         num_channels = 4,
         buffer_size = 4,
         enable_function=False,
         tfrec_dir='./Data/tfrecords/',
         multi_class=False, 
         ):
    t0 = time()

    num_classes = 7 if multi_class else 1
    train_ds, valid_ds = load_datasets(batch_size, buffer_size, tfrec_dir, multi_class)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model = build_model(num_channels, num_classes)

        trainer = Train(epochs, batch_size, enable_function,
                        model, optimizer, dice_loss, strategy)
        
        train_ds = strategy.experimental_distribute_dataset(train_ds)
        valid_ds = strategy.experimental_distribute_dataset(valid_ds)

        trainer.train_model_loop(train_ds, valid_ds, strategy, num_to_visualise)
    print(f"{time() - t0:.02f}")


if __name__ == "__main__":
    setup_gpu()
    main()
    
    

    

    

    
    

    
    

    
