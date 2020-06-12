import sys
import os
from glob import glob
import datetime
import tensorflow as tf
import numpy as np
from time import time

from Segmentation.train.utils import setup_gpu, LearningRateUpdate
from Segmentation.utils.data_loader import read_tfrecord_3d
from Segmentation.utils.losses import dice_loss, tversky_loss
from Segmentation.plotting.voxels import plot_volume, plot_slice, plot_to_image
from Segmentation.model.vnet import VNet

colour_maps = {
    1: [tf.constant([1, 1, 1], dtype=tf.float32), tf.constant([[[[255, 255, 0]]]], dtype=tf.float32)],  # background / black
    2: [tf.constant([2, 2, 2], dtype=tf.float32), tf.constant([[[[0, 255, 255]]]], dtype=tf.float32)],
    3: [tf.constant([3, 3, 3], dtype=tf.float32), tf.constant([[[[255, 0, 255]]]], dtype=tf.float32)],
    4: [tf.constant([4, 4, 4], dtype=tf.float32), tf.constant([[[[255, 255, 255]]]], dtype=tf.float32)],
    5: [tf.constant([5, 5, 5], dtype=tf.float32), tf.constant([[[[120, 120, 120]]]], dtype=tf.float32)],
    6: [tf.constant([6, 6, 6], dtype=tf.float32), tf.constant([[[[255, 165, 0]]]], dtype=tf.float32)],
}

class Train:
    def __init__(self, epochs, batch_size, enable_function,
                 model, optimizer, loss_func, lr_manager, predict_slice,
                 tfrec_dir='./Data/tfrecords/', log_dir="logs/"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.enable_function = enable_function
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.lr_manager = lr_manager
        self.predict_slice = predict_slice
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
                         multi_class, debug=False, num_to_visualise=0):
        """ Trains 3D model with custom tf loop and MirrorStrategy
        """
        vol_visual_freq = 5

        def replace_vector(img, search, replace):
            condition = tf.equal(img, search)
            condition = tf.reduce_all(condition, axis=-1)
            condition = tf.stack((condition,) * img.shape[-1], axis=-1)
            replace_tiled = tf.tile(replace, img.shape[:-1])
            replace_tiled = tf.reshape(replace_tiled, img.shape)
            return tf.where(condition, replace_tiled, img)

        def get_mid_slice(x, y, pred, multi_class):
            mid = tf.cast(tf.divide(tf.shape(y)[1], 2), tf.int32)
            x_slice = tf.slice(x, [0, mid, 0, 0, 0], [1, 1, -1, -1, -1])
            y_slice = tf.slice(y, [0, mid, 0, 0, 0], [1, 1, -1, -1, -1])
            pred_slice = tf.slice(pred, [0, mid, 0, 0, 0], [1, 1, -1, -1, -1])
            if multi_class:
                x_slice = tf.squeeze(x_slice, axis=-1)
                x_slice = tf.stack((x_slice,) * 3, axis=-1)
                y_slice = tf.argmax(y_slice, axis=-1)
                y_slice = tf.stack((y_slice,) * 3, axis=-1)
                y_slice = tf.cast(y_slice, tf.float32)
                pred_slice = tf.argmax(pred_slice, axis=-1)
                pred_slice = tf.stack((pred_slice,) * 3, axis=-1)
                pred_slice = tf.cast(pred_slice, tf.float32)
                for c in colour_maps:
                    y_slice = replace_vector(y_slice, colour_maps[c][0], colour_maps[c][1])
                    pred_slice = replace_vector(pred_slice, colour_maps[c][0], colour_maps[c][1])
            else:
                pred_slice = tf.math.round(pred_slice)
            img = tf.concat((x_slice, y_slice, pred_slice), axis=-2)
            return tf.reshape(img, (img.shape[1:]))

        def get_mid_vol(y, pred, multi_class, rad=8):
            y_shape = tf.shape(y)
            y_subvol = tf.slice(y, [0, (y_shape[1] // 2) - rad, (y_shape[2] // 2) - rad, (y_shape[3] // 2) - rad, 0], [1, rad * 2, rad * 2, rad * 2, -1])
            if multi_class:
                y_subvol = tf.argmax(y_subvol, axis=-1)
                y_subvol = tf.cast(y_subvol, tf.float32)
            else:
                y_subvol = tf.reshape(y_subvol, (y_subvol.shape[1:4]))
            y_subvol = tf.stack((y_subvol,) * 3, axis=-1)
            pred_subvol = tf.slice(pred, [0, (y_shape[1] // 2) - rad, (y_shape[2] // 2) - rad, (y_shape[3] // 2) - rad, 0], [1, rad * 2, rad * 2, rad * 2, -1])
            if multi_class:
                pred_subvol = tf.argmax(pred_subvol, axis=-1)
                pred_subvol = tf.cast(pred_subvol, tf.float32)
            else:
                pred_subvol = tf.math.round(pred_subvol)  # new
                pred_subvol = tf.reshape(pred_subvol, (pred_subvol.shape[1:4]))
            pred_subvol = tf.stack((pred_subvol,) * 3, axis=-1)
            if multi_class:
                for c in colour_maps:
                    y_subvol = replace_vector(y_subvol, colour_maps[c][0], tf.divide(colour_maps[c][1], 255))
                    pred_subvol = replace_vector(pred_subvol, colour_maps[c][0], tf.divide(colour_maps[c][1], 255))
                y_subvol = tf.squeeze(y_subvol, axis=0)
                pred_subvol = tf.squeeze(pred_subvol, axis=0)
            fig = plot_volume(y_subvol, show=False)
            y_img = plot_to_image(fig)
            fig = plot_volume(pred_subvol, show=False)
            pred_img = plot_to_image(fig)
            img = tf.concat((y_img, pred_img), axis=-2)
            return img

        def run_train_strategy(x, y, visualise):
            total_step_loss, pred = strategy.run(self.train_step, args=(x, y, visualise, ))
            return strategy.reduce(
                tf.distribute.ReduceOp.SUM, total_step_loss, axis=None), pred

        def run_test_strategy(x, y, visualise):
            total_step_loss, pred = strategy.run(self.test_step, args=(x, y, visualise, ))
            return strategy.reduce(
                tf.distribute.ReduceOp.SUM, total_step_loss, axis=None), pred

        def distributed_train_epoch(train_ds, epoch, strategy, num_to_visualise, multi_class, slice_writer, vol_writer, vol_visual_freq, predict_slice):
            total_loss, num_train_batch = 0.0, 0.0
            for x_train, y_train in train_ds:
                visualise = (num_train_batch < num_to_visualise)
                loss, pred = run_train_strategy(x_train, y_train, visualise)
                loss /= strategy.num_replicas_in_sync
                total_loss += loss
                if visualise:
                    # img = get_mid_slice(x_train, y_train, pred, multi_class)
                    img = get_mid_slice(x_train.values[0], y_train.values[0], pred.values[0], multi_class)
                    with slice_writer.as_default():
                        tf.summary.image("Train - Slice", img, step=epoch)
                    if epoch % vol_visual_freq == 0:
                        if not predict_slice:
                            # img = get_mid_vol(y_train, pred, multi_class)
                            img = get_mid_vol(y_train.values[0], pred.values[0], multi_class)
                            with vol_writer.as_default():
                                tf.summary.image("Train - Volume", img, step=epoch)
                num_train_batch += 1
            return total_loss / num_train_batch

        def distributed_test_epoch(valid_ds, epoch, strategy, num_to_visualise, multi_class, slice_writer, vol_writer, vol_visual_freq, predict_slice):
            total_loss, num_test_batch = 0.0, 0.0
            for x_valid, y_valid in valid_ds:
                visualise = (num_test_batch < num_to_visualise)
                loss, pred = run_test_strategy(x_valid, y_valid, visualise)
                loss /= strategy.num_replicas_in_sync
                total_loss += loss
                if visualise:
                    # img = get_mid_slice(x_valid, y_valid, pred, multi_class)
                    img = get_mid_slice(x_valid.values[0], y_valid.values[0], pred.values[0], multi_class)
                    with slice_writer.as_default():
                        tf.summary.image("Validation - Slice", img, step=epoch)
                    if epoch % vol_visual_freq == 0:
                        if not predict_slice:
                            # img = get_mid_vol(y_valid, pred, multi_class)
                            img = get_mid_vol(y_valid.values[0], pred.values[0], multi_class)
                            with vol_writer.as_default():
                                tf.summary.image("Validation - Volume", img, step=epoch)
                num_test_batch += 1
            return total_loss / num_test_batch

        if self.enable_function:
            run_train_strategy = tf.function(run_train_strategy)
            run_test_strategy = tf.function(run_test_strategy)

        name = "/vnet" if not self.predict_slice else "/vnet_slice"
        db = "/debug" if debug else "/test"
        mc = "/multi" if multi_class else "/binary"
        log_dir_now = self.log_dir + name + db + mc + datetime.datetime.now().strftime("/%Y%m%d/%H%M%S")
        train_summary_writer = tf.summary.create_file_writer(log_dir_now + '/train')
        test_summary_writer = tf.summary.create_file_writer(log_dir_now + '/validation')
        train_img_slice_writer = tf.summary.create_file_writer(log_dir_now + '/train/img/slice')
        test_img_slice_writer = tf.summary.create_file_writer(log_dir_now + '/validation/img/slice')
        train_img_vol_writer = tf.summary.create_file_writer(log_dir_now + '/train/img/vol')
        test_img_vol_writer = tf.summary.create_file_writer(log_dir_now + '/validation/img/vol')
        lr_summary_writer = tf.summary.create_file_writer(log_dir_now + '/lr')

        for e in range(self.epochs):
            self.optimizer.learning_rate = self.lr_manager.update_lr(e)

            et0 = time()

            train_loss = distributed_train_epoch(train_ds, e, strategy, num_to_visualise, multi_class, train_img_slice_writer, train_img_vol_writer, vol_visual_freq, self.predict_slice)

            with train_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', train_loss, step=e)

            test_loss = distributed_test_epoch(valid_ds, e, strategy, num_to_visualise, multi_class, test_img_slice_writer, test_img_vol_writer, vol_visual_freq, self.predict_slice)
            with test_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', test_loss, step=e)

            current_lr = self.optimizer.get_config()['learning_rate']
            with lr_summary_writer.as_default():
                tf.summary.scalar('epoch_lr', current_lr, step=e)
            print(f"Epoch {e+1}/{self.epochs} - {time() - et0:.0f}s - loss: {train_loss:.05f} - val_loss: {test_loss:.05f} - lr: {self.optimizer.get_config()['learning_rate']: .06f}")


def load_datasets(batch_size, buffer_size,
                  tfrec_dir='./Data/tfrecords/',
                  multi_class=False,
                  crop_size=144,
                  depth_crop_size=80,
                  aug=[],
                  predict_slice=False,
                  ):
    """
    Loads tf records datasets for 3D models.
    """
    args = {
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'multi_class': multi_class,
        'use_keras_fit': False,
        'crop_size': crop_size,
        'depth_crop_size': depth_crop_size,
        'aug': aug,
    }
    train_ds = read_tfrecord_3d(tfrecords_dir=os.path.join(tfrec_dir, 'train_3d/'),
                                is_training=True, predict_slice=predict_slice, **args)
    valid_ds = read_tfrecord_3d(tfrecords_dir=os.path.join(tfrec_dir, 'valid_3d/'),
                                is_training=False, predict_slice=predict_slice, **args)
    return train_ds, valid_ds


def build_model(num_channels, num_classes, **kwargs):
    """
    Builds standard vnet for 3D
    """
    model = VNet(num_channels, num_classes, **kwargs)
    return model


def main(epochs,
         batch_size=2,
         lr=1e-4,
         lr_drop=0.9,
         lr_drop_freq=5,
         lr_warmup=3,
         num_to_visualise=2,
         num_channels=4,
         buffer_size=4,
         enable_function=True,
         tfrec_dir='./Data/tfrecords/',
         multi_class=False,
         crop_size=144,
         depth_crop_size=80,
         aug=[],
         debug=False,
         predict_slice=False,
         tpu=False,
         **model_kwargs,
         ):
    t0 = time()

    if tpu:
        tfrec_dir = 'gs://oai-challenge-dataset/tfrecords'

    num_classes = 7 if multi_class else 1
    
    train_ds, valid_ds = load_datasets(batch_size, buffer_size, tfrec_dir, multi_class,
                                       crop_size=crop_size, depth_crop_size=depth_crop_size, aug=aug,
                                       predict_slice=predict_slice)

    num_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
    steps_per_epoch = len(glob(os.path.join(tfrec_dir, 'train_3d/*'))) / (batch_size)

    if tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='pit-tpu')
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    with strategy.scope():
        loss_func = tversky_loss if multi_class else dice_loss

        lr_manager = LearningRateUpdate(lr, lr_drop, lr_drop_freq, warmup=lr_warmup)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model = build_model(num_channels, num_classes, predict_slice=predict_slice, **model_kwargs)

        trainer = Train(epochs, batch_size, enable_function,
                        model, optimizer, loss_func, lr_manager, predict_slice, tfrec_dir=tfrec_dir)

        train_ds = strategy.experimental_distribute_dataset(train_ds)
        valid_ds = strategy.experimental_distribute_dataset(valid_ds)

        trainer.train_model_loop(train_ds, valid_ds, strategy, multi_class, debug, num_to_visualise)
        # tf.print("starting")
        # for idx, data in enumerate(train_ds):
        #     x, y = data
        #     print(type(x))
        #     print(type(y))
        #     print("==========================")
        #     # tf.print(idx, "x", tf.shape(x), tf.reduce_max(x))
        #     # tf.print(idx, "y", tf.shape(y), tf.reduce_sum(y))
        #     tf.print(idx, "y", tf.shape(y.values[0]), tf.reduce_sum(y.values[0]), tf.reduce_sum(y.values[1]))
        #     tf.print("=====================")
        #     # pred = model(x, training=True)
        #     # print("p", tf.shape(pred.values[0]))
        #     # model.summary()

    print(f"{time() - t0:.02f}")


if __name__ == "__main__":
    use_tpu = False
    if not use_tpu:
        setup_gpu()

    # a = tf.constant([100, 900])
    # tf.print(a)
    # b = a < 1000
    # tf.print(b)
    # c = tf.reduce_all(b)
    # tf.print(c)
    #tf.print(tf.reduce_sum(tf.constant([100, 1900]) < 1000))
    #tf.print(tf.reduce_all((tf.constant([100, 1900]) < 1000)))

    # def sq(x, idx, a):
    #     tf.print("===========")
    #     tf.print(x)
    #     tf.print(idx)
    #     tf.print(a)
    #     idx += 1
    #     tf.print(idx)
    #     return x*x

    # elems = np.array([1, 2, 3, 4])
    # counter = tf.([0])
    # squares = tf.map_fn(lambda x: sq(x, counter, "hello"), elems)
    # tf.print(squares)

    # a = tf.constant([[1,2,3],[4,5,6],[3,2,1]])
    # b = tf.constant([[True, False, True]], dtype=tf.bool)

    # c = tf.map_fn(lambda x: (x[0], x[1], x[2]), (a,b), dtype=(tf.int32, tf.bool))

    # tf.print(c)

    # def sq(idx, x, x2):
    #     tf.print(idx, x, x2)
    #     return idx, x*x, x2
    
    # # elems = np.array([3, 2, 6, 4])
    # #idx = np.array([0,1,2,3])
    # # idx = np.array(range(len(elems)))

    # elems = tf.constant([3, 2, 6, 4])
    # elems_2 = tf.constant([[2,23], [1,23], [4,16], [2,0]])
    # idx = tf.constant(range(len(elems)))

    # squares = tf.map_fn(lambda x: sq(x[0], x[1], x[2]), (idx, elems, elems_2))
    # tf.print(squares)



    es = 3
    main(epochs=es, lr=1e-4, dropout_rate=1e-5, use_spatial_dropout=False, use_batchnorm=False, noise=1e-5,
        crop_size=64, depth_crop_size=32, num_channels=16, lr_drop_freq=10,
        num_conv_layers=3, batch_size=2, multi_class=False, kernel_size=(3, 3, 3),
        aug=['resize'], use_transpose=False, debug=True, tpu=use_tpu)  # decent performance

    main(epochs=es, lr=1e-4, dropout_rate=1e-5, use_spatial_dropout=False, use_batchnorm=False, noise=1e-5,
        crop_size=64, depth_crop_size=32, num_channels=16, lr_drop_freq=10,
        num_conv_layers=3, batch_size=2, multi_class=False, kernel_size=(3, 3, 3),
        aug=['rotate'], use_transpose=False, debug=True, tpu=use_tpu)  # decent performance

    # main(epochs=es, lr=1e-4, dropout_rate=1e-5, use_spatial_dropout=False, use_batchnorm=False, noise=1e-5,
    #     crop_size=64, depth_crop_size=32, num_channels=16, lr_drop_freq=10,
    #     num_conv_layers=3, batch_size=2, multi_class=False, kernel_size=(3, 3, 3),
    #     aug=['flip', 'rotate', 'resize'], use_transpose=False, debug=True, tpu=use_tpu)  # decent performance

    # main(epochs=es, lr=1e-4, dropout_rate=1e-5, use_spatial_dropout=False, use_batchnorm=False, noise=1e-5,
    #     crop_size=64, depth_crop_size=32, num_channels=8, lr_drop_freq=5,
    #     num_conv_layers=3, batch_size=2, multi_class=True, kernel_size=(3, 3, 3),
    #     aug=['flip', 'rotate', 'resize'], use_transpose=True)  # just predicts background class

    # main(epochs=es, lr=1e-4, dropout_rate=1e-5, use_spatial_dropout=False, use_batchnorm=True, noise=1e-5,
    #      crop_size=120, depth_crop_size=64, num_channels=4, lr_drop_freq=10,
    #      num_conv_layers=3, batch_size=2, multi_class=False, kernel_size=(3, 3, 3),
    #      aug=['flip'], use_transpose=True)  # poor performance
