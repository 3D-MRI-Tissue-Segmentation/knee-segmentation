import os
import datetime
import tensorflow as tf
from time import time

from Segmentation.train.utils import Metric
from Segmentation.utils.data_loader import read_tfrecord_3d
from Segmentation.utils.visualise_utils import visualise_sample

class Trainer:
    def __init__(self,
                 epochs,
                 batch_size,
                 run_eager,
                 model,
                 optimizer,
                 loss_func,
                 lr_manager,
                 predict_slice,
                 metrics,
                 tfrec_dir='./Data/tfrecords/',
                 log_dir="logs"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.run_eager = run_eager
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.lr_manager = lr_manager
        self.predict_slice = predict_slice
        self.metrics = Metric(metrics)
        self.tfrec_dir = tfrec_dir
        self.log_dir = log_dir

    def train_step(self,
                   x_train,
                   y_train,
                   visualise):
        with tf.GradientTape() as tape:
            predictions = self.model(x_train, training=True)
            loss = self.loss_func(y_train, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.metrics.store_metric(y_train, predictions, training=True)
        if visualise:
            return loss, predictions
        return loss, None

    def test_step(self,
                  x_test,
                  y_test,
                  visualise):
        predictions = self.model(x_test, training=False)
        loss = self.loss_func(y_test, predictions)
        self.metrics.store_metric(y_test, predictions, training=False)
        if visualise:
            return loss, predictions
        return loss, None

    def train_model_loop(self,
                         train_ds,
                         valid_ds,
                         strategy,
                         multi_class,
                         visual_save_freq=5,
                         debug=False,
                         num_to_visualise=0):
        """
        Trains 3D model with custom tf loop and MirrorStrategy
        """

        def run_train_strategy(x, y, visualise):
            total_step_loss, pred = strategy.run(self.train_step, args=(x, y, visualise, ))
            return strategy.reduce(
                tf.distribute.ReduceOp.SUM, total_step_loss, axis=None), pred

        def run_test_strategy(x, y, visualise):
            total_step_loss, pred = strategy.run(self.test_step, args=(x, y, visualise, ))
            return strategy.reduce(
                tf.distribute.ReduceOp.SUM, total_step_loss, axis=None), pred

        # TODO(Joe): This needs to be rewritten so that it works with 2D as well
        def distributed_train_epoch(train_ds,
                                    epoch,
                                    strategy,
                                    num_to_visualise,
                                    multi_class,
                                    slice_writer,
                                    vol_writer,
                                    visual_save_freq,
                                    predict_slice):

            total_loss, num_train_batch = 0.0, 0.0
            is_training = True
            use_2d = False

            for x_train, y_train in train_ds:
                visualise = (num_train_batch < num_to_visualise)
                loss, pred = run_train_strategy(x_train, y_train, visualise)
                loss /= strategy.num_replicas_in_sync
                total_loss += loss
                if visualise:
                    # let's check if this works
                    num_to_visualise = visualise_sample(x_train,
                                                        y_train,
                                                        pred,
                                                        num_to_visualise,
                                                        slice_writer,
                                                        vol_writer,
                                                        use_2d,
                                                        epoch,
                                                        multi_class,
                                                        predict_slice,
                                                        is_training)
                num_train_batch += 1
            return total_loss / num_train_batch

        def distributed_test_epoch(valid_ds,
                                   epoch,
                                   strategy,
                                   num_to_visualise,
                                   multi_class,
                                   slice_writer,
                                   vol_writer,
                                   visual_save_freq,
                                   predict_slice):
            total_loss, num_test_batch = 0.0, 0.0
            is_training = False
            use_2d = False
            for x_valid, y_valid in valid_ds:
                visualise = (num_test_batch < num_to_visualise)
                loss, pred = run_test_strategy(x_valid, y_valid, visualise)
                loss /= strategy.num_replicas_in_sync
                total_loss += loss
                if visualise:
                    num_to_visualise = visualise_sample(x_valid,
                                                        y_valid,
                                                        pred,
                                                        num_to_visualise,
                                                        slice_writer,
                                                        vol_writer,
                                                        use_2d,
                                                        epoch,
                                                        multi_class,
                                                        predict_slice,
                                                        is_training)
                num_test_batch += 1
            return total_loss / num_test_batch

        if self.run_eager:
            run_train_strategy = tf.function(run_train_strategy)
            run_test_strategy = tf.function(run_test_strategy)

        # TODO: This whole chunk of code needs to be refactored. Perhaps write it as a function
        name = "/" + self.model.name
        db = "/debug" if debug else "/test"
        mc = "/multi" if multi_class else "/binary"
        log_dir_now = self.log_dir + name + db + mc + datetime.datetime.now().strftime("/%Y%m%d/%H%M%S")
        train_summary_writer = tf.summary.create_file_writer(log_dir_now + '/train')
        test_summary_writer = tf.summary.create_file_writer(log_dir_now + '/val')
        test_min_summary_writer = tf.summary.create_file_writer(log_dir_now + '/val_min')
        train_img_slice_writer = tf.summary.create_file_writer(log_dir_now + '/train/img/slice')
        test_img_slice_writer = tf.summary.create_file_writer(log_dir_now + '/val/img/slice')
        train_img_vol_writer = tf.summary.create_file_writer(log_dir_now + '/train/img/vol')
        test_img_vol_writer = tf.summary.create_file_writer(log_dir_now + '/val/img/vol')
        lr_summary_writer = tf.summary.create_file_writer(log_dir_now + '/lr')

        self.metrics.add_metric_summary_writer(log_dir_now)

        best_loss = None
        for e in range(self.epochs):
            self.optimizer.learning_rate = self.lr_manager.update_lr(e)

            et0 = time()

            train_loss = distributed_train_epoch(train_ds,
                                                 e,
                                                 strategy,
                                                 num_to_visualise,
                                                 multi_class,
                                                 train_img_slice_writer,
                                                 train_img_vol_writer,
                                                 visual_save_freq,
                                                 self.predict_slice)

            with train_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', train_loss, step=e)

            test_loss = distributed_test_epoch(valid_ds,
                                               e,
                                               strategy,
                                               num_to_visualise,
                                               multi_class,
                                               test_img_slice_writer,
                                               test_img_vol_writer,
                                               visual_save_freq,
                                               self.predict_slice)
            with test_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', test_loss, step=e)

            current_lr = self.optimizer.get_config()['learning_rate']
            with lr_summary_writer.as_default():
                tf.summary.scalar('epoch_lr', current_lr, step=e)

            self.metrics.record_metric_to_summary(e)
            metric_str = self.metrics.reset_metrics_get_str()
            print(f"Epoch {e+1}/{self.epochs} - {time() - et0:.0f}s - loss: {train_loss:.05f} - val_loss: {test_loss:.05f} - lr: {self.optimizer.get_config()['learning_rate']: .06f}" + metric_str)

            if best_loss is None:
                self.model.save_weights(os.path.join(log_dir_now + '/best_weights.tf'))
                best_loss = test_loss
            else:
                if test_loss < best_loss:
                    self.model.save_weights(os.path.join(log_dir_now + '/best_weights.tf'))
                    best_loss = test_loss
            with test_min_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', best_loss, step=e)
        return log_dir_now


