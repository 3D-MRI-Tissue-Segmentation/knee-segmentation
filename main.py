import tensorflow as tf

import os
from pathlib import Path
from datetime import datetime
from absl import app
from absl import logging

from Segmentation.utils.data_loader import read_tfrecord
from Segmentation.utils.data_loader import parse_fn_2d, parse_fn_3d
from Segmentation.utils.losses import dice_coef_loss, tversky_loss, dice_coef, iou_loss  # focal_tversky
from Segmentation.utils.evaluation_metrics import dice_coef_eval, iou_loss_eval
from Segmentation.utils.training_utils import LearningRateSchedule
# from Segmentation.utils.evaluation_utils import plot_and_eval_3D, confusion_matrix, epoch_gif, volume_gif, take_slice
from Segmentation.utils.evaluation_utils import eval_loop
from Segmentation.train.train import Train

from flags import FLAGS
from select_model import select_model


def main(argv):

    if FLAGS.visual_file:
        assert FLAGS.train is False, "Train must be set to False if you are doing a visual."

    del argv  # unused arg
    tf.random.set_seed(FLAGS.seed)  # set seed

    # set whether to train on GPU or TPU
    if FLAGS.use_gpu:
        logging.info('Using GPU...')
        # strategy requires: export TF_FORCE_GPU_ALLOW_GROWTH=true to be wrote in cmd
        if FLAGS.num_cores == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.MirroredStrategy()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_visible_devices(gpu, 'GPU')
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')

                    logging.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
                except RuntimeError as e:
                    # Visible devices must be set before GPUs have been initialized
                    print(e)
    else:
        logging.info('Use TPU at %s',
                     FLAGS.tpu if FLAGS.tpu is not None else 'local')
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)

    # set dataset configuration
    if FLAGS.dataset == 'oai_challenge':

        batch_size = FLAGS.batch_size * FLAGS.num_cores

        if FLAGS.use_2d:
            steps_per_epoch = 19200 // batch_size
            validation_steps = 4480 // batch_size
        else:
            steps_per_epoch = 120 // batch_size
            validation_steps = 28 // batch_size

        logging.info('Using Augmentation Strategy: {}'.format(FLAGS.aug_strategy))

        train_dir = 'train/' if FLAGS.use_2d else 'train_3d/'
        valid_dir = 'valid/' if FLAGS.use_2d else 'valid_3d/'

        ds_args = {
            'batch_size': batch_size,
            'buffer_size': FLAGS.buffer_size,
            'augmentation': FLAGS.aug_strategy,
            'parse_fn': parse_fn_2d if FLAGS.use_2d else parse_fn_3d,
            'multi_class': FLAGS.multi_class,
            'is_training': True,
            'use_bfloat16': FLAGS.use_bfloat16,
            'use_RGB': False if FLAGS.backbone_architecture == 'default' else True
        }

        train_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir, train_dir),
                                 **ds_args)
        valid_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir, valid_dir),
                                 **ds_args)

        num_classes = 7 if FLAGS.multi_class else 1

    if FLAGS.multi_class:
        loss_fn = tversky_loss
        crossentropy_loss_fn = tf.keras.losses.categorical_crossentropy
    else:
        loss_fn = dice_coef_loss
        crossentropy_loss_fn = tf.keras.losses.binary_crossentropy

    if FLAGS.use_bfloat16:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    # set model architecture

    model_fn, model_args = select_model(FLAGS, num_classes)

    with strategy.scope():
        model = model_fn(*model_args)

        if FLAGS.custom_decay_lr:
            lr_decay_epochs = FLAGS.lr_decay_epochs
        else:
            lr_decay_epochs = list(range(FLAGS.lr_warmup_epochs + 1, FLAGS.train_epochs))

        lr_rate = LearningRateSchedule(steps_per_epoch,
                                       FLAGS.base_learning_rate,
                                       FLAGS.min_learning_rate,
                                       FLAGS.lr_drop_ratio,
                                       lr_decay_epochs,
                                       FLAGS.lr_warmup_epochs)

        if FLAGS.optimizer == 'adam':
            optimiser = tf.keras.optimizers.Adam(learning_rate=lr_rate)
        elif FLAGS.optimizer == 'rms-prop':
            optimiser = tf.keras.optimizers.RMSprop(learning_rate=lr_rate)
        elif FLAGS.optimizer == 'sgd':
            optimiser = tf.keras.optimizers.SGD(learning_rate=lr_rate)
        else:
            print('Not a valid input optimizer, using Adam.')
            optimiser = tf.keras.optimizers.Adam(learning_rate=lr_rate)

        # for some reason, if i build the model then it can't load checkpoints. I'll see what I can do about this
        if FLAGS.train:
            if FLAGS.model_architecture != 'vnet':
                if FLAGS.backbone_architecture == 'default':
                    model.build((None, 288, 288, 1))
                else:
                    model.build((None, 288, 288, 3))
            else:
                model.build((None, 160, 384, 384, 1))
            model.summary()

        if FLAGS.multi_class:
            if FLAGS.use_2d:
                metrics = [dice_coef, iou_loss, dice_coef_eval, iou_loss_eval, crossentropy_loss_fn, 'acc']
            else:
                metrics = [dice_coef, iou_loss, crossentropy_loss_fn, 'acc']
        else:
            metrics = [dice_coef, iou_loss, crossentropy_loss_fn, 'acc']

        model.compile(optimizer=optimiser,
                        loss=loss_fn,
                        metrics=metrics)

    if FLAGS.train:
        # define checkpoints
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        training_history_dir = os.path.join(FLAGS.fig_dir, FLAGS.tpu)
        training_history_dir = os.path.join(training_history_dir, time)
        Path(training_history_dir).mkdir(parents=True, exist_ok=True)
        flag_name = os.path.join(training_history_dir, 'test_flags.cfg')
        FLAGS.append_flags_into_file(flag_name)

        logdir = os.path.join(FLAGS.logdir, FLAGS.tpu)
        logdir = os.path.join(logdir, time)
        logdir_arch = os.path.join(logdir, FLAGS.model_architecture)
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(logdir_arch + '_weights.{epoch:03d}.ckpt',
                                                     save_best_only=False,
                                                     save_weights_only=True)
        tb = tf.keras.callbacks.TensorBoard(logdir, update_freq='epoch')

        history = model.fit(train_ds,
                            steps_per_epoch=steps_per_epoch,
                            epochs=FLAGS.train_epochs,
                            validation_data=valid_ds,
                            validation_steps=validation_steps,
                            callbacks=[ckpt_cb, tb])
        
        """
        lr_manager = LearningRateSchedule(steps_per_epoch=steps_per_epoch,
                                          initial_learning_rate=FLAGS.base_learning_rate,
                                          drop=FLAGS.lr_drop_ratio,
                                          epochs_drop=FLAGS.lr_decay_epochs,
                                          warmup_epochs=FLAGS.lr_warmup_epochs)
        
        train = Train(epochs=FLAGS.train_epochs,
                      batch_size=FLAGS.batch_size,
                      enable_function=True,
                      model=model,
                      optimizer=optimiser,
                      loss_func=loss_fn,
                      lr_manager=lr_manager,
                      predict_slice=FLAGS.which_slice,
                      metrics=metrics,
                      tfrec_dir='./Data/tfrecords/',
                      log_dir="logs")
    
        log_dir_now = train.train_model_loop(train_ds=train_ds,
                                             valid_ds=valid_ds,
                                             strategy=strategy,
                                             visual_save_freq=FLAGS.visual_save_freq,
                                             multi_class=FLAGS.multi_class,
                                             debug=False,
                                             num_to_visualise=0)
        
        """
    elif FLAGS.visual_file is not None:
        tpu = FLAGS.tpu_dir if FLAGS.tpu_dir else FLAGS.tpu

        eval_loop(dataset=valid_ds, validation_steps=validation_steps, aug_strategy=FLAGS.aug_strategy,
                  bucket_name=FLAGS.bucket, logdir=FLAGS.logdir, tpu_name=tpu, visual_file=FLAGS.visual_file, weights_dir=FLAGS.weights_dir,
                  fig_dir=FLAGS.fig_dir,
                  which_volume=FLAGS.gif_volume, which_epoch=FLAGS.gif_epochs, which_slice=FLAGS.gif_slice,
                  multi_as_binary=False,
                  trained_model=model, model_architecture=FLAGS.model_architecture,
                  callbacks=[tb],
                  num_classes=num_classes)

    else:
        # load the checkpoint in the FLAGS.weights_dir file
        # maybe_weights = os.path.join(FLAGS.weights_dir, FLAGS.tpu, FLAGS.visual_file)

        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join(FLAGS.logdir, FLAGS.tpu)
        logdir = os.path.join(logdir, time)
        tb = tf.keras.callbacks.TensorBoard(logdir, update_freq='epoch', write_images=True)
        # confusion_matrix(trained_model=model,
        #                  weights_dir=FLAGS.weights_dir,
        #                  fig_dir=FLAGS.fig_dir,
        #                  dataset=valid_ds,
        #                  validation_steps=validation_steps,
        #                  multi_class=FLAGS.multi_class,
        #                  model_architecture=FLAGS.model_architecture,
        #                  callbacks=[tb],
        #                  num_classes=num_classes
        #                  )


if __name__ == '__main__':
    app.run(main)
