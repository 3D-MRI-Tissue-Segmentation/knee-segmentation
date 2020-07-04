import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path
from datetime import datetime
from absl import app
from absl import logging
from glob import glob

from Segmentation.model.unet import UNet, R2_UNet, Nested_UNet, Nested_UNet_v2
from Segmentation.model.segnet import SegNet
from Segmentation.model.deeplabv3 import Deeplabv3, Deeplabv3_plus
from Segmentation.model.vnet import VNet
from Segmentation.model.Hundred_Layer_Tiramisu import Hundred_Layer_Tiramisu
from Segmentation.utils.data_loader import read_tfrecord_2d as read_tfrecord
from Segmentation.utils.data_loader import parse_fn_3d
from Segmentation.utils.losses import dice_coef_loss, tversky_loss, dice_coef, iou_loss, focal_tversky
from Segmentation.utils.evaluation_metrics import dice_coef_eval, iou_loss_eval
from Segmentation.utils.training_utils import plot_train_history_loss, LearningRateSchedule
from Segmentation.utils.evaluation_utils import plot_and_eval_3D, confusion_matrix, epoch_gif, volume_gif, take_slice
from flags import FLAGS


def main(argv):

    if FLAGS.visual_file:
        assert FLAGS.train is False, "Train must be set to False if you are doing a visual."

    del argv  # unused arg
    # tf.random.set_seed(FLAGS.seed)

    # set whether to train on GPU or TPU
    if FLAGS.use_gpu:
        logging.info('Using GPU...')
        # strategy requires: export TF_FORCE_GPU_ALLOW_GROWTH=true to be wrote in cmd
        if FLAGS.num_cores == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.MirroredStrategy()  # works
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_visible_devices(gpu, 'GPU')
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')

                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
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
        steps_per_epoch = 19200 // batch_size
        validation_steps = 4480 // batch_size
        logging.info('Using Augmentation Strategy: {}'.format(FLAGS.aug_strategy))

        if FLAGS.model_architecture != 'vnet':
            train_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir, 'train/'),
                                     batch_size=batch_size,
                                     buffer_size=FLAGS.buffer_size,
                                     augmentation=FLAGS.aug_strategy,
                                     multi_class=FLAGS.multi_class,
                                     is_training=True,
                                     use_bfloat16=FLAGS.use_bfloat16,
                                     use_RGB=False if FLAGS.backbone_architecture == 'default' else True)
            valid_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir, 'valid/'),
                                     batch_size=batch_size,
                                     buffer_size=FLAGS.buffer_size,
                                     augmentation=FLAGS.aug_strategy,
                                     multi_class=FLAGS.multi_class,
                                     is_training=False,
                                     use_bfloat16=FLAGS.use_bfloat16,
                                     use_RGB=False if FLAGS.backbone_architecture == 'default' else True)
        else:
            train_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir, 'train_3d/'),
                                     batch_size=batch_size,
                                     buffer_size=FLAGS.buffer_size,
                                     augmentation=FLAGS.aug_strategy,
                                     parse_fn=parse_fn_3d,
                                     multi_class=FLAGS.multi_class,
                                     is_training=True,
                                     use_bfloat16=FLAGS.use_bfloat16,
                                     use_RGB=False)

            valid_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir, 'valid_3d/'),
                                     batch_size=batch_size,
                                     buffer_size=FLAGS.buffer_size,
                                     augmentation=FLAGS.aug_strategy,
                                     multi_class=FLAGS.multi_class,
                                     is_training=False,
                                     use_bfloat16=FLAGS.use_bfloat16,
                                     use_RGB=False)

        num_classes = 7 if FLAGS.multi_class else 1 

        # for step, (image, label) in enumerate(train_ds):
        #     print(step)
        #     print(image.shape)
        #     print(label.shape)

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
    model_fn = None
    model_args = []

    if FLAGS.model_architecture == 'unet':
        model_args = [FLAGS.num_filters,
                      num_classes,
                      FLAGS.backbone_architecture,
                      FLAGS.num_conv,
                      FLAGS.kernel_size,
                      FLAGS.activation,
                      FLAGS.use_attention,
                      FLAGS.use_batchnorm,
                      FLAGS.use_bias,
                      FLAGS.use_dropout,
                      FLAGS.dropout_rate,
                      FLAGS.use_spatial,
                      FLAGS.channel_order]

        model_fn = UNet
    elif FLAGS.model_architecture == 'vnet':
        model_args = [64,
                      num_classes,
                      FLAGS.num_conv,
                      FLAGS.kernel_size,
                      FLAGS.activation,
                      FLAGS.use_batchnorm,
                      FLAGS.dropout_rate,
                      FLAGS.use_spatial,
                      FLAGS.channel_order]

        model_fn = VNet
    elif FLAGS.model_architecture == 'r2unet':
        model_args = [FLAGS.num_filters,
                      num_classes,
                      FLAGS.num_conv,
                      FLAGS.kernel_size,
                      FLAGS.activation,
                      2,
                      FLAGS.use_attention,
                      FLAGS.use_batchnorm,
                      FLAGS.use_bias,
                      FLAGS.channel_order]

        model_fn = R2_UNet

    elif FLAGS.model_architecture == 'segnet':
        model_args = [FLAGS.num_filters,
                      num_classes,
                      FLAGS.backbone_architecture,
                      FLAGS.kernel_size,
                      (2, 2),
                      FLAGS.activation,
                      FLAGS.use_batchnorm,
                      FLAGS.use_bias,
                      FLAGS.use_transpose,
                      FLAGS.use_dropout,
                      FLAGS.dropout_rate,
                      FLAGS.use_spatial,
                      FLAGS.channel_order]

        model_fn = SegNet

    elif FLAGS.model_architecture == 'unet++':
        model_args = [FLAGS.num_filters,
                      num_classes,
                      FLAGS.num_conv,
                      FLAGS.kernel_size,
                      FLAGS.activation,
                      FLAGS.use_batchnorm,
                      FLAGS.use_bias,
                      FLAGS.channel_order]

        model_fn = Nested_UNet

    elif FLAGS.model_architecture == '100-Layer-Tiramisu':
        model_args = [FLAGS.growth_rate,
                      FLAGS.layers_per_block,
                      FLAGS.init_num_channels,
                      num_classes,
                      FLAGS.kernel_size,
                      FLAGS.pool_size,
                      FLAGS.activation,
                      FLAGS.dropout_rate,
                      FLAGS.strides,
                      FLAGS.padding]

        model_fn = Hundred_Layer_Tiramisu

    elif FLAGS.model_architecture == 'deeplabv3':
        model_args = [num_classes,
                      FLAGS.kernel_size_initial_conv,
                      FLAGS.num_filters_atrous,
                      FLAGS.num_filters_DCNN,
                      FLAGS.num_filters_ASPP,
                      FLAGS.kernel_size_atrous,
                      FLAGS.kernel_size_DCNN,
                      FLAGS.kernel_size_ASPP,
                      'same',
                      FLAGS.activation,
                      FLAGS.use_batchnorm,
                      FLAGS.use_bias,
                      FLAGS.channel_order,
                      FLAGS.MultiGrid,
                      FLAGS.rate_ASPP,
                      FLAGS.output_stride]

        model_fn = Deeplabv3

    elif FLAGS.model_architecture == 'deeplabv3_plus':
        model_args = [num_classes,
                      FLAGS.kernel_size_initial_conv,
                      FLAGS.num_filters_atrous,
                      FLAGS.num_filters_DCNN,
                      FLAGS.num_filters_ASPP,
                      FLAGS.kernel_size_atrous,
                      FLAGS.kernel_size_DCNN, 
                      FLAGS.kernel_size_ASPP,
                      FLAGS.num_filters_final_encoder,
                      FLAGS.num_filters_from_backbone,
                      FLAGS.num_channels_UpConv,
                      FLAGS.kernel_size_UpConv,
                      (2, 2),
                      False,
                      FLAGS.use_transpose,
                      'same',
                      FLAGS.activation,
                      FLAGS.use_batchnorm,
                      FLAGS.use_bias,
                      FLAGS.channel_order,
                      FLAGS.MultiGrid,
                      FLAGS.rate_ASPP,
                      FLAGS.output_stride]

        model_fn = Deeplabv3_plus

    else:
        logging.error('The model architecture {} is not supported!'.format(FLAGS.model_architecture))

    with strategy.scope():
        model = model_fn(*model_args)

        if FLAGS.custom_decay_lr:
            lr_decay_epochs = FLAGS.lr_decay_epochs
        else:
            lr_decay_epochs = list(range(FLAGS.lr_warmup_epochs + 1, FLAGS.train_epochs))

        lr_rate = LearningRateSchedule(steps_per_epoch,
                                       FLAGS.base_learning_rate,
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
            if FLAGS.model_architecture != 'vnet':
                model.compile(optimizer=optimiser,
                              loss=loss_fn,
                              metrics=[dice_coef, iou_loss, dice_coef_eval, iou_loss_eval, crossentropy_loss_fn, 'acc'])
            else:
                model.compile(optimizer=optimiser,
                              loss=loss_fn,
                              metrics=[dice_coef, iou_loss, crossentropy_loss_fn, 'acc'])
        else:
            model.compile(optimizer=optimiser,
                          loss=loss_fn,
                          metrics=[dice_coef, iou_loss, crossentropy_loss_fn, 'acc'])

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

        plot_train_history_loss(history, multi_class=FLAGS.multi_class, savefig=training_history_dir)
    elif not FLAGS.visual_file == "":
        tpu = FLAGS.tpu_dir if FLAGS.tpu_dir else FLAGS.tpu
        print('model_fn', model_fn)

        if not FLAGS.which_representation == '':

            if FLAGS.which_representation == 'volume':
                volume_gif(model=model_fn,
                           logdir=FLAGS.logdir,
                           tfrecords_dir=os.path.join(FLAGS.tfrec_dir, 'valid/'),
                           aug_strategy=FLAGS.aug_strategy,
                           visual_file=FLAGS.visual_file,
                           tpu_name=FLAGS.tpu_dir,
                           bucket_name=FLAGS.bucket,
                           weights_dir=FLAGS.weights_dir,
                           is_multi_class=FLAGS.multi_class,
                           model_args=model_args,
                           which_epoch=FLAGS.gif_epochs,
                           which_volume=FLAGS.gif_volume,
                           gif_dir=FLAGS.gif_directory,
                           gif_cmap=FLAGS.gif_cmap,
                           clean=FLAGS.clean_gif)

            elif FLAGS.which_representation == 'epoch':
                epoch_gif(model=model_fn,
                          logdir=FLAGS.logdir,
                          tfrecords_dir=os.path.join(FLAGS.tfrec_dir, 'valid/'),
                          aug_strategy=FLAGS.aug_strategy,
                          visual_file=FLAGS.visual_file,
                          tpu_name=FLAGS.tpu_dir,
                          bucket_name=FLAGS.bucket,
                          weights_dir=FLAGS.weights_dir,
                          is_multi_class=FLAGS.multi_class,
                          model_args=model_args,
                          which_slice=FLAGS.gif_slice,
                          which_volume=FLAGS.gif_volume,
                          epoch_limit=FLAGS.gif_epochs,
                          gif_dir=FLAGS.gif_directory,
                          gif_cmap=FLAGS.gif_cmap,
                          clean=FLAGS.clean_gif)

            elif FLAGS.which_representation == 'slice':
                take_slice(model=model_fn,
                           logdir=FLAGS.logdir,
                           tfrecords_dir=os.path.join(FLAGS.tfrec_dir, 'valid/'),
                           aug_strategy=FLAGS.aug_strategy,
                           visual_file=FLAGS.visual_file,
                           tpu_name=FLAGS.tpu_dir,
                           bucket_name=FLAGS.bucket,
                           weights_dir=FLAGS.weights_dir,
                           multi_as_binary=False,
                           is_multi_class=FLAGS.multi_class,
                           model_args=model_args,
                           which_epoch=FLAGS.gif_epochs,
                           which_slice=FLAGS.gif_slice,
                           which_volume=FLAGS.gif_volume,
                           save_dir=FLAGS.gif_directory,
                           cmap=FLAGS.gif_cmap,
                           clean=FLAGS.clean_gif)
            else:
                print("The 'which_representation' flag does not match any of the options, try either 'volume', 'epoch' or 'slice'")

        else:
            plot_and_eval_3D(model=model_fn,
                             logdir=FLAGS.logdir,
                             visual_file=FLAGS.visual_file,
                             tpu_name=tpu,
                             bucket_name=FLAGS.bucket,
                             weights_dir=FLAGS.weights_dir,
                             is_multi_class=FLAGS.multi_class,
                             dataset=valid_ds,
                             save_freq=FLAGS.save_freq,
                             model_args=model_args)

    else:
        # load the checkpoint in the FLAGS.weights_dir file
        # maybe_weights = os.path.join(FLAGS.weights_dir, FLAGS.tpu, FLAGS.visual_file)

        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join(FLAGS.logdir, FLAGS.tpu)
        logdir = os.path.join(logdir, time)
        tb = tf.keras.callbacks.TensorBoard(logdir, update_freq='epoch', write_images=True)
        confusion_matrix(trained_model=model,
                         weights_dir=FLAGS.weights_dir,
                         fig_dir=FLAGS.fig_dir,
                         dataset=valid_ds,
                         validation_steps=validation_steps,
                         multi_class=FLAGS.multi_class,
                         model_architecture=FLAGS.model_architecture,
                         callbacks=[tb],
                         num_classes=num_classes
                         )

if __name__ == '__main__':
    app.run(main)
