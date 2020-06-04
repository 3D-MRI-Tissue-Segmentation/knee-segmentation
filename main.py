import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime
from absl import app
from absl import flags
from absl import logging

from Segmentation.model.unet import UNet, R2_UNet, Nested_UNet
from Segmentation.model.segnet import SegNet
from Segmentation.utils.data_loader import read_tfrecord
from Segmentation.utils.training_utils import dice_coef, dice_coef_loss, tversky_loss, tversky_loss_v2
from Segmentation.utils.training_utils import plot_train_history_loss, visualise_multi_class, LearningRateSchedule

# Dataset/training options
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('batch_size', 32, 'Batch size per TPU Core / GPU')
flags.DEFINE_float('base_learning_rate', 3.2e-04, 'base learning rate at the start of training session')
flags.DEFINE_integer('lr_warmup_epochs', 1, 'No. of epochs for a warmup to the base_learning_rate. 0 for no warmup')
flags.DEFINE_float('lr_drop_ratio', 0.8, 'Amount to decay the learning rate')
flags.DEFINE_bool('custom_decay_lr', False, 'Whether to specify epochs to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', None, 'Epochs to decay the learning rate by. Only used if custom_decay_lr is True')
flags.DEFINE_string('dataset', 'oai_challenge', 'Dataset: oai_challenge, isic_2018 or oai_full')
flags.DEFINE_bool('2D', True, 'True to train on 2D slices, False to train on 3D data')
flags.DEFINE_bool('corruptions', False, 'Whether to test on corrupted dataset')
flags.DEFINE_integer('train_epochs', 50, 'Number of training epochs.')

# Model options
flags.DEFINE_string('model_architecture', 'unet', 'unet, r2unet, segnet, unet++')
flags.DEFINE_string('backbone_architecture', 'default', 'default, vgg16, vgg19, resnet50, resnet101, resnet152')
flags.DEFINE_string('channel_order', 'channels_last', 'channels_last (Default) or channels_first')
flags.DEFINE_bool('multi_class', True, 'Whether to train on a multi-class (Default) or binary setting')
flags.DEFINE_bool('use_batchnorm', True, 'Whether to use batch normalisation')
flags.DEFINE_bool('use_bias', True, 'Wheter to use bias')
flags.DEFINE_bool('use_spatial', False, 'Whether to use spatial Dropout')
flags.DEFINE_bool('use_transpose', False, 'Whether to use transposed convolution or upsampling + convolution')
flags.DEFINE_bool('use_attention', False, 'Whether to use attention mechanism')
flags.DEFINE_bool('use_dropout', False, 'Whether to use dropout')
flags.DEFINE_float('dropout_rate', 0.0, 'Dropout rate. Only used if use_dropout is True')
flags.DEFINE_string('activation', 'relu', 'activation function to be used')
flags.DEFINE_integer('buffer_size', 5000, 'shuffle buffer size')
flags.DEFINE_integer('kernel_size', 3, 'kernel size to be used')
flags.DEFINE_integer('num_conv', 2, 'number of convolution layers in each block')
flags.DEFINE_list('num_filters', [64, 128, 256, 512, 1024], 'number of filters in the model')

# Logging, saving and testing options
flags.DEFINE_string('tfrec_dir', './Data/tfrecords/', 'directory for TFRecords folder')
flags.DEFINE_string('logdir', 'checkpoints', 'directory for checkpoints')
flags.DEFINE_string('weights_dir', 'checkpoints', 'directory for saved model or weights. Only used if train is False')
flags.DEFINE_bool('train', True, 'If True (Default), train the model. Otherwise, test the model')

# Accelerator flags
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', 'oai-tpu-machine', 'Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS

def main(argv):

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

        train_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir, 'train/'),
                                 batch_size=batch_size,
                                 buffer_size=FLAGS.buffer_size,
                                 multi_class=FLAGS.multi_class,
                                 is_training=True,
                                 use_bfloat16=FLAGS.use_bfloat16,
                                 use_RGB=False if FLAGS.backbone_architecture == 'default' else True)
        valid_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir, 'valid/'),
                                 batch_size=batch_size,
                                 buffer_size=FLAGS.buffer_size,
                                 multi_class=FLAGS.multi_class,
                                 is_training=False,
                                 use_bfloat16=FLAGS.use_bfloat16,
                                 use_RGB=False if FLAGS.backbone_architecture == 'default' else True)

        num_classes = 7 if FLAGS.multi_class else 1

    if FLAGS.multi_class:
        crossentropy_loss_fn = tf.keras.losses.categorical_crossentropy
    else:
        crossentropy_loss_fn = tf.keras.losses.binary_crossentropy

    if FLAGS.use_bfloat16:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    # set model architecture
    with strategy.scope():

        if FLAGS.model_architecture == 'unet':

            model = UNet(FLAGS.num_filters,
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
                         FLAGS.channel_order)

        elif FLAGS.model_architecture == 'r2unet':

            model = R2_UNet(FLAGS.num_filters,
                            num_classes,
                            FLAGS.num_conv,
                            FLAGS.kernel_size,
                            FLAGS.activation,
                            2,
                            FLAGS.use_attention,
                            FLAGS.use_batchnorm,
                            FLAGS.use_bias,
                            FLAGS.channel_order)

        elif FLAGS.model_architecture == 'segnet':

            model = SegNet(FLAGS.num_filters,
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
                           FLAGS.channel_order)

        elif FLAGS.model_architecture == 'unet++':

            model = Nested_UNet(FLAGS.num_filters,
                                num_classes,
                                FLAGS.num_conv,
                                FLAGS.kernel_size,
                                FLAGS.activation,
                                FLAGS.use_batchnorm,
                                FLAGS.use_bias,
                                FLAGS.channel_order)

        else:
            logging.error('The model architecture {} is not supported!'.format(FLAGS.model_architecture))

        if FLAGS.custom_decay_lr:
            lr_decay_epochs = FLAGS.lr_decay_epochs
        else:
            lr_decay_epochs = list(range(FLAGS.lr_warmup_epochs + 1, FLAGS.train_epochs))

        lr_rate = LearningRateSchedule(steps_per_epoch,
                                       FLAGS.base_learning_rate,
                                       FLAGS.lr_drop_ratio,
                                       lr_decay_epochs,
                                       FLAGS.lr_warmup_epochs)
        optimiser = tf.keras.optimizers.Adam(learning_rate=lr_rate)
        if FLAGS.backbone_architecture == 'default':
            model.build((None, None, None, 1))
        else:
            model.build((None, None, None, 3))
        model.summary()
        model.compile(optimizer=optimiser,
                      loss=tversky_loss,
                      metrics=[dice_coef, crossentropy_loss_fn, 'acc'])

    if FLAGS.train:

        # define checkpoints
        logdir = os.path.join(FLAGS.logdir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        logdir_arch = os.path.join(logdir, FLAGS.model_architecture)
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(logdir_arch + '_weights.{epoch:03d}.hdf5',
                                                     save_best_only=False,
                                                     save_weights_only=True)
        tb = tf.keras.callbacks.TensorBoard(logdir, update_freq='epoch')

        history = model.fit(train_ds,
                            steps_per_epoch=steps_per_epoch,
                            epochs=FLAGS.train_epochs,
                            validation_data=valid_ds,
                            validation_steps=validation_steps,
                            callbacks=[ckpt_cb, tb])
        FLAGS.append_flags_into_file(logdir_arch + '_test_flags.cfg')
        plot_train_history_loss(history, multi_class=FLAGS.multi_class)

    else:
        # load the latest checkpoint in the FLAGS.logdir file
        model.load_weights(FLAGS.weights_dir)
        for step, (image, label) in enumerate(valid_ds):
            if step >= 80:
                pred = model(image, training=False)
                visualise_multi_class(label, pred)

if __name__ == '__main__':
    app.run(main)
