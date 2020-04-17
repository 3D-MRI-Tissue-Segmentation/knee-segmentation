import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime
from absl import app
from absl import flags
from absl import logging

from Segmentation.model.unet import UNet, AttentionUNet_v1, MultiResUnet
from Segmentation.utils.data_loader import read_tfrecord
from Segmentation.utils.training_utils import dice_coef, dice_coef_loss, tversky_loss, iou_loss_core, Mean_IOU
from Segmentation.utils.training_utils import plot_train_history_loss, visualise_multi_class, make_lr_scheduler, visualise_binary

# Dataset/training options
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('batch_size', 5, 'Batch size per TPU Core / GPU')
flags.DEFINE_float('base_learning_rate', 5e-05, 'base learning rate at the start of training session')
flags.DEFINE_string('dataset', 'oai_challenge', 'Dataset: oai_challenge, isic_2018 or oai_full')
flags.DEFINE_bool('2D', True, 'True to train on 2D slices, False to train on 3D data')
flags.DEFINE_bool('corruptions', False, 'Whether to test on corrupted dataset')
flags.DEFINE_integer('train_epochs', 5, 'Number of training epochs.')

# Model options
flags.DEFINE_string('model_architecture', 'unet', 'Model: unet (default), multires_unet, attention_unet_v1, R2_unet, R2_attention_unet')
flags.DEFINE_string('channel_order', 'channels_last', 'channels_last (Default) or channels_first')
flags.DEFINE_bool('multi_class', True, 'Whether to train on a multi-class (Default) or binary setting')
flags.DEFINE_bool('batchnorm', True, 'Whether to use batch normalisation')
flags.DEFINE_bool('use_spatial', False, 'Whether to use spatial Dropout')
flags.DEFINE_float('dropout_rate', 0.0, 'Dropout rate')
flags.DEFINE_string('activation', 'relu', 'activation function to be used')
flags.DEFINE_integer('buffer_size', 50, 'shuffle buffer size (default: 1000)')
flags.DEFINE_integer('respath_length', 2, 'residual path length')
flags.DEFINE_integer('kernel_size', 3, 'kernel size to be used')
flags.DEFINE_integer('num_conv', 2, 'number of convolution layers in each block')
flags.DEFINE_integer('num_filters', 32, 'number of filters in the model')

# Logging, saving and testing options
flags.DEFINE_string('tfrec_dir', './Data/tfrecords/', 'directory for TFRecords folder')
flags.DEFINE_string('log_dir', './checkpoints', 'directory for checkpoints')
flags.DEFINE_string('weights_dir', './checkpoints', 'directory for saved model or weights. Only used if train is False')
flags.DEFINE_bool('train', True, 'If True (Default), train the model. Otherwise, test the model')

# Accelerator flags
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 1, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', 'oai-tpu-machine', 'Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS

def main(argv):

    del argv  # unused arg
    tf.random.set_seed(FLAGS.seed)

    # set whether to train on GPU or TPU
    if FLAGS.use_gpu:
        logging.info('Using GPU...')
        strategy = tf.distribute.MirroredStrategy()
    else:
        logging.info('Use TPU at %s',
                     FLAGS.tpu if FLAGS.tpu is not None else 'local')
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)

    # set dataset configuration 
    if FLAGS.dataset == 'oai_challenge':
        
        batch_size = FLAGS.batch_size*FLAGS.num_cores
        steps_per_epoch = 19200 // batch_size
        validation_steps = 4480 // batch_size 

        train_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir, 'train/'),
                                 batch_size=batch_size,
                                 buffer_size=FLAGS.buffer_size,
                                 multi_class=FLAGS.multi_class,
                                 is_training=True)
        valid_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir, 'valid/'),
                                 batch_size=batch_size,
                                 buffer_size=FLAGS.buffer_size,
                                 multi_class=FLAGS.multi_class,
                                 is_training=False)

        num_classes = 7 if FLAGS.multi_class else 1

    crossentropy_loss_fn = tf.keras.losses.categorical_crossentropy if FLAGS.multi_class else tf.keras.losses.binary_crossentropy

    if FLAGS.use_bfloat16:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    # set model architecture
    with strategy.scope():
        if FLAGS.model_architecture == 'unet':
            model = UNet(FLAGS.num_filters,
                         num_classes,
                         FLAGS.num_conv,
                         FLAGS.kernel_size,
                         FLAGS.activation,
                         FLAGS.batchnorm,
                         FLAGS.dropout_rate,
                         FLAGS.use_spatial,
                         FLAGS.channel_order)

        elif FLAGS.model_architecture == 'multires_unet':
            model = MultiResUnet(FLAGS.num_filters,
                                 num_classes,
                                 FLAGS.respath_length,
                                 FLAGS.num_conv,
                                 FLAGS.kernel_size,
                                 use_bias=False,
                                 padding='same',
                                 nonlinearity=FLAGS.activation,
                                 use_batchnorm=FLAGS.batchnorm,
                                 use_transpose=True,
                                 data_format=FLAGS.channel_order)

        elif FLAGS.model_architecture == 'attention_unet_v1':
            model = AttentionUNet_v1(FLAGS.num_filters,
                                     num_classes,
                                     FLAGS.num_conv,
                                     FLAGS.kernel_size,
                                     use_bias=False,
                                     padding='same',
                                     nonlinearity=FLAGS.activation,
                                     use_batchnorm=FLAGS.batchnorm,
                                     use_transpose=True,
                                     data_format=FLAGS.channel_order)

        else:
            print("%s is not a valid or supported model architecture." % FLAGS.model_architecture)
            exit()

        optimiser = tf.keras.optimizers.Adam(learning_rate=FLAGS.base_learning_rate)

        model.compile(optimizer=optimiser,
                      loss=tversky_loss,
                      metrics=[dice_coef, crossentropy_loss_fn, 'acc'])

    if FLAGS.train:

        # define checkpoints
        logdir = 'checkpoints\\' + datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(FLAGS.logdir + '/' + FLAGS.model_architecture + '_weights.{epoch:03d}.ckpt',
                                                     save_best_only=False, save_weights_only=True)
        lr_schedule = make_lr_scheduler(FLAGS.base_learning_rate)
        tb = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch')

        history = model.fit(train_ds,
                            steps_per_epoch=steps_per_epoch,
                            epochs=FLAGS.train_epochs,
                            validation_data=valid_ds,
                            validation_steps=validation_steps,
                            callbacks=[ckpt_cb, lr_schedule, tb])

        plot_train_history_loss(history, multi_class=FLAGS.multi_class)

    else:
        # load the latest checkpoint in the FLAGS.logdir file
        # latest = tf.train.latest_checkpoint(FLAGS.logdir)
        model.load_weights('./checkpoints/unet/14-4-2020/unet_weights.005.ckpt').expect_partial()
        for step, (image, label) in enumerate(valid_ds):
            if step >= 80:
                pred = model(image, training=False)
                visualise_multi_class(label, pred)

if __name__ == '__main__':
    app.run(main)
