import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import time
from absl import app
from absl import flags
from absl import logging

from Segmentation.model.unet import UNet, AttentionUNet_v1, MultiResUnet
from Segmentation.tests.test_unet import UNetTest
from Segmentation.utils.data_loader import dataset_generator, get_multiclass
from Segmentation.utils.training_utils import dice_coef, jaccard_distance_loss, dice_coef_loss, tversky_loss
from Segmentation.utils.training_utils import visualise_multi_class, LearningRateSchedule

# Dataset/training options
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('batch_size', 5, 'Batch size per GPU')
flags.DEFINE_float('base_learning_rate', 5e-05, 'base learning rate at the start of training session')
flags.DEFINE_string('dataset', 'oai_challenge', 'Dataset: oai_challenge, isic_2018 or oai_full')
flags.DEFINE_bool('2D', True, 'True to train on 2D slices, False to train on 3D data')
flags.DEFINE_bool('corruptions', False, 'Whether to test on corrupted dataset')
flags.DEFINE_integer('train_epochs', 2, 'Number of training epochs.')

# Model options
flags.DEFINE_string('model_architecture', 'unet', 'Model: unet (default), multires_unet, attention_unet_v1, R2_unet, R2_attention_unet')
flags.DEFINE_string('channel_order', 'channels_last', 'channels_last (Default) or channels_first')
flags.DEFINE_bool('batchnorm', True, 'Whether to use batch normalisation')
flags.DEFINE_bool('use_spatial', False, 'Whether to use spatial Dropout')
flags.DEFINE_float('dropout_rate', 0.0, 'Dropout rate')
flags.DEFINE_string('activation', 'relu', 'activation function to be used')
flags.DEFINE_integer('respath_length', 2, 'residual path length')
flags.DEFINE_integer('kernel_size', 3, 'kernel size to be used')
flags.DEFINE_integer('num_conv', 2, 'number of convolution layers in each block')
flags.DEFINE_integer('num_filters', 64, 'number of filters in the model')
flags.DEFINE_integer('num_classes', 7, 'number of classes: 1 for binary (default) and 7 for multi-class')

# Logging, saving and testing options
flags.DEFINE_string('logdir', './checkpoints/', 'directory for checkpoints')
flags.DEFINE_bool('train', True, 'If True (Default), train the model. Otherwise, test the model')

FLAGS = flags.FLAGS

def main(argv):

    del argv

    summary_writer = tf.summary.create_file_writer(os.path.join(FLAGS.logdir, 'summaries'))

    # select mode architecture
    if FLAGS.model_architecture == 'unet':
        model = UNet(FLAGS.num_filters,
                     FLAGS.num_classes,
                     FLAGS.num_conv,
                     FLAGS.kernel_size,
                     FLAGS.activation,
                     FLAGS.batchnorm,
                     FLAGS.dropout_rate,
                     FLAGS.use_spatial,
                     FLAGS.channel_order)

    elif FLAGS.model_architecture == 'multires_unet':
        model = MultiResUnet(FLAGS.num_filters,
                             FLAGS.num_classes,
                             FLAGS.res_path_length,
                             FLAGS.num_conv,
                             FLAGS.kernel_size,
                             use_bias=False,
                             padding='same',
                             activation=FLAGS.activation,
                             use_batchnorm=FLAGS.batchnorm,
                             use_transpose=True,
                             data_format=FLAGS.channel_order)

    elif FLAGS.model_architecture == 'attention_unet_v1':
        model = AttentionUNet_v1(FLAGS.num_filters,
                                 FLAGS.num_classes,
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

    lr_schedule = LearningRateSchedule(19200 / FLAGS.batch_size,
                                       FLAGS.base_learning_rate,
                                       0.5,
                                       1)
    optimiser = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    metrics = {
        'train/tversky_loss': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.CategoricalAccuracy(),
        'train/dice_coef': tf.keras.metrics.Mean(),
        'train/cce': tf.keras.metrics.CategoricalCrossentropy(),
        'valid/tversky_loss': tf.keras.metrics.Mean(),
        'valid/accuracy': tf.keras.metrics.CategoricalAccuracy(),
        'valid/dice_coef': tf.keras.metrics.Mean(),
        'valid/cce': tf.keras.metrics.CategoricalCrossentropy()
    }

    train_ds = dataset_generator('./Data/train_2d/', batch_size=FLAGS.batch_size, shuffle=True)
    valid_ds = dataset_generator('./Data/valid_2d/', batch_size=FLAGS.batch_size, shuffle=False)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            tversky = tf.reduce_mean(tversky_loss(labels, predictions))
            dice = tf.reduce_mean(dice_coef(labels, predictions))
        gradients = tape.gradient(tversky, model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))

        metrics['train/tversky_loss'].update_state(tversky)
        metrics['train/dice_coef'].update_state(dice)
        metrics['train/cce'].update_state(labels, predictions)
        metrics['train/accuracy'].update_state(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        tversky = tf.reduce_mean(tversky_loss(labels, predictions))
        dice = tf.reduce_mean(dice_coef(labels, predictions))

        metrics['valid/tversky_loss'].update_state(tversky)
        metrics['valid/dice_coef'].update_state(dice)
        metrics['valid/cce'].update_state(labels, predictions)
        metrics['valid/accuracy'].update_state(labels, predictions)

    for epoch in range(FLAGS.train_epochs):

        start = time.process_time()
        for step, (images, labels) in enumerate(train_ds):
            if FLAGS.num_classes != 1:
                labels = get_multiclass(labels)
            else:
                labels = np.sum(labels, axis=3)

            train_step(images, labels)
            template = 'Epoch {}, Step {}, Elapsed Time: {:3f}, Tversky Loss: {:.3f}, Dice Coefficient:{:.3f}, Categorical CE: {:.3f}, Accuracy: {:.2f}'
            print(template.format(epoch + 1,
                                  step + 1,
                                  time.process_time() - start,
                                  metrics['train/tversky_loss'].result(),
                                  metrics['train/dice_coef'].result(),
                                  metrics['train/cce'].result(),
                                  metrics['train/accuracy'].result() * 100))

            if step == (19200 / FLAGS.batch_size):
                break

        for valid_step, (valid_images, valid_labels) in enumerate(valid_ds):
            if FLAGS.num_classes != 1:
                valid_labels = get_multiclass(valid_labels)
            else:
                valid_labels = np.sum(valid_labels, axis=3)

            test_step(valid_images, valid_labels)
            if (valid_step + 1) % 100 == 0:
                pred = model(valid_images, training=False)
                visualise_multi_class(valid_labels, pred)

            if valid_step == (4480 / FLAGS.batch_size):
                break

        valid_template = 'Validation results: Epoch {}, Tversky Loss: {:.3f}, Dice Coefficient:{:.3f}, Categorical CE: {:.3f}, Accuracy: {:.2f}'
        print(valid_template.format(epoch + 1,
                                    metrics['valid/tversky_loss'].result(),
                                    metrics['valid/dice_coef'].result(),
                                    metrics['valid/cce'].result(),
                                    metrics['valid/accuracy'].result() * 100))

        total_results = {name: metric.result() for name, metric in metrics.items()}
        with summary_writer.as_default():
            for name, result in total_results.items():
                tf.summary.scalar(name, result, step=epoch + 1)

        for metric in metrics.values():
            metric.reset_states()

        model.save_weights(FLAGS.logdir + '/' + FLAGS.model_architecture + str(epoch + 1) + '.ckpt')

if __name__ == '__main__':
    app.run(main)
