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
from Segmentation.utils.training_utils import dice_coef, dice_coef_loss, tversky_loss, iou_loss_core, Mean_IOU, iou_loss_core
from Segmentation.utils.training_utils import plot_train_history_loss, visualise_multi_class, make_lr_scheduler, visualise_binary

## Dataset/training options
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('batch_size', 4, 'Batch size per GPU')
flags.DEFINE_float('base_learning_rate', 1e-04, 'base learning rate at the start of training session')
flags.DEFINE_string('dataset', 'oai_challenge', 'Dataset: oai_challenge, isic_2018 or oai_full')
flags.DEFINE_bool('2D', True, 'True to train on 2D slices, False to train on 3D data')
flags.DEFINE_bool('corruptions', False, 'Whether to test on corrupted dataset')
flags.DEFINE_integer('train_epochs', 5, 'Number of training epochs.')

## Model options
flags.DEFINE_string('model_architecture', 'attention_unet_v1', 'Model: unet (default), multires_unet, attention_unet_v1, R2_unet, R2_attention_unet')
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

## Logging, saving and testing options
flags.DEFINE_string('tfrec_dir', './Data/tfrecords/', 'directory for TFRecords folder')
flags.DEFINE_string('logdir', './checkpoints', 'directory for checkpoints')
flags.DEFINE_bool('train', True, 'If True (Default), train the model. Otherwise, test the model')

FLAGS = flags.FLAGS

def main(argv):
    
    del argv #unused arg
    
    #select mode architecture
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
    
    optimiser = tf.keras.optimizers.Adam(learning_rate=FLAGS.base_learning_rate)
    
    if FLAGS.num_classes == 1:
        
        train_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir,'train/'), batch_size=FLAGS.batch_size, is_training=True)
        valid_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir,'valid/'), batch_size=FLAGS.batch_size, is_training=False)

        model.compile(optimizer=optimiser, 
                    loss=dice_coef_loss, 
                    metrics=[dice_coef, 'binary_crossentropy', 'acc'])

    else:                
        
        train_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir,'train/'), batch_size=FLAGS.batch_size, is_training=True)
        valid_ds = read_tfrecord(tfrecords_dir=os.path.join(FLAGS.tfrec_dir,'valid/'), batch_size=FLAGS.batch_size, is_training=False)

        model.compile(optimizer=optimiser, 
                loss=tversky_loss, 
                metrics=[dice_coef, 'categorical_crossentropy', 'acc'])

    if FLAGS.train:
        
        #define checkpoints 
        logdir = 'checkpoints\\' + datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_cb   = tf.keras.callbacks.ModelCheckpoint(FLAGS.logdir + '/' + FLAGS.model_architecture + '_weights.{epoch:03d}.ckpt',save_best_only=False, save_weights_only=True)
        lr_schedule = make_lr_scheduler(FLAGS.base_learning_rate)
        tb = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch')

        history = model.fit(train_ds,
                            steps_per_epoch=19200//FLAGS.batch_size,
                            epochs=FLAGS.train_epochs,
                            validation_data=valid_ds,
                            validation_steps=4480//FLAGS.batch_size,
                            callbacks=[ckpt_cb, lr_schedule,tb])

        if FLAGS.num_classes == 1:
            plot_train_history_loss(history, multi_class=False)
        else:
            plot_train_history_loss(history, multi_class=True)

    else:
        #load the latest checkpoint in the FLAGS.logdir file 
        #latest = tf.train.latest_checkpoint(FLAGS.logdir)
        model.load_weights('./checkpoints/unet/14-4-2020/unet_weights.005.ckpt').expect_partial()
        for step, (image, label) in enumerate(valid_ds):

            if step >= 80:    
                pred = model(image, training=False)
                visualise_multi_class(label, pred)
        
if __name__ == '__main__':
  app.run(main)