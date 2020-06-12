import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path
from datetime import datetime
from absl import app
from absl import flags
from absl import logging
from glob import glob

from Segmentation.model.unet import UNet, R2_UNet, Nested_UNet
from Segmentation.model.segnet import SegNet
from Segmentation.model.deeplabv3 import Deeplabv3
from Segmentation.model.Hundred_Layer_Tiramisu import Hundred_Layer_Tiramisu
from Segmentation.utils.data_loader import read_tfrecord
from Segmentation.utils.losses import dice_coef, dice_coef_loss, tversky_loss
from Segmentation.utils.training_utils import plot_train_history_loss, LearningRateSchedule
from Segmentation.utils.training_utils import visualise_multi_class, visualise_binary
from Segmentation.utils.evaluation_metrics import get_confusion_matrix, plot_confusion_matrix

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
flags.DEFINE_integer('train_epochs', 50, 'Number of training epochs.')
flags.DEFINE_string('aug_strategy', None, 'Augmentation Strategies: None, random-crop, noise, crop_and_noise')

# Model options
flags.DEFINE_string('model_architecture', 'unet', 'unet, r2unet, segnet, unet++, 100-Layer-Tiramisu, deeplabv3')
flags.DEFINE_integer('buffer_size', 5000, 'shuffle buffer size')
flags.DEFINE_bool('multi_class', True, 'Whether to train on a multi-class (Default) or binary setting')
flags.DEFINE_integer('kernel_size', 3, 'kernel size to be used')
flags.DEFINE_bool('use_batchnorm', True, 'Whether to use batch normalisation')
flags.DEFINE_bool('use_bias', True, 'Wheter to use bias')
flags.DEFINE_string('channel_order', 'channels_last', 'channels_last (Default) or channels_first')
flags.DEFINE_string('activation', 'relu', 'activation function to be used')
flags.DEFINE_bool('use_dropout', False, 'Whether to use dropout')
flags.DEFINE_bool('use_spatial', False, 'Whether to use spatial Dropout')
flags.DEFINE_float('dropout_rate', 0.0, 'Dropout rate. Only used if use_dropout is True')

# UNet parameters
flags.DEFINE_list('num_filters', [64, 128, 256, 512, 1024], 'number of filters in the model')
flags.DEFINE_integer('num_conv', 2, 'number of convolution layers in each block')
flags.DEFINE_string('backbone_architecture', 'default', 'default, vgg16, vgg19, resnet50, resnet101, resnet152')
flags.DEFINE_bool('use_transpose', False, 'Whether to use transposed convolution or upsampling + convolution')
flags.DEFINE_bool('use_attention', False, 'Whether to use attention mechanism')

# 100-layer Tiramisu parameters
flags.DEFINE_list('layers_per_block', [4, 5, 7, 10, 12, 15], 'number of convolutional layers per block')
flags.DEFINE_integer('growth_rate', 16, 'number of feature maps increase after each convolution')
flags.DEFINE_integer('pool_size', 2, 'pooling filter size to be used')
flags.DEFINE_integer('strides', 2, 'strides size to be used')
flags.DEFINE_string('padding', 'same', 'padding mode to be used')
flags.DEFINE_string('optimizer', 'adam', 'Which optimizer to use for model: adam, rms-prop')
flags.DEFINE_integer('init_num_channels', 48, 'Initial number of filters needed for the firstconvolutional layer')

# Deeplab parametersi
flags.DEFINE_bool('use_nonlinearity', True, 'Whether to use the activation')
flags.DEFINE_integer('kernel_size_initial_conv', 3, 'kernel size for the first convolution')
flags.DEFINE_integer('num_filters_atrous', 256, 'number of filters for the atrous convolution block')
flags.DEFINE_list('num_filters_DCNN', [256, 512, 1024], 'number of filters for the first three blocks of the DCNN')
flags.DEFINE_integer('num_filters_ASPP', 256, 'number of filters for the ASPP term')
flags.DEFINE_integer('kernel_size_atrous', 3, 'kernel size for the atrous convolutions')
flags.DEFINE_list('kernel_size_DCNN', [1, 3], 'kernel sizes for the blocks of the DCNN')
flags.DEFINE_list('kernel_size_ASPP', [1, 3, 3, 3], 'kernel size for the ASPP term')
flags.DEFINE_list('MultiGrid', [1, 2, 4], 'relative convolution rates for the atrous convolutions')
flags.DEFINE_list('rate_ASPP', [1, 6, 12, 18], 'rates for the ASPP term convolutions')
flags.DEFINE_integer('output_stride', 16, 'final output stride (taking into account max pooling)')

# Logging, saving and testing options
flags.DEFINE_string('tfrec_dir', './Data/tfrecords/', 'directory for TFRecords folder')
flags.DEFINE_string('logdir', 'checkpoints', 'directory for checkpoints')
flags.DEFINE_string('weights_dir', 'checkpoints', 'directory for saved model or weights. Only used if train is False')
flags.DEFINE_string('fig_dir', 'figures', 'directory for saved figures')
flags.DEFINE_bool('train', True, 'If True (Default), train the model. Otherwise, test the model')
flags.DEFINE_string('visual_file', '', 'If not "", creates a visual of the model for the time stamp provided.')

# Accelerator flags
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', 'joe', 'Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS

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

        elif FLAGS.model_architecture == '100-Layer-Tiramisu':

            model = Hundred_Layer_Tiramisu(FLAGS.growth_rate,
                                           FLAGS.layers_per_block,
                                           FLAGS.init_num_channels,
                                           num_classes,
                                           FLAGS.kernel_size,
                                           FLAGS.pool_size,
                                           FLAGS.activation,
                                           FLAGS.dropout_rate,
                                           FLAGS.strides,
                                           FLAGS.padding)

        elif FLAGS.model_architecture == 'deeplabv3':

            model = Deeplabv3(num_classes,
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
                              FLAGS.output_stride)

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

        if FLAGS.optimizer == 'adam':
            optimiser = tf.keras.optimizers.Adam(learning_rate=lr_rate)
        elif FLAGS.optimizer == 'rms-prop':
            optimiser = tf.keras.optimizers.RMSprop(learning_rate=lr_rate)
        else:
            print('Not a valid input optimizer, using Adam.')
            optimiser = tf.keras.optimizers.Adam(learning_rate=lr_rate)

        # for some reason, if i build the model then it can't load checkpoints. I'll see what I can do about this
        if FLAGS.train:
            if FLAGS.backbone_architecture == 'default':

                model.build((batch_size, 288, 288, 1))
            else:
                model.build((batch_size, 288, 288, 3))

            model.summary()
        
        model.compile(optimizer=optimiser,
                      loss=loss_fn,
                      metrics=[dice_coef, crossentropy_loss_fn, 'acc'])
       
                          
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
        #pit code
        training_history_dir = os.path.join(FLAGS.logdir, FLAGS.tpu)
        training_history_dir = os.path.join(training_history_dir, FLAGS.visual_file)
        checkpoints = Path(training_history_dir).glob('*')

        """ add visualisation code here """
        #path = os.path.join(FLAGS.logdir, FLAGS.tpu, FLAGS.visual_file)
        print(training_history_dir)
        # checkpoints = glob(os.path.join(path, "*"))
        print("+========================================================")
        print(f"Does the selected path exist: {Path(training_history_dir).is_dir()}")
        print(f"The glob object is: {checkpoints}")
        print("\n\nThe directories are:")
        for dirs in checkpoints:
            print("\n", dirs)
        print("+========================================================")
        # from os import listdir:
        #     chkp = listdir(path)
        # print(chkp)
        print("Hey Olivia")
        print("+========================================================")
    else:
        # load the checkpoint in the FLAGS.weights_dir file
        model.load_weights(FLAGS.weights_dir).expect_partial()
        model.evaluate(valid_ds, steps=validation_steps)
        cm = np.zeros((num_classes, num_classes))
        classes = ["Background",
                   "Femoral",
                   "Medial Tibial",
                   "Lateral Tibial",
                   "Patellar",
                   "Lateral Meniscus",
                   "Medial Meniscus"]
        for step, (image, label) in enumerate(valid_ds):
            print(step)
            pred = model.predict(image)
            visualise_binary(label, pred, FLAGS.fig_dir)
            # cm = cm + get_confusion_matrix(label, pred, classes=list(range(0, num_classes)))

            if step > validation_steps - 1:
                break

        # fig_file = FLAGS.model_architecture + '_matrix.png'
        # fig_dir = os.path.join(FLAGS.fig_dir, fig_file)
        # plot_confusion_matrix(cm, fig_dir, classes=classes)
if __name__ == '__main__':
    app.run(main)
