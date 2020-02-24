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
from Segmentation.utils.data_loader import DataGenerator
from Segmentation.utils.training_utils import dice_loss, jaccard_distance_loss, dice_coef_loss, tversky_loss
from Segmentation.utils.training_utils import plot_train_history_loss, plot_results, visualise_multi_class

## Dataset/training options
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('batch_size', 4, 'Batch size per GPU')
flags.DEFINE_float('base_learning_rate', 1e-04, 'base learning rate at the start of training session')
flags.DEFINE_string('dataset', 'oai_challenge', 'Dataset: oai_challenge, isic_2018 or oai_full')
flags.DEFINE_bool('2D', True, 'True to train on 2D slices, False to train on 3D data')
flags.DEFINE_bool('corruptions', False, 'Whether to test on corrupted dataset')
flags.DEFINE_integer('train_epochs', 20, 'Number of training epochs.')

## Model options
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

## Logging, saving and testing options
flags.DEFINE_string('logdir', '/checkpoints/', 'directory for checkpoints')
flags.DEFINE_string('savedir', '/checkpoints/', 'directory for saved models')

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
    
    optimiser = tf.keras.optimizers.Adam(learning_rate=FLAGS.base_learning_rate)

    if FLAGS.num_classes == 1:
        if FLAGS.dataset == 'oai_challenge':
                generator_train = DataGenerator("./Data/train_2d/samples/", 
                                                "./Data/train_2d/labels/",
                                                batch_size=FLAGS.batch_size,
                                                shuffle=True,
                                                multi_class=False)
                generator_valid = DataGenerator("./Data/valid_2d/samples/", 
                                                "./Data/valid_2d/labels/",
                                                batch_size=FLAGS.batch_size,
                                                shuffle=True,
                                                multi_class=False)
        model.compile(optimizer=optimiser, 
                loss=dice_loss, 
                metrics=[dice_coef_loss, 'binary_crossentropy'])

    else:
        if FLAGS.dataset == 'oai_challenge':
                generator_train = DataGenerator("./Data/train_2d/samples/", 
                                                "./Data/train_2d/labels/",
                                                batch_size=FLAGS.batch_size,
                                                shuffle=True,
                                                multi_class=True)
                generator_valid = DataGenerator("./Data/valid_2d/samples/", 
                                                "./Data/valid_2d/labels/",
                                                batch_size=FLAGS.batch_size,
                                                shuffle=True,
                                                multi_class=True)
        
        model.compile(optimizer=optimiser, 
                loss=tversky_loss, 
                metrics=['categorical_crossentropy'])


    model.fit_generator(generator=generator_train,
                        epochs=FLAGS.train_epochs, 
                        validation_data=generator_valid,
                        use_multiprocessing=True,
                        shuffle=True,
                        workers=16,
                        max_queue_size=32)

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)

    model_path = FLAGS.model_architecture + '_' + current_time + '_' + str(FLAGS.num_classes) + '.ckpt'
    save_path = os.path.join(FLAGS.savedir, model_path)
    
    model.save_weights(save_path)
        
if __name__ == '__main__':
  app.run(main)