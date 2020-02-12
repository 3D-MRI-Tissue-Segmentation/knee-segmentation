import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Segmentation.model.unet import UNet
from Segmentation.tests.test_unet import UNetTest
from Segmentation.utils.data_loader import create_OAI_dataset

import os
import time
from absl import app
from absl import flags
from absl import logging 


## Dataset/training options
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('batch_size', 8, 'Batch size per GPU')
flags.DEFINE_float('base_learning_rate', 0.1, 'base learning rate at the start of training session')
flags.DEFINE_string('dataset', 'oai_challenge', 'Dataset: oai_challenge, isic_2018 or oai_full')
flags.DEFINE_bool('2D', True, 'True to train on 2D slices, False to train on 3D data')
flags.DEFINE_bool('corruptions', False, 'Whether to test on corrupted dataset')
flags.DEFINE_integer('train_epochs', 50, 'Number of training epochs.'

## Model options
flags.DEFINE_string('model_architecture', 'baseline_unet', 'Model: baseline_unet, multires_unet, attention_unet, R2_unet', 'R2_attention_unet')
flags.DEFINE_bool('batchnorm', True, 'Whether to use batch normalisation')
flags.DEFINE_bool('dropout', True, 'Whether to use Dropout')
flags.DEFINE_bool('spatial _dropout', False, 'Whether to use spatial Dropout')
flags.DEFINE_integer('num_filters', 64, 'number of filters in the model')

## Logging, saving and testing options
flags.DEFINE_string('logdir', '/checkpoints/', 'directory for checkpoints')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_string('test_set', False, 'Set to True to test on the test set, False to test on the validation set')

## TODO(Joonsu): Implement train and test functions


                
