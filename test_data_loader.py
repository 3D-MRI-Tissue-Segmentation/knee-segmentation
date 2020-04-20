import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from Segmentation.utils.data_loader import read_tfrecord
from Segmentation.utils.training_utils import visualise_multi_class

train_ds = read_tfrecord(tfrecords_dir='./Data/tfrecords/train/', batch_size=5, is_training=True)

for image, label in train_ds:
    visualise_multi_class(label, label)    
