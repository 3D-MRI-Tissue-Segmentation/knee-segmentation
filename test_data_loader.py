import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from Segmentation.utils.data_loader import read_tfrecord
from Segmentation.utils.training_utils import visualise_multi_class

train_ds = read_tfrecord(tfrecords_dir='gs://oai-challenge-dataset/tfrecords/train/', batch_size=5, buffer_size=5000, is_training=True)

start = time.process_time()
for image, label in train_ds:
    #visualise_multi_class(label, label) 
    print(image.shape)
    print(label.shape)
    print(time.process_time() - start)
