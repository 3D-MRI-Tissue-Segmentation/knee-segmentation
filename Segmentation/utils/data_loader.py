import h5py
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import math
from functools import partial
import tensorflow as tf
from glob import glob

from Segmentation.utils.augmentation import flip_randomly_left_right_image_pair_2d, rotate_randomly_image_pair_2d, \
    translate_randomly_image_pair_2d

def get_multiclass(label):

    # label shape
    # (batch_size, height, width, channels)

    batch_size = label.shape[0]
    height = label.shape[1]
    width = label.shape[2]
    channels = label.shape[3]

    background = np.zeros((batch_size, height, width, 1))
    label_sum = np.sum(label, axis=3)
    background[label_sum == 0] = 1

    label = np.concatenate((label, background), axis=3)

    return label

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float /p double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_OAI_dataset(data_folder, tfrecord_directory, get_train=True, use_2d=True):

    if not os.path.exists(tfrecord_directory):
        os.mkdir(tfrecord_directory)

    train_val = 'train' if get_train else 'valid'
    files = glob(os.path.join(data_folder, f'*.im'))

    for idx, f in enumerate(files):
        f_name = f.split("/")[-1]
        f_name = f_name.split(".")[0]

        fname_img = f'{f_name}.im'
        fname_seg = f'{f_name}.seg'

        img_filepath = os.path.join(data_folder, fname_img)
        seg_filepath = os.path.join(data_folder, fname_seg)

        assert os.path.exists(seg_filepath), f"Seg file does not exist: {seg_filepath}"

        with h5py.File(img_filepath, 'r') as hf:
            img = np.array(hf['data'])
        with h5py.File(seg_filepath, 'r') as hf:
            seg = np.array(hf['data'])

        img = np.rollaxis(img, 2, 0)
        seg = np.rollaxis(seg, 2, 0)

        img = img[:, 48:336, 48:336]
        seg = seg[:, 48:336, 48:336, :]

        seg_temp = np.zeros((160, 288, 288, 1), dtype=np.int8)
        seg_sum = np.sum(seg, axis=3)
        seg_temp[seg_sum == 0] = 1
        seg = np.concatenate([seg_temp, seg], axis=3)
        img = np.expand_dims(img, axis=3)

        shard_dir = f'{idx:03d}-of-{len(files) - 1:03d}.tfrecords'
        tfrecord_filename = os.path.join(tfrecord_directory, shard_dir)

        with tf.io.TFRecordWriter(tfrecord_filename) as writer:
            if use_2d:
                for k in range(len(img)):
                    img_slice = img[k, :, :, :]
                    seg_slice = seg[k, :, :, :]

                    img_raw = img_slice.tostring()
                    seg_raw = seg_slice.tostring()

                    height = img_slice.shape[0]
                    width = img_slice.shape[1]
                    num_channels = seg_slice.shape[-1]

                    feature = {
                        'height': _int64_feature(height),
                        'width': _int64_feature(width),
                        'num_channels': _int64_feature(num_channels),
                        'image_raw': _bytes_feature(img_raw),
                        'label_raw': _bytes_feature(seg_raw)
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
            else:
                height = img.shape[0]
                width = img.shape[1]
                depth = img.shape[2]
                num_channels = seg.shape[-1]

                img_raw = img.tostring()
                seg_raw = seg.tostring()

                feature = {
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'depth': _int64_feature(depth),
                    'num_channels': _int64_feature(num_channels),
                    'image_raw': _bytes_feature(img_raw),
                    'label_raw': _bytes_feature(seg_raw)
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        print(f'{idx} out of {len(files) - 1} datasets have been processed')

def parse_fn_2d(example_proto, training, multi_class=True):

    features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'num_channels': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string)
    }

    # Parse the input tf.Example proto using the dictionary above.
    image_features = tf.io.parse_single_example(example_proto, features)
    image_raw = tf.io.decode_raw(image_features['image_raw'], tf.float32)
    image = tf.reshape(image_raw, [288, 288, 1])

    seg_raw = tf.io.decode_raw(image_features['label_raw'], tf.int16)
    seg = tf.reshape(seg_raw, [288, 288, 7])
    seg = tf.cast(seg, tf.float32)

    #if training:
    #    image, seg = flip_randomly_left_right_image_pair_2d(image, seg)
    #    image, seg = translate_randomly_image_pair_2d(image, seg, 24, 12)
    #    image, seg = rotate_randomly_image_pair_2d(image, seg, tf.constant(-math.pi / 12), tf.constant(math.pi / 12))

    if not multi_class:
        seg = tf.math.reduce_sum(seg, axis=-1)

    return (image, seg)

def parse_fn_3d(example_proto, training, multi_class=True):

    features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'num_channels': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string)
    }

    # Parse the input tf.Example proto using the dictionary above.
    image_features = tf.io.parse_single_example(example_proto, features)
    image_raw = tf.io.decode_raw(image_features['image_raw'], tf.float32)
    image = tf.reshape(image_raw, [image_features['height'], image_features['width'], image_features['depth'], 1])

    seg_raw = tf.io.decode_raw(image_features['label_raw'], tf.int16)
    seg = tf.reshape(seg_raw, [image_features['height'], image_features['width'],
                               image_features['depth'], image_features['num_channels']])
    seg = tf.cast(seg, tf.float32)

    if not multi_class:
        seg = tf.math.reduce_sum(seg, axis=-1)

    return (image, seg)

def read_tfrecord(tfrecords_dir, batch_size, buffer_size, parse_fn=parse_fn_2d, multi_class=True, is_training=False):

    file_list = tf.io.matching_files(os.path.join(tfrecords_dir, '*-*'))
    shards = tf.data.Dataset.from_tensor_slices(file_list)
    if is_training:
        shards = shards.shuffle(tf.cast(tf.shape(file_list)[0], tf.int64))
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    parser = partial(parse_fn, training=is_training, multi_class=multi_class)
    dataset = dataset.map(map_func=parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    # optimise dataset performance
    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
