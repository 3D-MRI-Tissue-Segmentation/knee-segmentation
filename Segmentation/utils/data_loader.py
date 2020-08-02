import h5py
import numpy as np
import os
from functools import partial
import tensorflow as tf
from glob import glob

from Segmentation.utils.augmentation import crop_randomly_image_pair_2d, adjust_contrast_randomly_image_pair_2d
from Segmentation.utils.augmentation import adjust_brightness_randomly_image_pair_2d
from Segmentation.utils.augmentation import apply_centre_crop_3d, apply_valid_random_crop_3d
from Segmentation.utils.augmentation import apply_random_brightness_3d, apply_random_contrast_3d, apply_random_gamma_3d
from Segmentation.utils.augmentation import apply_flip_3d, apply_rotate_3d, normalise

def get_multiclass(label):

    # label shape
    # (batch_size, height, width, channels)

    batch_size = label.shape[0]
    height = label.shape[1]
    width = label.shape[2]

    background = np.zeros((batch_size, height, width, 1))
    label_sum = np.sum(label, axis=3)
    background[label_sum == 0] = 1

    label = np.concatenate((label, background), axis=3)

    return label

def parse_fn_2d(example_proto, training, augmentation, multi_class=True, use_bfloat16=False, use_RGB=False):

    if use_bfloat16:
        dtype = tf.bfloat16
    else:
        dtype = tf.float32

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
    image = tf.cast(tf.reshape(image_raw, [384, 384, 1]), dtype)

    if use_RGB:
        image = tf.image.grayscale_to_rgb(image)

    seg_raw = tf.io.decode_raw(image_features['label_raw'], tf.int16)
    seg = tf.reshape(seg_raw, [384, 384, 7])
    seg = tf.cast(seg, dtype)

    if training:
        if augmentation == 'random_crop':
            image, seg = crop_randomly_image_pair_2d(image, seg)
        elif augmentation == 'noise':
            image, seg = adjust_brightness_randomly_image_pair_2d(image, seg)
            image, seg = adjust_contrast_randomly_image_pair_2d(image, seg)
        elif augmentation == 'crop_and_noise':
            image, seg = crop_randomly_image_pair_2d(image, seg)
            image, seg = adjust_brightness_randomly_image_pair_2d(image, seg)
            image, seg = adjust_contrast_randomly_image_pair_2d(image, seg)
        elif augmentation is None:
            image = tf.image.resize_with_crop_or_pad(image, 288, 288)
            seg = tf.image.resize_with_crop_or_pad(seg, 288, 288)
        else:
            "Augmentation strategy {} does not exist or is not supported!".format(augmentation)

    else:
        image = tf.image.resize_with_crop_or_pad(image, 288, 288)
        seg = tf.image.resize_with_crop_or_pad(seg, 288, 288)

    if not multi_class:
        seg = tf.slice(seg, [0, 0, 1], [-1, -1, 6])
        seg = tf.math.reduce_sum(seg, axis=-1)
        seg = tf.expand_dims(seg, axis=-1)
        seg = tf.clip_by_value(seg, 0, 1)

    return (image, seg)


def parse_fn_3d(example_proto, training, multi_class=True, use_bfloat16=False, use_RGB=False):

    if use_bfloat16:
        dtype = tf.bfloat16
    else:
        dtype = tf.float32

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

    image = tf.reshape(image_raw, [image_features['height'], image_features['width'],
                                   image_features['depth'], 1])
    image = tf.cast(image, dtype)

    seg_raw = tf.io.decode_raw(image_features['label_raw'], tf.int16)
    seg = tf.reshape(seg_raw, [image_features['height'], image_features['width'],
                               image_features['depth'], image_features['num_channels']])
    seg = tf.cast(seg, dtype)

    if not multi_class:
        seg = tf.slice(seg, [0, 0, 0, 1], [-1, -1, -1, 6])
        seg = tf.math.reduce_sum(seg, axis=-1)
        seg = tf.expand_dims(seg, axis=-1)
        seg = tf.clip_by_value(seg, 0, 1)

    # if training:
    #     dx = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=128), tf.int32)
    #     dy = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=96), tf.int32)
    #     dz = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=96), tf.int32)

    #     image = image[dx: dx + 32, dy: dy + 288, dz: dz + 288, :]
    #     seg = seg[dx: dx + 32, dy: dy + 288, dz: dz + 288, :]
    # else:
    #     image = image[64:96, 48:336, 48:336, :]
    #     seg = seg[64:96, 48:336, 48:336, :]

    # image = tf.reshape(image, [32, 288, 288, 1])
    # tf.print(seg.shape)
    # print(seg.shape)
    # seg = tf.reshape(seg, [32, 288, 288, 7])

    return (image, seg)


<<<<<<< HEAD
def read_tfrecord(tfrecords_dir,
                  batch_size,
                  buffer_size,
                  parse_fn=parse_fn_2d,
                  multi_class=True,
                  is_training=False,
                  crop_size=None,
                  use_2d=False,
                  augmentation=None,
                  use_bfloat16=False,
                  use_RGB=False):

    """This function reads and returns TFRecords dataset in tf.data.Dataset format
    """

=======
def read_tfrecord(tfrecords_dir, batch_size, buffer_size, parse_fn=parse_fn_2d,
                  multi_class=True, is_training=False, use_keras_fit=True, crop_size=None,
                  use_2d=True, augmentation=None, use_bfloat16=False, use_RGB=False):
>>>>>>> 8bfeb3791bc4d88ddc715842770cfee726b60521
    file_list = tf.io.matching_files(os.path.join(tfrecords_dir, '*-*'))
    shards = tf.data.Dataset.from_tensor_slices(file_list)
    if is_training:
        shards = shards.shuffle(tf.cast(tf.shape(file_list)[0], tf.int64))
    cycle_length = 4
    if use_2d:
        cycle_length = 8 if is_training else 1

    dataset = shards.interleave(tf.data.TFRecordDataset,
                                cycle_length=cycle_length,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    if use_2d:
        parser = partial(parse_fn,
                         training=is_training,
                         augmentation=augmentation,
                         multi_class=multi_class,
                         use_bfloat16=use_bfloat16,
                         use_RGB=use_RGB)
    else:
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


def read_tfrecord_2d(tfrecords_dir, batch_size, buffer_size, augmentation_2d,
                     parse_fn=parse_fn_2d, multi_class=True,
                     is_training=False, use_bfloat16=False,
                     use_RGB=False):
    dataset = read_tfrecord(tfrecords_dir, batch_size, buffer_size,)
    return dataset


def read_tfrecord_3d(tfrecords_dir,
                     batch_size,
                     buffer_size,
                     is_training,
                     crop_size=None,
                     depth_crop_size=80,
                     aug=[],
                     predict_slice=False,
                     **kwargs):

    dataset = read_tfrecord(tfrecords_dir=tfrecords_dir,
                            batch_size=batch_size,
                            buffer_size=buffer_size,
                            parse_fn=parse_fn_3d,
                            is_training=is_training,
                            **kwargs)

    if crop_size is not None:
        if is_training:
            resize = "resize" in aug
            random_shift = "shift" in aug
            parse_crop = partial(apply_valid_random_crop_3d, crop_size=crop_size, depth_crop_size=depth_crop_size, resize=resize, random_shift=random_shift, output_slice=predict_slice)
            dataset = dataset.map(map_func=parse_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            if "bright" in aug:
                dataset = dataset.map(apply_random_brightness_3d)
            if "contrast" in aug:
                dataset = dataset.map(apply_random_contrast_3d)
            if "gamma" in aug:
                dataset = dataset.map(apply_random_gamma_3d)
            if "flip" in aug:
                dataset = dataset.map(apply_flip_3d)
            if "rotate" in aug:
                dataset = dataset.map(apply_rotate_3d)
        else:
            parse_crop = partial(apply_centre_crop_3d, crop_size=crop_size, depth_crop_size=depth_crop_size, output_slice=predict_slice)
            dataset = dataset.map(map_func=parse_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(normalise)
    return dataset

def load_dataset(batch_size,
                 dataset_dir,
                 augmentation,
                 use_2d,
                 multi_class,
                 crop_size=None,
                 buffer_size=5000,
                 use_bfloat16=False,
                 use_RGB=False,
                 ):

    """Function for loading dataset
    """

    # Define the directories of the training and validation files
    train_dir = 'train/' if use_2d else 'train_3d/'
    valid_dir = 'valid/' if use_2d else 'valid_3d/'

    # Print out the augmentation strategy used during training
    print('Augmentation Strategy used: {}'.format(augmentation))

    # Define the datasets as tf.data.Datasets using read_tfrecord function

    ds_args = {
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'augmentation': augmentation,
        'parse_fn': parse_fn_2d if use_2d else parse_fn_3d,
        'multi_class': multi_class,
        'crop_size': crop_size,
        'use_2d': use_2d,
        'use_bfloat16': use_bfloat16,
        'use_RGB': use_RGB
    }

    train_ds = read_tfrecord(tfrec_dir=os.path.join(dataset_dir, train_dir),
                             is_training=True,
                             **ds_args
                             )

    valid_ds = read_tfrecord(tfrec_dir=os.path.join(dataset_dir, valid_dir),
                             is_training=False,
                             **ds_args)

    return train_ds, valid_ds
