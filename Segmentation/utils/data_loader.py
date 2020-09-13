import h5py
import numpy as np
import os
from functools import partial
import tensorflow as tf
from glob import glob

from Segmentation.utils.augmentation import crop_randomly_image_pair_2d, flip_randomly_image_pair_2d, normalise
from Segmentation.utils.augmentation import apply_centre_crop_3d, apply_valid_random_crop_3d
from Segmentation.utils.augmentation import apply_random_brightness_3d, apply_random_contrast_3d, apply_random_gamma_3d
from Segmentation.utils.augmentation import apply_flip_3d, apply_rotate_3d

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


def parse_fn_2d(example_proto,
                crop_size=None,
                training=False,
                augmentation=[],
                multi_class=True,
                use_bfloat16=False,
                use_RGB=False):

    """ Parse and augment 2d data by batch """

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

    tf.debugging.check_numerics(image, "Invalid value in your input!")
    tf.debugging.check_numerics(seg, "Invalid value in your label!")

    supported_augs = ["random_crop", "random_noise", "random_flip"]
    if crop_size is None:
        crop_size = image_features['height']

    if training:
        if augmentation is None:
            print("Using default augmentation strategy: center_crop")
            image = tf.image.resize_with_crop_or_pad(image, crop_size, crop_size)
            seg = tf.image.resize_with_crop_or_pad(seg, crop_size, crop_size)
        else:
            if 'center_crop' in augmentation:
                image = tf.image.resize_with_crop_or_pad(image, crop_size, crop_size)
                seg = tf.image.resize_with_crop_or_pad(seg, crop_size, crop_size)
            if 'random_crop' in augmentation:
                print("Applying random crop...")
                image, seg = crop_randomly_image_pair_2d(image, seg, size=[crop_size,crop_size])
            if 'random_noise' in augmentation:
                print("Applying random noise...")
                image = tf.image.random_brightness(image, max_delta=0.05)
                image = tf.image.random_contrast(image, 0.01, 0.05)
            if 'random_flip' in augmentation:
                print("Applying random flip...")
                image, seg = flip_randomly_image_pair_2d(image, seg)

            unsupported_augs = np.setdiff1d(supported_augs, augmentation)
            for item in supported_augs:
                if item in unsupported_augs:
                    unsupported_augs = np.delete(unsupported_augs, np.argwhere(unsupported_augs == item))

            if unsupported_augs:
                print("Augmentation strategy {} does not exist or is not supported!".format(unsupported_augs))
    else:
        image = tf.image.resize_with_crop_or_pad(image, crop_size, crop_size)
        seg = tf.image.resize_with_crop_or_pad(seg, crop_size, crop_size)

    if not multi_class:
        seg = seg[..., 1:]
        seg = tf.math.reduce_sum(seg, axis=-1)
        seg = tf.expand_dims(seg, axis=-1)
        seg = tf.clip_by_value(seg, 0, 1)

    image = normalise(image)
    # image, seg = normalise(image, seg) # THIS LINE WILL BREAK THE CODE IF RANDOM CROP IS USED!!!

    return (image, seg)


def parse_fn_3d(example_proto,
                training=False,
                augmentation=[],
                crop_size=None,
                depth_crop_size=80,
                multi_class=True,
                use_bfloat16=False,
                use_RGB=False):

    """ Parse and augment 3d data by batch """

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

    # TODO: put into augmentation rather than parse fxn
    supported_augs = ["bright", "contrast", "gamma", "flip", "rotate"]
    if training:
        if "bright" in augmentation:
            image, seg = apply_random_brightness_3d(image, seg)
        if "contrast" in augmentation:
            image, seg = apply_random_contrast_3d(image, seg)
        if "gamma" in augmentation:
            image, seg = apply_random_gamma_3d(image, seg)
        if "flip" in augmentation:
            image, seg = apply_flip_3d(image, seg)
        if "rotate" in augmentation:
            image, seg = apply_rotate_3d(image, seg)
        if augmentation is None:
            image, seg = apply_centre_crop_3d(image,
                                          seg,
                                          crop_size=crop_size,
                                          depth_crop_size=depth_crop_size,
                                          output_slice=predict_slice)
        unsupported_augs = np.setdiff1d(supported_augs, augmentation)
        if unsupported_augs is not None:
            "Augmentation strategy {} does not exist or is not supported!".format(unsupported_augs)
    else:
        image, seg = apply_centre_crop_3d(image,
                                          seg,
                                          crop_size=crop_size,
                                          depth_crop_size=depth_crop_size,
                                          output_slice=predict_slice)  # predict_slice is undefined

    image, seg = normalise(image)

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

    file_list = tf.io.matching_files(os.path.join(tfrecords_dir, '*-*'))
    shards = tf.data.Dataset.from_tensor_slices(file_list)
    if is_training:
        shards = shards.shuffle(tf.cast(tf.shape(file_list)[0], tf.int64))
    cycle_length = 4
    if use_2d:
        cycle_length = 8 if is_training else 1
    
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset,
                                cycle_length=cycle_length,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    if use_2d:
        parser = partial(parse_fn,
                         crop_size=crop_size,
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

    """Function for loading and parsing dataset
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

    train_ds = read_tfrecord(tfrecords_dir=os.path.join(dataset_dir, train_dir),
                             is_training=True,
                             **ds_args
                             )

    valid_ds = read_tfrecord(tfrecords_dir=os.path.join(dataset_dir, valid_dir),
                             is_training=False,
                             **ds_args)

    return train_ds, valid_ds
