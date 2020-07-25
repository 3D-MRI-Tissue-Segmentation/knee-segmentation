import h5py
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import math
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


def create_OAI_dataset(data_folder, tfrecord_directory, get_train=True, use_2d=True, crop_size=None):

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

        if crop_size is not None:

            img_mid = (int(img.shape[0] / 2), int(img.shape[1] / 2))
            seg_mid = (int(seg.shape[0] / 2), int(seg.shape[1] / 2))

            assert img_mid == seg_mid, "We expect the mid shapes to be the same size"

            seg_total = np.sum(seg)

            img = img[img_mid[0] - crop_size:img_mid[0] + crop_size,
                      img_mid[1] - crop_size:img_mid[1] + crop_size, :]
            seg = seg[seg_mid[0] - crop_size:seg_mid[0] + crop_size,
                      seg_mid[1] - crop_size:seg_mid[1] + crop_size, :, :]

            # assert np.sum(seg) == seg_total, "We are losing information in the initial cropping."
            assert img.shape == (crop_size * 2, crop_size * 2, 160)
            assert seg.shape == (crop_size * 2, crop_size * 2, 160, 6)

        img = np.rollaxis(img, 2, 0)
        seg = np.rollaxis(seg, 2, 0)
        seg_temp = np.zeros((*seg.shape[0:3], 1), dtype=np.int8)

        assert seg.shape[0:3] == seg_temp.shape[0:3]

        seg_sum = np.sum(seg, axis=-1)
        seg_temp[seg_sum == 0] = 1
        seg = np.concatenate([seg_temp, seg], axis=-1)  # adds additional channel for no class
        img = np.expand_dims(img, axis=-1)
        assert img.shape[-1] == 1
        assert seg.shape[-1] == 7

        shard_dir = f'{idx:03d}-of-{len(files) - 1:03d}.tfrecords'
        tfrecord_filename = os.path.join(tfrecord_directory, shard_dir)

        target_shape, label_shape = None, None
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

                    target_shape = img_slice.shape
                    label_shape = seg.shape

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

                target_shape = img.shape
                label_shape = seg.shape

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
        print(f'{idx} out of {len(files) - 1} datasets have been processed. Target: {target_shape}, Label: {label_shape}')


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
    
    image = tf.reshape(image_raw, [160, 384, 384, 1])
    image = tf.cast(image, dtype)

    seg_raw = tf.io.decode_raw(image_features['label_raw'], tf.int16)
    seg = tf.reshape(seg_raw, [160, 384, 384, 7])
    seg = tf.cast(seg, dtype)

    if not multi_class:
        seg = tf.slice(seg, [0, 0, 0, 1], [-1, -1, -1, 6])
        seg = tf.math.reduce_sum(seg, axis=-1)
        seg = tf.expand_dims(seg, axis=-1)
        seg = tf.clip_by_value(seg, 0, 1)

    if training:
        dx = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=128), tf.int32)
        dy = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=96), tf.int32)
        dz = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=96), tf.int32)

        image = image[dx:dx+32, dy:dy+288, dz:dz+288, :]
        seg = seg[dx:dx+32, dy:dy+288, dz:dz+288, :]
    else:
        image = image[64:96, 48:336, 48:336, :]
        seg = seg[64:96, 48:336, 48:336, :]

    image = tf.reshape(image, [32, 288, 288, 1])
    seg = tf.reshape(seg, [32, 288, 288, 7])

    return (image, seg)


def read_tfrecord(tfrecords_dir, batch_size, buffer_size, parse_fn=parse_fn_2d,
                  multi_class=True, is_training=False, use_keras_fit=True, crop_size=None,
                  use_2d=False, augmentation_2d=None, use_bfloat16_2d=False, use_RGB_2d=False):
    file_list = tf.io.matching_files(os.path.join(tfrecords_dir, '*-*'))
    shards = tf.data.Dataset.from_tensor_slices(file_list)
    if is_training:
        shards = shards.shuffle(tf.cast(tf.shape(file_list)[0], tf.int64))
    if use_keras_fit:
        shards = shards.repeat()
    cycle_length = 4
    if use_2d:
        cycle_l = 8 if is_training else 1

    dataset = shards.interleave(tf.data.TFRecordDataset,
                                cycle_length=cycle_length,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    if use_2d:
        parser = partial(parse_fn,
                        training=is_training,
                        augmentation=augmentation_2d,
                        multi_class=multi_class,
                        use_bfloat16=use_bfloat16_2d,
                        use_RGB=use_RGB_2d)
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
