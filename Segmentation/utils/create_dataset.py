import tensorflow as tf
import glob
import h5py
import os
import numpy as np

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

    # train_val = 'train' if get_train else 'valid'
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

            # seg_total = np.sum(seg)

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
