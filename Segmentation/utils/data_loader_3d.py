from glob import glob
import h5py
import numpy as np
from random import randint
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from math import ceil


class VolumeGenerator(Sequence):
    def __init__(self, batch_size, sample_shape=(364, 364, 160),
                 file_path='t', shuffle_order=True,
                 normalise_input=True, remove_outliers=True,
                 transform_angle=False, transform_position=False,
                 get_slice=False, get_position=False, skip_empty=True,
                 examples_per_load=1, train_debug=False):
        self.batch_size = batch_size
        self.sample_shape = sample_shape
        self.data_paths = VolumeGenerator.get_paths(file_path)
        self.shuffle_order = shuffle_order
        self.normalise_input = normalise_input
        self.remove_outliers = remove_outliers
        self.transform_angle = transform_angle
        self.transform_position = transform_position
        self.get_slice = get_slice
        self.get_position = get_position
        self.skip_empty = skip_empty
        self.examples_per_load = examples_per_load
        self.train_debug = train_debug

        if self.train_debug:
            cut = int(len(self.data_paths) / 10)
            self.data_paths = self.data_paths[:cut]

        assert self.batch_size <= len(self.data_paths), f"Batch size {self.batch_size} must be less than or equal to number of training examples {len(self.data_paths)}"
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_paths))
        if self.shuffle_order:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return ceil(len(self.data_paths) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.data_paths[idx] for idx in indexes]
        x, y = self.generate_batch(batch)
        return x, y

    def generate_batch(self, batch, skip_fail=3):
        x_train, y_train = [], []
        if self.get_position:
            image_arr, pos_arr = [], []
        for sample_path in batch:
            count = self.examples_per_load
            skip_count = skip_fail
            x_path, y_path = sample_path

            volume_x_original = VolumeGenerator.load_file(x_path)
            volume_y_original = VolumeGenerator.load_file(y_path)

            while count > 0:
                sample_pos, sample_pos_max = VolumeGenerator.get_sample_pos(volume_x_original.shape, self.sample_shape,
                                                                            self.transform_position)

                volume_x = VolumeGenerator.sample_from_volume(volume_x_original, self.sample_shape, sample_pos)
                volume_y = VolumeGenerator.sample_from_volume(volume_y_original, self.sample_shape, sample_pos)
                volume_y = np.any(volume_y, axis=-1)

                if self.normalise_input or self.remove_outliers:
                    mean = tf.math.reduce_mean(volume_x)
                    if self.remove_outliers:
                        np.clip(volume_x, None, 0.01, volume_x)
                    if self.normalise_input:
                        volume_x = VolumeGenerator.normalise(volume_x, mean)

                volume_x = VolumeGenerator.expand_dim_as_float(volume_x)
                volume_y = VolumeGenerator.expand_dim_as_float(volume_y)

                if self.get_slice:
                    slice_idx = int((self.sample_shape[2] + 1) / 2) - 1
                    assert slice_idx >= 0
                    volume_y = volume_y[:, :, slice_idx]

                if self.skip_empty:
                    if np.sum(volume_y) == 0:
                        skip_count -= 1
                        if skip_count > 0:
                            continue

                if self.get_position:
                    image_arr.append(volume_x)
                    pos = np.empty(3, dtype=np.float32)
                    for i in range(3):
                        pos[i] = VolumeGenerator.normalise_position(sample_pos[i], sample_pos_max[i])
                    pos_arr.append(pos)
                else:
                    x_train.append(volume_x)
                y_train.append(volume_y)
                count -= 1

        if self.get_position:
            image_arr = np.stack(image_arr, axis=0)
            pos_arr = np.stack(pos_arr, axis=0)
            x_train = [image_arr, pos_arr]
        else:
            x_train = np.stack(x_train, axis=0)
        y_train = np.stack(y_train, axis=0)
        return x_train, y_train

    @staticmethod
    def get_sample_pos(volume_shape, sample_shape, transform_position):
        """
        - Get the position required to translate the volumes by. Ranges from 0 to volume_shape - sample_shape
        - If (volume_shape - sample_shape) == 0, sample and volume same shape. Also the position is centred.
        """
        vol_x, vol_y, vol_z = volume_shape[0] - 1, volume_shape[1] - 1, volume_shape[2] - 1
        samp_x, samp_y, samp_z = sample_shape[0] - 1, sample_shape[1] - 1, sample_shape[2] - 1
        centre_x = int(vol_x / 2) - int(samp_x / 2)
        centre_y = int(vol_y / 2) - int(samp_y / 2)
        centre_z = int(vol_z / 2) - int(samp_z / 2)
        x_max = volume_shape[0] - sample_shape[0]
        y_max = volume_shape[1] - sample_shape[1]
        z_max = volume_shape[2] - sample_shape[2]
        pos_max = np.array([x_max, y_max, z_max], dtype=np.int32)
        pos = None
        if transform_position == "normal":
            stddev_x = int(centre_x / 4)
            stddev_y = int(centre_y / 4)
            stddev_z = int(centre_z / 4)
            x_pos = np.random.normal(centre_x, stddev_x)
            y_pos = np.random.normal(centre_y, stddev_y)
            z_pos = np.random.normal(centre_z, stddev_z)
            float_pos = np.array([x_pos, y_pos, z_pos], dtype=np.float32)
            float_pos = np.clip(float_pos, 0, [x_max, y_max, z_max])
            pos = np.rint(float_pos)
        elif transform_position == "uniform":
            x_pos = np.random.uniform(0, x_max)
            y_pos = np.random.uniform(0, y_max)
            z_pos = np.random.uniform(0, z_max)
            float_pos = np.array([x_pos, y_pos, z_pos], dtype=np.float32)
            pos = np.rint(float_pos)
        else:
            x_pos = centre_x
            y_pos = centre_y
            z_pos = centre_z
            pos = np.array([x_pos, y_pos, z_pos], dtype=np.int32)
        pos = pos.astype(int)
        return pos, pos_max

    @staticmethod
    def get_paths(file_path):
        if file_path == "t":
            file_path = "./Data/train/train"
        elif file_path == "v":
            file_path = "./Data/valid/valid"
        X_list = glob(f'{file_path}*.im')
        Y_list = glob(f'{file_path}*.seg')
        data_paths = []
        for x_name in X_list:
            x_id = x_name[-10:-3]
            y_name = f'{file_path}_{x_id}.seg'
            assert y_name in Y_list, "{y_name} is missing in the data file"
            data_paths.append([x_name, y_name])
        return data_paths

    @staticmethod
    def load_file(file):
        with h5py.File(file, 'r') as hf:
            volume = np.array(hf['data'])
        return volume

    @staticmethod
    def sample_from_volume(volume, sample_shape, sample_pos):
        pos_x, pos_y, pos_z = sample_pos
        volume_sample = volume[pos_x: pos_x + sample_shape[0],
                               pos_y: pos_y + sample_shape[1],
                               pos_z: pos_z + sample_shape[2]]
        return volume_sample

    @staticmethod
    def normalise(x_image, mean=None, std=None):
        if mean is None:
            mean = tf.math.reduce_mean(x_image)
        if std is None:
            std = tf.math.reduce_std(x_image)
        return (x_image - mean) / std

    @staticmethod
    def expand_dim_as_float(volume):
        return np.expand_dims(volume, axis=-1).astype(np.float32)

    @staticmethod
    def normalise_position(pos, pos_max):
        """
            - Recieved the pos which is a value from 0 to (length - sample size)
            - A value scaled between -1 and 1 where 0 represents a sample from the centre.
        """
        if pos_max == 0:
            return 0
        return 2 * ((pos / pos_max) - 0.5)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.getcwd())

    add_pos = True
    vol_gen = VolumeGenerator(1, (384, 384, 128), get_position=add_pos, examples_per_load=1)
    x, y = vol_gen.__getitem__(0)
    if add_pos:
        print(x[0].shape)
        print(x[1].shape)
    print(y.shape)
    print(y.dtype)
