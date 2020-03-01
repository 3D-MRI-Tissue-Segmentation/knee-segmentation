import glob
import h5py
import numpy as np
from random import randint
from tensorflow.keras.utils import Sequence
from math import ceil


class VolumeGenerator(Sequence):
    def __init__(self, batch_size, volume_shape, shuffle=True,
                 file_path='./Data/train/', data_type='train'):
        self.batch_size = batch_size
        self.volume_shape = volume_shape
        self.shuffle = shuffle

        self.data_paths = VolumeGenerator.get_list_of_data(file_path, data_type)
        assert self.batch_size <= len(self.data_paths), "Batch size must be less than or equal to number of training examples"
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return ceil(len(self.data_paths) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.data_paths[idx] for idx in indexes]
        x, y = self.generate_batch(batch)
        return x, y

    def generate_batch(self, batch):
        x_train = []
        y_train = []
        for sample in batch:
            x, y = VolumeGenerator.load_sample(sample, self.volume_shape)
            x_train.append(x)
            y_train.append(y)

        x_train = np.stack(x_train, axis=0)
        y_train = np.stack(y_train, axis=0)
        return x_train, y_train

    @staticmethod
    def get_list_of_data(file_path='./Data/train/', data_type='train'):
        X_list = glob.glob(f'{file_path}*.im')
        Y_list = glob.glob(f'{file_path}*.seg')
        data = []
        for x_name in X_list:
            x_id = x_name[-10:-3]
            y_name = f'{file_path}{data_type}_{x_id}.seg'
            assert y_name in Y_list, "{y_name} is missing in the data file"
            data.append([x_name, y_name])
        return data

    @staticmethod
    def load_file(file):
        with h5py.File(file, 'r') as hf:
            volume = np.array(hf['data'])
        return volume

    @staticmethod
    def sample_from_volume(volume, sample_shape, centre):
        rand_x, rand_y, rand_z = centre
        volume_sample = volume[rand_x - sample_shape[0]: rand_x,
                               rand_y - sample_shape[1]: rand_y,
                               rand_z - sample_shape[2]: rand_z]
        return volume_sample

    @staticmethod
    def load_sample(sample, sample_shape):
        x, y = sample

        volume_x = VolumeGenerator.load_file(x)
        vol_x_shape = volume_x.shape

        rand_x = randint(sample_shape[0], vol_x_shape[0])
        rand_y = randint(sample_shape[1], vol_x_shape[1])
        rand_z = randint(sample_shape[2], vol_x_shape[2])
        centre = rand_x, rand_y, rand_z

        volume_x = VolumeGenerator.sample_from_volume(volume_x, sample_shape, centre)
        volume_x = np.expand_dims(volume_x, axis=-1)
        volume_y = VolumeGenerator.load_file(y)
        volume_y = VolumeGenerator.sample_from_volume(volume_y, sample_shape, centre)
        volume_y = volume_y.astype(np.float32)
        return volume_x, volume_y

if __name__ == "__main__":
    vol_gen = VolumeGenerator(4, (48, 48, 48))
    x, y = vol_gen.__getitem__(0)
    print(x.shape)
    print(x.dtype)
    print(y.shape)
    print(y.dtype)
