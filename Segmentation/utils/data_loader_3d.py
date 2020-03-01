import glob
import h5py
import numpy as np
from random import randint
from tensorflow.keras.utils import Sequence
from math import ceil


class VolumeGenerator(Sequence):
    def __init__(self, batch_size, volume_shape, shuffle=True, add_pos=False,
                 file_path='./Data/train/', data_type='train'):
        self.batch_size = batch_size
        self.volume_shape = volume_shape
        self.shuffle = shuffle
        self.add_pos = add_pos

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
        if self.add_pos:
            image_arr = []
            pos_arr = []
        for sample in batch:
            image, y, pos = VolumeGenerator.load_sample(sample, self.volume_shape)
            if self.add_pos:
                image_arr.append(image)
                pos_arr.append(pos)
            else:
                x_train.append(image)
            y_train.append(y)
        if self.add_pos:
            image_arr = np.stack(image_arr, axis=0)
            pos_arr = np.stack(pos_arr, axis=0)
            x_train = [image_arr, pos_arr]
        else:
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
        vol_shape = volume.shape
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

        pos_x = ((rand_x - sample_shape[0]) / (vol_x_shape[0] - sample_shape[0])) - 0.5
        pos_x *= 2
        pos_y = ((rand_y - sample_shape[1]) / (vol_x_shape[1] - sample_shape[1])) - 0.5
        pos_y *= 2
        pos_z = ((rand_z - sample_shape[2]) / (vol_x_shape[2] - sample_shape[2])) - 0.5
        pos_z *= 2
        pos = np.array((pos_x, pos_y, pos_z), dtype=np.float32)

        volume_x_sample = VolumeGenerator.sample_from_volume(volume_x, sample_shape, centre)
        volume_x_sample = np.expand_dims(volume_x, axis=-1)
        volume_x_sample = volume_x.astype(np.float32)
        volume_y = VolumeGenerator.load_file(y)

        volume_y_sample = VolumeGenerator.sample_from_volume(volume_y, sample_shape, centre)
        volume_y_sample = np.any(volume_y, axis=-1)
        volume_y_sample = np.expand_dims(volume_y, axis=-1)
        volume_y_sample = volume_y.astype(np.float32)
        print(np.sum(volume_y_sample))

        # print(volume_x.shape, volume_y.shape, pos)
        return volume_x_sample, volume_y_sample, pos

if __name__ == "__main__":
    add_pos = True
    vol_gen = VolumeGenerator(3, (48, 48, 48), add_pos=add_pos)
    x, y = vol_gen.__getitem__(0)
    if add_pos:
        print(x[0].shape)
        print(x[1].shape)
    print(y.shape)
    print(y.dtype)
