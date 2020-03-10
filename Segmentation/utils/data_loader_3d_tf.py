import tensorflow as tf
from glob import glob
import numpy as np
import h5py


def generator(data_paths, sample_shape,
              normalise_input, remove_outliers,
              transform_angle, transform_position,
              get_slice, get_position, skip_empty):
    for paths in data_paths:
        x_path, y_path = paths
        volume_x = load_file(x_path)
        volume_y = load_file(y_path)

        sample_pos, sample_pos_max = get_sample_pos(volume_x.shape, sample_shape, transform_position)

        if not np.array_equal(volume_x.shape, sample_shape):
            volume_x = sample_from_volume(volume_x, sample_shape, sample_pos)
            volume_y = sample_from_volume(volume_y, sample_shape, sample_pos)
        volume_y = np.any(volume_y, axis=-1)

        if normalise_input or remove_outliers:
            mean = tf.math.reduce_mean(volume_x)
            if remove_outliers:
                np.clip(volume_x, None, 0.01, volume_x)
            if normalise_input:
                volume_x = normalise(volume_x, mean)

        volume_x = expand_dim_as_float(volume_x)
        volume_y = expand_dim_as_float(volume_y)

        if get_slice:
            slice_idx = int((sample_shape[2] + 1) / 2) - 1
            assert slice_idx >= 0
            volume_y = volume_y[:, :, slice_idx]

        if skip_empty:
            if np.sum(volume_y) == 0:
                continue

        if get_position:
            pos = np.empty(3)
            for i in range(3):
                pos[i] = normalise_position(sample_pos[i], sample_pos_max[i])
            yield [volume_x, pos], volume_y
        else:
            yield volume_x, volume_y

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

def normalise_position(pos, pos_max):
    """
        - Recieved the pos which is a value from 0 to (length - sample size)
        - A value scaled between -1 and 1 where 0 represents a sample from the centre.
    """
    if pos_max == 0:
        pos = 0
    else:
        pos = 2 * ((pos / pos_max) - 0.5)
    return pos

def sample_from_volume(volume, sample_shape, sample_pos):
    pos_x, pos_y, pos_z = sample_pos
    volume_sample = volume[pos_x: pos_x + sample_shape[0],
                           pos_y: pos_y + sample_shape[1],
                           pos_z: pos_z + sample_shape[2]]
    return volume_sample

def normalise(x_image, mean=None, std=None):
    if mean is None:
        mean = tf.math.reduce_mean(x_image)
    if std is None:
        std = tf.math.reduce_std(x_image)
    return (x_image - mean) / std

def load_file(file):
    with h5py.File(file, 'r') as hf:
        volume = np.array(hf['data'])
    return volume

def expand_dim_as_float(volume):
    return np.expand_dims(volume, axis=-1).astype(np.float32)

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

def get_dataset(file_path="t",
                sample_shape=(384, 384, 160),
                normalise_input=True,
                remove_outliers=True,
                transform_angle=False,
                transform_position=False,
                get_slice=False,
                get_position=False,
                skip_empty=True):
    data_paths = get_paths(file_path)
    steps = len(data_paths)
    output_shape = (*sample_shape, 1)
    if get_position:
        output_types = [[tf.float32, tf.float32], tf.float32]
        output_shapes = [[output_shape, (3,)], output_shape]
    else:
        output_types = [tf.float32, tf.float32]
        output_shapes = [output_shape, output_shape]
    if get_slice:
        output_shapes[-1] = (sample_shape[0], sample_shape[1], 1)
    output_types = tuple(output_types)
    output_shapes = tuple(output_shapes)

    print(output_shapes)
    print(output_types)
    print(output_shape)

    return tf.data.Dataset.from_generator(
        generator,
        output_types=output_types,
        output_shapes=output_shapes,
        args=(data_paths, output_shape,
              normalise_input, remove_outliers,
              transform_angle, transform_position,
              get_slice, get_position, skip_empty)
    ), steps

if __name__ == "__main__":

    import time

    def benchmark(dataset, num_epochs=1):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            for sample in dataset:
                # Performing a training step
                time.sleep(0.1)
        tf.print("Execution time:", time.perf_counter() - start_time)

    benchmark(get_dataset(file_path="t",
                          sample_shape=(40, 40, 20),
                          transform_position="normal",
                          get_slice=True,
                          normalise_input=False,
                          get_position=True).batch(2))

    benchmark(get_dataset(file_path="t",
                          sample_shape=(40, 40, 20),
                          transform_position="normal",
                          get_slice=True,
                          normalise_input=False,
                          get_position=True).prefetch(tf.data.experimental.AUTOTUNE))

    import sys
    from os import getcwd
    sys.path.insert(0, getcwd())

    
