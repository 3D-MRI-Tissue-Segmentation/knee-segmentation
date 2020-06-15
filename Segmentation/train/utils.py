import tensorflow as tf
from glob import glob
import itertools
import math


def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


class LearningRateUpdate:
    def __init__(self,
                 init_lr,
                 drop_ratio,
                 drop_freq,
                 min_lr=1e-7,
                 warmup=5,
                 ):
        self.init_lr = init_lr
        self.drop_ratio = drop_ratio
        self.drop_freq = drop_freq
        self.min_lr = min_lr
        self.warmup = warmup

    def update_lr(self, epoch):
        if epoch < self.warmup:
            return self.init_lr
        new_lr = self.init_lr * math.pow(self.drop_ratio, math.floor((epoch - self.warmup) / self.drop_freq))
        if new_lr < self.min_lr:
            return self.min_lr
        else:
            return new_lr

def get_validation_stride_coords(pad, full_shape, iterator, strides_required):
    coords = [pad]
    last_coord = full_shape - pad
    if not iterator == None: # for when more strides than just corners is required.
        for stride in range(strides_required):
            new_coord = coords[-1] + iterator # is not garanteed to be whole number
            coords.append(new_coord) # adds to coords, we will round at the end
    if (last_coord != coords[0]) and (last_coord != coords[-1]):
        coords.append(last_coord)
    for idx, i in enumerate(coords):
        coords[idx] = int(round(i, 0))
        if idx > 0:
            assert coords[idx] <= (coords[idx-1] + (pad * 2)), f"Missing points since: {coords[idx]} > {coords[idx-1] + (pad * 2)}"
    return coords


def get_validation_spots(crop_size, depth_crop_size, full_shape=(160, 288, 288)):
    
    model_shape = (depth_crop_size * 2, crop_size * 2, crop_size * 2)

    depth_pad = model_shape[0] / 2
    height_pad = model_shape[1] / 2
    width_pad = model_shape[2] / 2

    depth_working = full_shape[0] - model_shape[0]
    height_working = full_shape[1] - model_shape[1]
    width_working = full_shape[2] - model_shape[2]

    depth_strides_required = math.ceil(depth_working / model_shape[0]) + 1
    height_strides_required = math.ceil(height_working / model_shape[1]) + 1
    width_strides_required = math.ceil(width_working / model_shape[2]) + 1

    depth_iterator = None if depth_strides_required == 0 else depth_working / depth_strides_required
    height_iterator = None if height_strides_required == 0 else height_working / height_strides_required
    width_iterator = None if width_strides_required == 0 else width_working / width_strides_required
    
    depth_coords = get_validation_stride_coords(depth_pad, full_shape[0], depth_iterator, depth_strides_required)
    height_coords = get_validation_stride_coords(height_pad, full_shape[1], height_iterator, height_strides_required)
    width_coords = get_validation_stride_coords(width_pad, full_shape[2], width_iterator, width_strides_required)

    coords = [depth_coords, height_coords, width_coords]
    coords = list(itertools.product(*coords))
    coords = [list(ele) for ele in coords]
    return coords


def get_paddings(crop_size, depth_crop_size, full_shape=(160,288,288)):
    coords = get_validation_spots(crop_size, depth_crop_size, full_shape)
    paddings = []
    for i in coords:
        depth = [i[0] - depth_crop_size, full_shape[0] - (i[0] + depth_crop_size)]
        height = [i[1] - crop_size, full_shape[1] - (i[1] + crop_size)]
        width = [i[2] - crop_size, full_shape[2] - (i[2] + crop_size)]

        assert depth[0] + depth[1] + (depth_crop_size * 2) == full_shape[0]
        assert height[0] + height[1] + (crop_size * 2) == full_shape[1]
        assert width[0] + width[1] + (crop_size * 2) == full_shape[2]

        padding = [depth, height, width, [0, 0]]
        paddings.append(padding)
    return paddings, coords
