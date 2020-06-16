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


def get_val_coords(model_dim, full_dim, slice_output=False, iterator_increase=0):
    if slice_output:
        coords = list(range(full_dim))
    else:
        pad = model_dim / 2
        working = full_dim - model_dim
        strides_required = math.ceil(working / model_dim)
        iterator = None if strides_required == 0 else (working / strides_required) + iterator_increase
        coords = get_validation_stride_coords(pad, full_dim, iterator, strides_required)
    return coords


def get_validation_spots(crop_size, depth_crop_size, full_shape=(160, 288, 288), slice_output=False, iterator_increase=0):
    model_shape = (depth_crop_size * 2, crop_size * 2, crop_size * 2)

    depth_coords = get_val_coords(model_shape[0], full_shape[0], slice_output, iterator_increase=iterator_increase)
    height_coords = get_val_coords(model_shape[1], full_shape[1], iterator_increase=iterator_increase)
    width_coords = get_val_coords(model_shape[2], full_shape[2], iterator_increase=iterator_increase)

    coords = [depth_coords, height_coords, width_coords]
    coords = list(itertools.product(*coords))
    coords = [list(ele) for ele in coords]
    return coords


def get_paddings(crop_size, depth_crop_size, full_shape=(160,288,288), iterator_increase=1):
    coords = get_validation_spots(crop_size, depth_crop_size, full_shape, iterator_increase=iterator_increase)
    paddings = []
    for i in coords:
        depth = [i[0] - depth_crop_size, full_shape[0] - (i[0] + depth_crop_size)]
        height = [i[1] - crop_size, full_shape[1] - (i[1] + crop_size)]
        width = [i[2] - crop_size, full_shape[2] - (i[2] + crop_size)]

        assert depth[0] + depth[1] + (depth_crop_size * 2) == full_shape[0]
        assert height[0] + height[1] + (crop_size * 2) == full_shape[1]
        assert width[0] + width[1] + (crop_size * 2) == full_shape[2]

        padding = [[0, 0], depth, height, width, [0, 0]]
        paddings.append(padding)
    return paddings, coords

def get_slice_paddings(crop_size, depth_crop_size, full_shape=(160,288,288), slice_output=True):
    coords = get_validation_spots(crop_size, depth_crop_size, full_shape, slice_output)
    paddings = []
    for i in coords:
        depth_lower = i[0] - depth_crop_size
        depth_upper = full_shape[0] - (i[0] + 1 + depth_crop_size)
        
        depth = [depth_lower, depth_upper]
        height = [i[1] - crop_size, full_shape[1] - (i[1] + crop_size)]
        width = [i[2] - crop_size, full_shape[2] - (i[2] + crop_size)]

        assert depth[0] + depth[1] + (depth_crop_size * 2) + 1 == full_shape[0]
        assert height[0] + height[1] + (crop_size * 2) == full_shape[1]
        assert width[0] + width[1] + (crop_size * 2) == full_shape[2]

        padding = [[0, 0], depth, height, width, [0, 0]]
        paddings.append(padding)
    return paddings, coords


class Metric():
    def __init__(self, metrics):
        self.metrics = metrics

    def store_metric(self, y, predictions, training=False):
        training = 0 if training else 1
        for metric_loss in self.metrics:
            for metric in self.metrics[metric_loss]:
                if metric_loss == 'metrics':
                    self.metrics[metric_loss][metric][training](y, predictions)
                else:
                    m_loss = self.metrics[metric_loss][metric][0](y, predictions)
                    self.metrics[metric_loss][metric][training + 1](m_loss)

    def reset_metrics_get_str(self):
        metric_str = ""
        for metric_loss in self.metrics:
            for metric in self.metrics[metric_loss]:
                for training in range(2):
                    val = "" if training else "val_"
                    pos = 0 if training else 1
                    if metric_loss == 'metrics':
                        metric_str += f" - {val}{metric}: {self.metrics[metric_loss][metric][pos].result():.06f}"
                    else:
                        metric_str += f" - {val}{metric}: {self.metrics[metric_loss][metric][pos + 1].result():.06f}"
        return metric_str

    def add_metric_summary_writer(self, log_dir_now):
        for metric_loss in self.metrics:
            for metric in self.metrics[metric_loss]:
                for training in range(2):
                    val = "" if training else "val_"
                    pos = -2 if training else -1
                    self.metrics[metric_loss][metric][pos] = tf.summary.create_file_writer(log_dir_now + f'/{val}{metric}')

    def record_metric_to_summary(self, e):
        for metric_loss in self.metrics:
            for metric in self.metrics[metric_loss]:
                for training in range(2):    
                    pos = -2 if training else -1
                    with self.metrics[metric_loss][metric][pos].as_default():
                        tf.summary.scalar('metrics', self.metrics[metric_loss][metric][pos - 2].result(), step=e)
