import tensorflow as tf
from glob import glob
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
