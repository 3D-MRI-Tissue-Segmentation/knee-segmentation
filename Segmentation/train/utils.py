import tensorflow as tf
import math

"""
def setup_accelerator()
    
"""


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
                        self.metrics[metric_loss][metric][pos].reset_states()
                    else:
                        metric_str += f" - {val}{metric}: {self.metrics[metric_loss][metric][pos + 1].result():.06f}"
                        self.metrics[metric_loss][metric][pos + 1].reset_states()
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
