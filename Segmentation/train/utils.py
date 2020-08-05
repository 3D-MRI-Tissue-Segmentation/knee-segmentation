import tensorflow as tf
import math

# Check that this class is even necessary
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
