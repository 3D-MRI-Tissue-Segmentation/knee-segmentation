import tensorflow as tf
from glob import glob

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

class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self,
                 steps_per_epoch,
                 initial_learning_rate,
                 drop,
                 drop_freq,
                 min_lr=1e-6):
        super(LearningRateSchedule, self).__init__()
        self.steps_per_epoch = steps_per_epoch
        self.initial_learning_rate = initial_learning_rate
        self.drop = drop
        self.drop_freq = drop_freq
        self._step = 0
        self.min_lr = min_lr

    @tf.function
    def __call__(self, step):
        print("LR STEP")
        tf.print("LR STEP")
        self._step += 1
        tf.print("lr step", self._step)
        lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
        print("lr_epoch", lr_epoch)
        tf.print("lr_epoch", lr_epoch)
        lrate = self.initial_learning_rate * tf.math.pow(self.drop, tf.math.floor((1 + lr_epoch) / self.drop_freq))
        tf.print("lr", lrate)
        if lrate < self.min_lr:
            lrate = self.min_lr
        return lrate

    def get_config(self):
        tf.print("Steps", self._step)
        lr_epoch = tf.cast(self._step, tf.float32) / self.steps_per_epoch
        tf.print("config e", lr_epoch)
        tf.print("config lr", (self.initial_learning_rate * tf.math.pow(self.drop, tf.math.floor((1 + lr_epoch) / self.drop_freq))).numpy())
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'step': self._step,
            'current_learning_rate': (self.initial_learning_rate * tf.math.pow(self.drop, tf.math.floor((1 + lr_epoch) / self.drop_freq))).numpy()
        }