import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import math
import numpy as np
import os


def plot_train_history_loss(history, multi_class=True, savefig=None):
    # summarize history for loss
    fig, ax = plt.subplots(2, 1)
    if multi_class:
        ax[0].plot(history.history['dice_coef'])
        ax[0].plot(history.history['val_dice_coef'])
        ax[0].plot(history.history['categorical_crossentropy'])
        ax[0].plot(history.history['val_categorical_crossentropy'])
        ax[0].set_title('Model Loss')
        ax[0].set(xlabel='epoch', ylabel='loss')
        ax[0].legend(['train_dice', 'val_dice', 'train_cce', 'val_cce'], loc='upper right')
        ax[0].legend(['train_tversky', 'val_tversky', 'train_cce', 'val_cce'], loc='upper right')
    else:
        ax[0].plot(history.history['dice_coef'])
        ax[0].plot(history.history['val_dice_coef'])
        ax[0].plot(history.history['binary_crossentropy'])
        ax[0].plot(history.history['val_binary_crossentropy'])
        ax[0].set_title('Model Loss')
        ax[0].set(xlabel='epoch', ylabel='loss')
        ax[0].legend(['train_dice', 'val_dice', 'train_bce', 'val_bce'], loc='upper right')
    ax[1].plot(history.history['acc'])
    ax[1].plot(history.history['val_acc'])
    ax[1].set_title('Model Accuracy')
    ax[1].set(xlabel='epoch', ylabel='accuracy')
    ax[1].legend(['train_accuracy', 'val_accuracy'], loc='upper right')
    fig.tight_layout()
    plt.show()

    if savefig is not None:
        filename = os.path.join(savefig, 'training_history.png')
        plt.savefig(filename)

    plt.close()


def visualise_binary(y_true, y_pred, savefig=None):

    batch_size = y_true.shape[0]

    for i in range(batch_size):
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(y_true[i, :, :, 0], cmap='gray')
        ax[0].set_title('Ground Truth')
        ax[1].imshow(y_pred[i, :, :, 0], cmap='gray')
        ax[1].set_title('Prediction')

        fig.tight_layout()
        plt.show()
        if savefig is not None:
            plt.savefig(savefig)


def visualise_multi_class(y_true, y_pred, savefig=None):

    batch_size = y_true.shape[0]

    for i in range(batch_size):

        grd_truth = y_true[i, :, :, :]
        pred = y_pred[i, :, :, :]

        length = int(math.sqrt(y_true.shape[1]))
        channel = y_true.shape[3]

        pred_max = np.argmax(pred, axis=2)
        pred_img_color = label2color(pred_max)
        y_max = np.argmax(grd_truth, axis=2)
        label_img_color = label2color(y_max)

        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(label_img_color / 255)
        ax[0].set_title('Ground Truth')
        ax[1].imshow(pred_img_color / 255)
        ax[1].set_title('Prediction')

        fig.tight_layout()
        plt.show()
        if savefig is not None:
            plt.savefig(savefig)


colour_maps = {
    0: [0, 0, 0],       # background / black
    1: [255, 255, 0],   # yellow
    2: [0, 255, 255],   # cyan
    3: [255, 0, 255],   # pink
    4: [255, 255, 255], # white
    5: [120, 120, 120], # gray
    6: [255, 165, 0]    # orange 
}


def label2color(img):
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(colour_maps[label])
    return img_color

class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self,
                 steps_per_epoch,
                 initial_learning_rate,
                 min_learning_rate,
                 drop,
                 epochs_drop,
                 warmup_epochs):
        super(LearningRateSchedule, self).__init__()
        self.steps_per_epoch = steps_per_epoch
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.drop = drop
        self.epochs_drop = epochs_drop
        self.warmup_epochs = warmup_epochs

    def __call__(self, step):
        lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
        lrate = self.initial_learning_rate
        if self.warmup_epochs >= 1:
            lrate *= lr_epoch / self.warmup_epochs
        epochs_drop = [self.warmup_epochs] + self.epochs_drop
        for index, start_epoch in enumerate(epochs_drop):
            lrate = tf.where(
                lr_epoch >= start_epoch,
                self.update_lr(index, self.min_learning_rate),
                lrate)
        return lrate
    
    def update_lr(self, idx, min_lr):
        
        new_lr = self.initial_learning_rate * self.drop**idx
        if new_lr <  min_lr:
            new_lr = min_lr

        return new_lr

    def get_config(self):
        return {
            'steps_per_epoch': self.steps_per_epoch,
            'initial_learning_rate': self.initial_learning_rate,
        }
