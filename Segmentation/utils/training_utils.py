import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import math
import numpy as np

def iou_loss_core(y_true, y_pred, smooth=1):

    y_true = K.cast_to_floatx(y_true)
    y_pred = K.cast_to_floatx(y_pred)

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou

def Mean_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.cast((true_labels & pred_labels), tf.int32)
        union = tf.cast((true_labels | pred_labels), tf.int32)
        legal_batches = K.sum(tf.cast(true_labels, tf.int32), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(ious[legal_batches]))
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = iou[legal_labels]
    return K.mean(iou)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred,smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_f = K.cast_to_floatx(y_true_f)
    y_pred_f = K.cast_to_floatx(y_pred_f)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-10):
    """ Tversky loss function.
    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    y_true = K.flatten(y_true)
    y_true = K.cast_to_floatx(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return -answer

def tversky_crossentropy(y_true, y_pred):

    tversky = tversky_loss(y_true, y_pred)
    crossentropy = K.categorical_crossentropy(y_true, y_pred)
    crossentropy = K.mean(crossentropy)

    return tversky + crossentropy

def iou_loss_core(y_true, y_pred, smooth=1):
    y_true = K.cast_to_floatx(y_true)
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou

def plot_train_history_loss(history, multi_class=True):
    
    # summarize history for loss
    fig, ax = plt.subplots(2,1)
    if multi_class:
        ax[0].plot(history.history['dice_coef'])
        ax[0].plot(history.history['val_dice_coef'])
        ax[0].plot(history.history['categorical_crossentropy'])
        ax[0].plot(history.history['val_categorical_crossentropy'])
        ax[0].set_title('Model Loss')
        ax[0].set(xlabel='epoch', ylabel='loss')
        ax[0].legend(['train_dice', 'val_dice', 'train_cce', 'val_cce'], loc='upper right')
        
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
    
def visualise_binary(y_true, y_pred):

    y_true = np.expand_dims(y_true,axis=3)
    y_pred = np.expand_dims(y_pred,axis=3)
    batch_size = y_true.shape[0]

    for i in range(batch_size):
        fig, ax = plt.subplots(2,1)
        ax[0].imshow(y_true[i,:,:,0], cmap='gray')
        ax[0].set_title('Ground Truth')
        ax[1].imshow(y_pred[i,:,:,0], cmap='gray')
        ax[1].set_title('Prediction')
        
        fig.tight_layout()
        plt.show()

def visualise_multi_class(y_true, y_pred):

    batch_size = y_true.shape[0]

    for i in range(batch_size):
        
        grd_truth = y_true[i,:,:,:]
        pred = y_pred[i,:,:,:]

        length = int(math.sqrt(y_true.shape[1]))
        channel = y_true.shape[3]
 
        pred_max = np.argmax(pred, axis=2)
        pred_img_color = label2color(pred_max)
        y_max = np.argmax(grd_truth, axis=2)
        label_img_color = label2color(y_max)

        fig, ax = plt.subplots(2,1)
        ax[0].imshow(label_img_color/255)
        ax[0].set_title('Ground Truth')
        ax[1].imshow(pred_img_color/255)
        ax[1].set_title('Prediction')

        fig.tight_layout()
        plt.show()
    
def label2color(img):
    
    colour_maps = {
        0: [255,0,0],
        1: [0,255,0],
        2: [0,0,255],
        3: [128,64,255],
        4: [70,255,70],
        5: [255,20,147],
        6: [0,0,0]
    }
    
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(colour_maps[label])

    return img_color
    
def make_lr_scheduler(init_lr):

    def step_decay(epoch):
        drop = 0.5
        epochs_drop = 1.0
        lrate = init_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    return tf.keras.callbacks.LearningRateScheduler(step_decay)

class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self,
               steps_per_epoch,
               initial_learning_rate,
               drop,
               epochs_drop):
    super(LearningRateSchedule, self).__init__()
    self.steps_per_epoch = steps_per_epoch
    self.initial_learning_rate = initial_learning_rate
    self.drop = drop
    self.epochs_drop = epochs_drop

  def __call__(self, step):
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    lrate = self.initial_learning_rate * tf.math.pow(self.drop, tf.math.floor((1+lr_epoch)/self.epochs_drop))
    return lrate

  def get_config(self):
    return {
        'initial_learning_rate': self.initial_learning_rate,
    }