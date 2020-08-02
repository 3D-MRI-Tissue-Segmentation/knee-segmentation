from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
import tensorflow.keras.backend as K
import tensorflow as tf

epsilon = 1e-5
smooth = 1

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def dsc(y_true, y_pred):
    # https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    # https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    loss = 1 - dsc(y_true, y_pred)
    return loss


def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-10):
    """ Tversky loss function.
    Parameters
    ----------
    y_true : tensor containing target mask.
    y_pred : tensor containing predicted mask.
    alpha : real value, weight of '0' class.
    beta : real value, weight of '1' class.
    smooth : small real value used for avoiding division by zero error.
    Returns
    -------
    tensor
        tensor containing tversky loss.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)

    return 1 - answer


def tversky_crossentropy(y_true, y_pred):

    tversky = tversky_loss(y_true, y_pred)
    crossentropy = K.categorical_crossentropy(y_true, y_pred)
    crossentropy = K.mean(crossentropy)

    return tversky + crossentropy


def iou_loss(y_true, y_pred, smooth=1):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou


def bce_dice_loss(y_true, y_pred):
    # https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    loss = binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)
    return loss


def cce_dice_loss(y_true, y_pred):
    # https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    loss = categorical_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)
    return loss


def iou_loss_eval_3d(y_true, y_pred):
    y_true = tf.slice(y_true, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 6])
    y_pred = tf.slice(y_pred, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 6])
    iou = iou_loss(y_true, y_pred)
    return iou


def dice_coef_eval_3d(y_true, y_pred):
    y_true = tf.slice(y_true, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 6])
    y_pred = tf.slice(y_pred, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 6])
    dice = dsc(y_true, y_pred)
    return dice


def dice_loss_weighted_3d(y_true, y_pred):
    dice_whole = 1 - dsc(y_true, y_pred)
    y_true = tf.slice(y_true, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 6])
    y_pred = tf.slice(y_pred, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 6])
    dice = 1 - dsc(y_true, y_pred)
    return dice + dice_whole


def precision(y_true, y_pred):
    # https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    prec = (tp + smooth) / (tp + fp + smooth)
    return prec


def recall(y_true, y_pred):
    # https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall


def confusion(y_true, y_pred):
    # https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return prec, recall


def tp(y_true, y_pred):
    # https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)
    return tp


def tn(y_true, y_pred):
    # https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
    return tn


def focal_tversky(y_true, y_pred):
    # https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    pt_1 = tversky_loss(y_true, y_pred)
    gamma = 0.75
    return K.pow((pt_1), gamma)


def weighted_cat_cross_entropy(y_true, y_pred, class_weights):
    class_weights = tf.reduce_sum(y_true, axis=-1, keepdims=True) / tf.reduce_sum(y_true)

    weights = tf.reduce_sum(class_weights * tf.cast(y_true, tf.float32), axis=-1)
    unweighted_losses = categorical_crossentropy(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32))
    weighted_losses = tf.cast(unweighted_losses, tf.float32) * tf.cast(weights, tf.float32)

    loss = tf.reduce_mean(weighted_losses)
    return loss
