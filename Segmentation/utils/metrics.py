import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.metrics import MeanMetricWrapper

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def IoU(y_true, y_pred, smooth=1):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou

def dice_coef_single(y_true, y_pred, idx):

    y_true = y_true[..., idx]
    y_pred = y_pred[..., idx]

    return dice_coef(y_true, y_pred)

def IoU_single(y_true, y_pred, idx):

    y_true = y_true[..., idx]
    y_pred = y_pred[..., idx]

    return IoU(y_true, y_pred)

class DiceMetrics(MeanMetricWrapper):
    
    def __init__(self, idx, dtype=None):
        name = 'dice_coef_{}'.format(idx)
        super(DiceMetrics, self).__init__(
            dice_coef_single, name, dtype=dtype, idx=idx)

class IoUMetrics(MeanMetricWrapper):

    def __init__(self, idx, dtype=None):
        name = 'IoU_{}'.format(idx)
        super(IoUMetrics, self).__init__(
            IoU_single, name, dtype=dtype, idx=idx)

def dice_coef_eval(y_true, y_pred):

    # remove background_classes
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]

    return dice_coef(y_true, y_pred)

def IoU_eval(y_true, y_pred):

    # remove background_classes
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]

    return IoU(y_true, y_pred)

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

