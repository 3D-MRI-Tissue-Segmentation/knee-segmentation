import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import os
from Segmentation.utils.losses import dice_coef, iou_loss

def iou_loss_eval(y_true, y_pred, drop_class_idx=0):

    y_true = np.delete(y_true, drop_class_idx, axis=-1)
    y_pred = np.delete(y_true, drop_class_idx, axis=-1)
    iou = iou_loss(y_true, y_pred)

    return iou

def dice_coef_eval(y_true, y_pred, drop_class_idx=0):

    y_true = np.delete(y_true, drop_class_idx, axis=-1)
    y_pred = np.delete(y_true, drop_class_idx, axis=-1)

    dice = dice_coef(y_true, y_pred)

    return dice

def get_confusion_matrix(y_true, y_pred, classes=None):

    y_true = np.reshape(y_true, (y_true.shape[0] * y_true.shape[1] * y_true.shape[2], y_true.shape[3]))
    y_pred = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2], y_pred.shape[3]))
    y_true_max = np.argmax(y_true, axis=1)
    y_pred_max = np.argmax(y_pred, axis=1)

    if classes is None:
        cm = confusion_matrix(y_true_max, y_pred_max)
    else:
        cm = confusion_matrix(y_true_max, y_pred_max, labels=classes)
    print(cm)

    return cm

def plot_confusion_matrix(cm, savefig, classes, normalise=True, title='confusion matrix', cmap=plt.cm.Blues):

    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    if savefig is not None:
        plt.savefig(savefig)