import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, normalise=True):

    cm = confusion_matrix(y_true, y_pred)

    if normalise:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]

    plt.matshow(cm)
    plt.show()
