import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import math
import numpy as np

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    loss = (1 - jac) * smooth
    return loss

def dice_coef_loss(y_true, y_pred):
    dice = -dice_loss(y_true, y_pred)
    return dice

def dice_loss(y_true, y_pred):

    szp = K.shape(y_pred)
    img_len = szp[1]*szp[2]*szp[3]

    y_true = K.reshape(y_true,(-1,img_len))
    y_pred = K.reshape(y_pred,(-1,img_len))

    ovlp = K.sum(y_true*y_pred,axis=-1)

    mu = K.epsilon()
    dice = (2.0 * ovlp + mu) / (K.sum(y_true,axis=-1) + K.sum(y_pred,axis=-1) + mu)
    loss = -dice

    return loss

def tversky_loss(y_true, y_pred):
    #hyperparameters
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

def plot_train_history_loss(history, multi_class=True):
    
    # summarize history for loss
    fig, ax = plt.subplots(2,1)
    if multi_class:
        ax[0].plot(history.history['loss'])
        ax[0].plot(history.history['val_loss'])
        ax[0].plot(history.history['categorical_crossentropy'])
        ax[0].plot(history.history['val_categorical_crossentropy'])
        ax[0].set_title('Model Loss')
        ax[0].set(xlabel='epoch', ylabel='loss')
        ax[0].legend(['train_tversky', 'val_tversky', 'train_cce', 'val_cce'], loc='upper right')
        
    else:
        ax[0].plot(history.history['dice_coef_loss'])
        ax[0].plot(history.history['val_dice_coef_loss'])
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
        
        grd_truth = y_true[i,:,:]
        pred = y_pred[i,:,:]

        length = int(math.sqrt(y_true.shape[1]))
        channel = y_true.shape[2]

        #reshape ground truth and predictions back to 2D
        grd_truth = np.reshape(grd_truth, (length,length,channel))
        pred = np.reshape(pred, (length,length,channel))
 
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
