import tensorflow as tf
import numpy as np
from Segmentation.utils.data_loader import read_tfrecord_3d
from Segmentation.utils.augmentation import crop_3d
from Segmentation.train.utils import get_paddings
from Segmentation.utils.losses import dice_loss
import os
from time import time


def validate_best_model(model, val_batch_size, buffer_size, tfrec_dir, multi_class,
                        crop_size, depth_crop_size, predict_slice):
    valid_ds = read_tfrecord_3d(tfrecords_dir=os.path.join(tfrec_dir, 'valid_3d/'), batch_size=val_batch_size, buffer_size=buffer_size, 
                                is_training=False, use_keras_fit=False, multi_class=multi_class)
    vad_padding, val_coord = get_paddings(crop_size, depth_crop_size)
    
    total_loss, total_count = 0.0, 0.0
    for idx,ds in enumerate(valid_ds):
        t0 = time()
        x, y = ds
        centre = [int(y.shape[1]/2), int(y.shape[2]/2), int(y.shape[3]/2)]
        y_crop = tf.cast(crop_3d(y, 144, 80, centre), tf.float32)
        mean_pred = np.ones(tf.shape(y_crop))
        counter = np.ones(tf.shape(y_crop))
        for pad, centre in zip(vad_padding, val_coord):
            x_model_crop = crop_3d(x, crop_size, depth_crop_size, centre)
            pred = model.predict(x_model_crop)
            del x_model_crop
            output_shape = pred.shape
            pred = np.pad(pred, pad, "constant")
            mean_pred += pred
            del pred
            count = np.ones(output_shape)
            count = np.pad(count, pad, "constant")
            counter += count
            del count
        mean_pred = np.divide(mean_pred, counter, dtype=np.float32)
        del counter
        loss = dice_loss(y_crop, mean_pred)
        total_loss += loss
        total_count += 1
        print(f"Validating for: {idx} - {time() - t0:.0f} s")
        if idx == 1:
            break
    total_loss /= total_count
    print("Dice Validation Loss:", total_loss)
    return total_loss
