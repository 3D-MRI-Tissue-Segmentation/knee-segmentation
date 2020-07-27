import tensorflow as tf
import numpy as np
from Segmentation.utils.data_loader import read_tfrecord_3d
from Segmentation.utils.augmentation import crop_3d, crop_3d_pad_slice
from Segmentation.utils.losses import dice_loss
from Segmentation.train.reshape import get_mid_vol, get_mid_slice, plot_through_slices
import os
from time import time
import datetime
import itertools
import math
import copy
import imageio


def get_validation_stride_coords(pad, full_shape, iterator, strides_required):
    coords = [pad]
    last_coord = full_shape - pad
    if iterator is None:  # for when more strides than just corners is required.
        for stride in range(strides_required):
            new_coord = coords[-1] + iterator  # is not garanteed to be whole number
            coords.append(new_coord)  # adds to coords, we will round at the end
    if (last_coord != coords[0]) and (last_coord != coords[-1]):
        coords.append(last_coord)
    for idx, i in enumerate(coords):
        coords[idx] = int(round(i, 0))
        if idx > 0:
            assert coords[idx] <= (coords[idx - 1] + (pad * 2)), f"Missing points since: {coords[idx]} > {coords[idx-1] + (pad * 2)}"
    return coords


def get_val_coords(model_dim, full_dim, slice_output=False, iterator_increase=0):
    if slice_output:
        coords = list(range(full_dim))
    else:
        pad = model_dim / 2
        working = full_dim - model_dim
        strides_required = math.ceil(working / model_dim) + iterator_increase
        iterator = None if strides_required == 0 else (working / strides_required)
        coords = get_validation_stride_coords(pad, full_dim, iterator, strides_required)
    return coords


def get_validation_spots(crop_size, depth_crop_size, full_shape=(160, 288, 288), slice_output=False, iterator_increase=0):
    model_shape = (depth_crop_size * 2, crop_size * 2, crop_size * 2)

    depth_coords = get_val_coords(model_shape[0], full_shape[0], slice_output, iterator_increase=iterator_increase)
    height_coords = get_val_coords(model_shape[1], full_shape[1], iterator_increase=iterator_increase)
    width_coords = get_val_coords(model_shape[2], full_shape[2], iterator_increase=iterator_increase)

    coords = [depth_coords, height_coords, width_coords]
    coords = list(itertools.product(*coords))
    coords = [list(ele) for ele in coords]
    return coords


def get_paddings(crop_size, depth_crop_size, full_shape=(160, 288, 288), iterator_increase=1):
    coords = get_validation_spots(crop_size, depth_crop_size, full_shape, iterator_increase=iterator_increase)
    paddings = []
    for i in coords:
        depth = [i[0] - depth_crop_size, full_shape[0] - (i[0] + depth_crop_size)]
        height = [i[1] - crop_size, full_shape[1] - (i[1] + crop_size)]
        width = [i[2] - crop_size, full_shape[2] - (i[2] + crop_size)]

        assert depth[0] + depth[1] + (depth_crop_size * 2) == full_shape[0]
        assert height[0] + height[1] + (crop_size * 2) == full_shape[1]
        assert width[0] + width[1] + (crop_size * 2) == full_shape[2]

        padding = [[0, 0], depth, height, width, [0, 0]]
        paddings.append(padding)
    return paddings, coords


def get_slice_paddings(crop_size, depth_crop_size, full_shape=(160, 288, 288), slice_output=True):
    coords = get_validation_spots(crop_size, depth_crop_size, full_shape, slice_output)
    paddings = []
    for i in coords:
        depth_lower = i[0] - depth_crop_size
        depth_upper = full_shape[0] - (i[0] + 1 + depth_crop_size)

        depth = [depth_lower, depth_upper]
        height = [i[1] - crop_size, full_shape[1] - (i[1] + crop_size)]
        width = [i[2] - crop_size, full_shape[2] - (i[2] + crop_size)]

        assert depth[0] + depth[1] + (depth_crop_size * 2) + 1 == full_shape[0]
        assert height[0] + height[1] + (crop_size * 2) == full_shape[1]
        assert width[0] + width[1] + (crop_size * 2) == full_shape[2]

        padding = [[0, 0], depth, height, width, [0, 0]]
        paddings.append(padding)
    return paddings, coords


def validate_best_model(model, log_dir_now, val_batch_size, buffer_size, tfrec_dir, multi_class,
                        crop_size, depth_crop_size, predict_slice, metrics):
    valid_ds = read_tfrecord_3d(tfrecords_dir=os.path.join(tfrec_dir, 'valid_3d/'), batch_size=val_batch_size, buffer_size=buffer_size,
                                is_training=False, use_keras_fit=False, multi_class=multi_class)

    now = datetime.datetime.now().strftime("/%Y%m%d/%H%M%S")

    if predict_slice:
        vad_padding, val_coord = get_slice_paddings(crop_size, depth_crop_size)
    else:
        vad_padding, val_coord = get_paddings(crop_size, depth_crop_size)
    total_loss, total_count = 0.0, 0.0
    for idx, ds in enumerate(valid_ds):
        t0 = time()
        x, y = ds

        centre = [int(y.shape[1] / 2), int(y.shape[2] / 2), int(y.shape[3] / 2)]
        x_crop = tf.cast(crop_3d(x, 144, 80, centre, False), tf.float32)
        y_crop = tf.cast(crop_3d(y, 144, 80, centre, False), tf.float32)

        mean_pred = np.zeros(tf.shape(y_crop))
        counter = np.zeros(tf.shape(y_crop))

        for pad, iter_centre in zip(vad_padding, val_coord):
            pad_copy = copy.deepcopy(pad)
            iter_centre_c = copy.deepcopy(iter_centre)
            if predict_slice:
                x_ = x_crop.numpy()
                if pad_copy[1][0] < 0:
                    # need to pad before
                    pad_by = pad_copy[1][0] * -1
                    iter_centre_c[0] += pad_by
                    x_[:, pad_by:, :, :, :] = x_[:, :-pad_by, :, :, :]
                    for i in range(pad_by):
                        x_[:, i, :, :, :] = x_[:, iter_centre_c[0], :, :, :]
                    pad_copy[1][0] = 0
                    pad_copy[1][1] = pad_copy[1][1] - pad_by
                elif pad_copy[1][1] < 0:
                    # pad after
                    pad_by = pad_copy[1][1] * -1
                    iter_centre_c[0] -= pad_by
                    x_[:, :pad_by, :, :, :] = x_[:, -pad_by:, :, :, :]
                    for i in range(pad_by):
                        x_[:, -i, :, :, :] = x_[:, iter_centre_c[0], :, :, :]
                    pad_copy[1][1] = 0
                    pad_copy[1][0] = pad_copy[1][0] - pad_by
                pad_copy[1][0] += depth_crop_size
                pad_copy[1][1] += depth_crop_size
                x_model_crop = crop_3d_pad_slice(x_, crop_size, depth_crop_size, iter_centre_c)
                del x_
            else:
                x_model_crop = crop_3d(x_crop, crop_size, depth_crop_size, iter_centre_c, False)

            pred = model.predict(x_model_crop)

            del x_model_crop
            output_shape = pred.shape
            pred = np.pad(pred, pad_copy, "constant")

            mean_pred += pred
            del pred
            count = np.ones(output_shape)
            count = np.pad(count, pad_copy, "constant")
            counter += count
            del count

        mean_pred = np.divide(mean_pred, counter, dtype=np.float32)
        del counter

        loss = dice_loss(y_crop, mean_pred)
        metrics.store_metric(y_crop, mean_pred)

        total_loss += loss
        total_count += 1
        print(f"Validating for: {idx} - {time() - t0:.0f} s")

        vol_writer = tf.summary.create_file_writer(log_dir_now + '/whole_val/img/vol' + now + f'/{idx}')
        slice_writer = tf.summary.create_file_writer(log_dir_now + '/whole_val/img/slice' + now + f'/{idx}')
        slices_writer = tf.summary.create_file_writer(log_dir_now + '/whole_val/img/all_slices' + now + f'/{idx}')

        if idx < 4:  # plot the first 4
            imgs = plot_through_slices(0, x_crop, y_crop, mean_pred, slices_writer, multi_class)
            imageio.mimsave(f'{log_dir_now}/whole_val/img/all_slices/val_{idx}.gif', imgs)

            img = get_mid_slice(x_crop, y_crop, mean_pred, multi_class)
            del x_crop
            with slice_writer.as_default():
                tf.summary.image("Whole Validation - Slice", img, step=idx)

            img = get_mid_vol(y_crop, mean_pred, multi_class)
            with vol_writer.as_default():
                tf.summary.image("Whole Validation - Vol", img, step=idx)

    metric_str = metrics.reset_metrics_get_str()
    total_loss /= total_count
    print("Dice Validation Loss:", total_loss)
    return total_loss, metric_str
