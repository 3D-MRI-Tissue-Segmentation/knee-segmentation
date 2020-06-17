from Segmentation.plotting.voxels import plot_volume, plot_to_image
import tensorflow as tf


colour_maps = {
    1: [tf.constant([1, 1, 1], dtype=tf.float32), tf.constant([[[[255, 255, 0]]]], dtype=tf.float32)],  # background / black
    2: [tf.constant([2, 2, 2], dtype=tf.float32), tf.constant([[[[0, 255, 255]]]], dtype=tf.float32)],
    3: [tf.constant([3, 3, 3], dtype=tf.float32), tf.constant([[[[255, 0, 255]]]], dtype=tf.float32)],
    4: [tf.constant([4, 4, 4], dtype=tf.float32), tf.constant([[[[255, 255, 255]]]], dtype=tf.float32)],
    5: [tf.constant([5, 5, 5], dtype=tf.float32), tf.constant([[[[120, 120, 120]]]], dtype=tf.float32)],
    6: [tf.constant([6, 6, 6], dtype=tf.float32), tf.constant([[[[255, 165, 0]]]], dtype=tf.float32)],
}


def replace_vector(img, search, replace):
    condition = tf.equal(img, search)
    condition = tf.reduce_all(condition, axis=-1)
    condition = tf.stack((condition,) * img.shape[-1], axis=-1)
    replace_tiled = tf.tile(replace, img.shape[:-1])
    replace_tiled = tf.reshape(replace_tiled, img.shape)
    return tf.where(condition, replace_tiled, img)

def get_mid_slice(x, y, pred, multi_class):
    mid = tf.cast(tf.divide(tf.shape(y)[1], 2), tf.int32)
    x_slice = tf.slice(x, [0, mid, 0, 0, 0], [1, 1, -1, -1, -1])
    y_slice = tf.slice(y, [0, mid, 0, 0, 0], [1, 1, -1, -1, -1])
    pred_slice = tf.slice(pred, [0, mid, 0, 0, 0], [1, 1, -1, -1, -1])
    if multi_class:
        x_slice = tf.squeeze(x_slice, axis=-1)
        x_slice = tf.stack((x_slice,) * 3, axis=-1)
        y_slice = tf.argmax(y_slice, axis=-1)
        y_slice = tf.stack((y_slice,) * 3, axis=-1)
        y_slice = tf.cast(y_slice, tf.float32)
        pred_slice = tf.argmax(pred_slice, axis=-1)
        pred_slice = tf.stack((pred_slice,) * 3, axis=-1)
        pred_slice = tf.cast(pred_slice, tf.float32)
        for c in colour_maps:
            y_slice = replace_vector(y_slice, colour_maps[c][0], colour_maps[c][1])
            pred_slice = replace_vector(pred_slice, colour_maps[c][0], colour_maps[c][1])
    else:
        pred_slice = tf.math.round(pred_slice)
    
    img_pad = tf.ones((pred_slice.shape[0], pred_slice.shape[1], pred_slice.shape[2], 3, pred_slice.shape[4]))
    img = tf.concat((x_slice, img_pad, y_slice, img_pad, pred_slice), axis=-2)

    return tf.reshape(img, (img.shape[1:]))

def get_mid_vol(y, pred, multi_class, rad=8):
    y_shape = tf.shape(y)
    y_subvol = tf.slice(y, [0, (y_shape[1] // 2) - rad, (y_shape[2] // 2) - rad, (y_shape[3] // 2) - rad, 0], [1, rad * 2, rad * 2, rad * 2, -1])
    if multi_class:
        y_subvol = tf.argmax(y_subvol, axis=-1)
        y_subvol = tf.cast(y_subvol, tf.float32)
    else:
        y_subvol = tf.reshape(y_subvol, (y_subvol.shape[1:4]))
    y_subvol = tf.stack((y_subvol,) * 3, axis=-1)
    pred_subvol = tf.slice(pred, [0, (y_shape[1] // 2) - rad, (y_shape[2] // 2) - rad, (y_shape[3] // 2) - rad, 0], [1, rad * 2, rad * 2, rad * 2, -1])
    if multi_class:
        pred_subvol = tf.argmax(pred_subvol, axis=-1)
        pred_subvol = tf.cast(pred_subvol, tf.float32)
    else:
        pred_subvol = tf.math.round(pred_subvol)  # new
        pred_subvol = tf.reshape(pred_subvol, (pred_subvol.shape[1:4]))
    pred_subvol = tf.stack((pred_subvol,) * 3, axis=-1)
    if multi_class:
        for c in colour_maps:
            y_subvol = replace_vector(y_subvol, colour_maps[c][0], tf.divide(colour_maps[c][1], 255))
            pred_subvol = replace_vector(pred_subvol, colour_maps[c][0], tf.divide(colour_maps[c][1], 255))
        y_subvol = tf.squeeze(y_subvol, axis=0)
        pred_subvol = tf.squeeze(pred_subvol, axis=0)
    fig = plot_volume(y_subvol, show=False)
    y_img = plot_to_image(fig)
    del y_subvol
    fig = plot_volume(pred_subvol, show=False)
    del pred_subvol
    pred_img = plot_to_image(fig)
    
    img = tf.concat((y_img, pred_img), axis=-2)
    return img
