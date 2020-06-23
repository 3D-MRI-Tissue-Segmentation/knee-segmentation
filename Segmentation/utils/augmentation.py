import tensorflow as tf
# import tensorflow_addons as tfa
import random
import sys

def crop_randomly_image_pair_2d(image_tensor, label_tensor):

    random_seed = random.randrange(sys.maxsize)
    random_var = tf.random.uniform(maxval=2, dtype=tf.int32, shape=[])

    randomly_cropped_img = tf.cond(pred=tf.equal(random_var, 0),
                                   true_fn=lambda: tf.image.random_crop(image_tensor,
                                                                        size=[288, 288, 1],
                                                                        seed=random_seed),
                                   false_fn=lambda: tf.image.resize_with_crop_or_pad(image_tensor, 288, 288))

    randomly_cropped_label = tf.cond(pred=tf.equal(random_var, 0),
                                     true_fn=lambda: tf.image.random_crop(label_tensor,
                                                                          size=[288, 288, 7],
                                                                          seed=random_seed),
                                     false_fn=lambda: tf.image.resize_with_crop_or_pad(label_tensor, 288, 288))

    return randomly_cropped_img, randomly_cropped_label

def flip_randomly_left_right_image_pair_2d(image_tensor, label_tensor):

    random_var = tf.random.uniform(maxval=2, dtype=tf.int32, shape=[])

    randomly_flipped_img = tf.cond(pred=tf.equal(random_var, 0),
                                   true_fn=lambda: tf.image.flip_left_right(image_tensor),
                                   false_fn=lambda: image_tensor)

    randomly_flipped_label = tf.cond(pred=tf.equal(random_var, 0),
                                     true_fn=lambda: tf.image.flip_left_right(label_tensor),
                                     false_fn=lambda: label_tensor)

    return randomly_flipped_img, randomly_flipped_label


def adjust_brightness_randomly_image_pair_2d(image_tensor, label_tensor):

    random_var = tf.random.uniform(maxval=2, dtype=tf.int32, shape=[])
    random_delta = tf.random.uniform(maxval=1, dtype=tf.float32, shape=[])

    randomly_brightened_img = tf.cond(pred=tf.equal(random_var, 0),
                                      true_fn=lambda: tf.image.adjust_brightness(image_tensor, delta=random_delta),
                                      false_fn=lambda: image_tensor)

    randomly_brightened_label = tf.cond(pred=tf.equal(random_var, 0),
                                        true_fn=lambda: tf.image.adjust_brightness(label_tensor, delta=random_delta),
                                        false_fn=lambda: label_tensor)

    return randomly_brightened_img, randomly_brightened_label

def adjust_contrast_randomly_image_pair_2d(image_tensor, label_tensor):

    random_var = tf.random.uniform(maxval=2, dtype=tf.int32, shape=[])
    random_contrast_factor = tf.random.uniform(maxval=1, dtype=tf.float32, shape=[])

    random_crop_img = tf.cond(pred=tf.equal(random_var, 0),
                              true_fn=lambda: tf.image.adjust_contrast(image_tensor, random_contrast_factor),
                              false_fn=lambda: image_tensor)

    random_crop_label = tf.cond(pred=tf.equal(random_var, 0),
                                true_fn=lambda: tf.image.adjust_contrast(label_tensor, random_contrast_factor),
                                false_fn=lambda: label_tensor)

    return random_crop_img, random_crop_label

def get_random_batch_centre(image_tensor, crop_size, depth_crop_size, pad=20):
    batch_size = tf.shape(image_tensor)[0]
    centre = (tf.cast(tf.math.divide(tf.shape(image_tensor)[1], 2), tf.int32), 
              tf.cast(tf.math.divide(tf.shape(image_tensor)[2], 2), tf.int32),
              tf.cast(tf.math.divide(tf.shape(image_tensor)[3], 2), tf.int32),
              )
    drc = tf.random.normal([batch_size], mean=tf.cast(centre[0], tf.float32), stddev=tf.cast(centre[0] / 4, tf.float32))
    hrc = tf.random.normal([batch_size], mean=tf.cast(centre[1], tf.float32), stddev=tf.cast(centre[1] / 4, tf.float32))
    wrc = tf.random.normal([batch_size], mean=tf.cast(centre[2], tf.float32), stddev=tf.cast(centre[2] / 4, tf.float32))
    drc = tf.clip_by_value(drc, tf.cast(depth_crop_size + pad, tf.float32), tf.cast(tf.shape(image_tensor)[1] - depth_crop_size - pad, tf.float32))
    hrc = tf.clip_by_value(hrc, tf.cast(crop_size + pad, tf.float32), tf.cast(tf.shape(image_tensor)[2] - crop_size - pad, tf.float32))
    wrc = tf.clip_by_value(wrc, tf.cast(crop_size + pad, tf.float32), tf.cast(tf.shape(image_tensor)[3] - crop_size - pad, tf.float32))
    drc = tf.cast(tf.math.round(drc), tf.int32)
    hrc = tf.cast(tf.math.round(hrc), tf.int32)
    wrc = tf.cast(tf.math.round(wrc), tf.int32)
    centre = (drc, hrc, wrc)
    return centre

def apply_valid_random_crop_3d(image_tensor,
                               label_tensor,
                               crop_size,
                               depth_crop_size,
                               resize,
                               random_shift,
                               output_slice,
                               factor=0.04):

    def crop_per_batch(x, y, centre, crop_size, depth_crop_size, resize, output_slice):
        if resize:
            original_height = tf.cast(tf.shape(x)[1], tf.float32)
            original_width = tf.cast(tf.shape(x)[2], tf.float32)
            new_height = tf.cast(original_height + tf.random.uniform([], minval=-original_height * factor, maxval=original_height * factor), tf.int32)
            new_width = tf.cast(original_width + tf.random.uniform([], minval=-original_width * factor, maxval=original_width * factor), tf.int32)
            x = tf.image.resize(x, [new_height, new_width])
            y = tf.image.resize(y, [new_height, new_width])
            y = tf.math.round(y)
        dc, hc, wc = centre
        if output_slice:
            x = tf.slice(x, [dc - depth_crop_size, hc - crop_size, wc - crop_size, 0], [1 + (depth_crop_size * 2), crop_size * 2, crop_size * 2, -1])
            y = tf.slice(y, [dc, hc - crop_size, wc - crop_size, 0], [1, crop_size * 2, crop_size * 2, -1])
        else:
            x = tf.slice(x, [dc - depth_crop_size, hc - crop_size, wc - crop_size, 0], [depth_crop_size * 2, crop_size * 2, crop_size * 2, -1])
            y = tf.slice(y, [dc - depth_crop_size, hc - crop_size, wc - crop_size, 0], [depth_crop_size * 2, crop_size * 2, crop_size * 2, -1])
        return x, y, centre

    if random_shift:
        centre = get_random_batch_centre(image_tensor, crop_size, depth_crop_size)
        image_tensor, label_tensor, centre = tf.map_fn(lambda x: crop_per_batch(x[0], x[1], x[2], crop_size, depth_crop_size, resize, output_slice), (image_tensor, label_tensor, centre))
    else:
        image_tensor, label_tensor = apply_centre_crop_3d(image_tensor, label_tensor, crop_size, depth_crop_size, output_slice)
    return image_tensor, label_tensor

def apply_centre_crop_3d(image_tensor, label_tensor, crop_size, depth_crop_size, output_slice):
    centre = (tf.cast(tf.math.divide(tf.shape(image_tensor)[1], 2), tf.int32),
              tf.cast(tf.math.divide(tf.shape(image_tensor)[2], 2), tf.int32),
              tf.cast(tf.math.divide(tf.shape(image_tensor)[3], 2), tf.int32),
              )
    image_tensor = crop_3d(image_tensor, crop_size, depth_crop_size, centre, output_slice, False)
    label_tensor = crop_3d(label_tensor, crop_size, depth_crop_size, centre, output_slice, True)
    return image_tensor, label_tensor

def crop_3d(img, crop_size, depth_crop_size, centre, output_slice, target=None):
    dc, hc, wc = centre
    if output_slice:
        if target:
            img = tf.slice(img, [0, dc, hc - crop_size, wc - crop_size, 0], [-1, 1, crop_size * 2, crop_size * 2, -1])
        else:
            img = tf.slice(img, [0, dc - depth_crop_size, hc - crop_size, wc - crop_size, 0], [-1, 1 + (depth_crop_size * 2), crop_size * 2, crop_size * 2, -1])
    else:
        img = tf.slice(img, [0, dc - depth_crop_size, hc - crop_size, wc - crop_size, 0], [-1, depth_crop_size * 2, crop_size * 2, crop_size * 2, -1])
    return img

def crop_3d_pad_slice(img, crop_size, depth_crop_size, centre):
    """ Gets slice with same padding, used for 3D to 3D slice with x """
    dc, hc, wc = centre

    img = tf.slice(img, [0, dc - depth_crop_size, hc - crop_size, wc - crop_size, 0], [-1, 1 + (depth_crop_size * 2), crop_size * 2, crop_size * 2, -1])
    return img


def apply_random_brightness_3d(image_tensor, label_tensor):
    do_brightness = tf.random.uniform([]) > 0.75
    norm = tf.math.abs(tf.random.normal([], 0.0, 0.1))
    norm = tf.clip_by_value(norm, 0, 0.999)
    image_tensor = tf.cond(do_brightness, lambda: tf.image.adjust_brightness(image_tensor, norm), lambda: image_tensor)
    return image_tensor, label_tensor


def apply_random_contrast_3d(image_tensor, label_tensor):
    do_contrast = tf.random.uniform([]) > 0.75
    contrast = tf.random.uniform([], 0.9, 1.1)
    image_tensor = tf.cond(do_contrast, lambda: tf.image.adjust_contrast(image_tensor, contrast), lambda: image_tensor)
    return image_tensor, label_tensor


def apply_random_gamma_3d(image_tensor, label_tensor):
    do_gamma = tf.random.uniform([]) > 0.75
    gamma = tf.random.uniform([], 0.9, 1.1)
    gain = tf.random.uniform([], 0.95, 1.05)
    image_tensor = tf.cond(do_gamma, lambda: tf.image.adjust_gamma(image_tensor, gamma=gamma, gain=gain), lambda: image_tensor)
    return image_tensor, label_tensor


def normalise(image_tensor, label_tensor):
    image_tensor = tf.map_fn(lambda x: normalise_per_batch(x), image_tensor)
    return image_tensor, label_tensor


def normalise_per_batch(image_tensor):
    mean = tf.math.reduce_mean(image_tensor)
    std = tf.math.reduce_std(image_tensor)
    image_tensor = tf.divide(tf.math.subtract(image_tensor, mean), std)
    return image_tensor


def apply_flip_3d_axis(image_tensor, label_tensor, axis):
    do_flip = tf.random.uniform([]) > 0.5
    image_tensor = tf.cond(do_flip, lambda: tf.reverse(image_tensor, [axis]), lambda: image_tensor)
    label_tensor = tf.cond(do_flip, lambda: tf.reverse(label_tensor, [axis]), lambda: label_tensor)
    return image_tensor, label_tensor


def apply_flip_3d(image_tensor, label_tensor):
    image_tensor, label_tensor = tf.map_fn(lambda x: apply_flip_3d_axis(x[0], x[1], axis=-2), (image_tensor, label_tensor))
    image_tensor, label_tensor = tf.map_fn(lambda x: apply_flip_3d_axis(x[0], x[1], axis=-3), (image_tensor, label_tensor))
    image_tensor, label_tensor = tf.map_fn(lambda x: apply_flip_3d_axis(x[0], x[1], axis=-4), (image_tensor, label_tensor))
    return image_tensor, label_tensor


def apply_rotate_3d(image_tensor, label_tensor):
    image_tensor, label_tensor = tf.map_fn(lambda x: rotate_per_batch_3d(x[0], x[1]), (image_tensor, label_tensor))
    return image_tensor, label_tensor


def rotate_per_batch_3d(image_tensor, label_tensor):
    k = tf.random.uniform([], minval=0, maxval=3, dtype=tf.int32)
    image_tensor = tf.image.rot90(image_tensor, k=k)
    label_tensor = tf.image.rot90(label_tensor, k=k)
    return image_tensor, label_tensor

"""
def rotate_randomly_image_pair_2d(image_tensor, label_tensor, min_angle, max_angle):

    random_var = tf.random.uniform(maxval=2, dtype=tf.int32, shape=[])
    random_angle = tf.random.uniform(minval=min_angle, maxval=max_angle, dtype=tf.float32, shape=[])

    randomly_rotated_img = tf.cond(pred=tf.equal(random_var, 0),
                                   true_fn=lambda: tfa.image.rotate(image_tensor, random_angle),
                                   false_fn=lambda: image_tensor)

    randomly_rotated_label = tf.cond(pred=tf.equal(random_var, 0),
                                     true_fn=lambda: tfa.image.rotate(label_tensor, random_angle),
                                     false_fn=lambda: label_tensor)

    randomly_rotated_label = one_hot_background_2d(randomly_rotated_label)

    return randomly_rotated_img, randomly_rotated_label

def translate_randomly_image_pair_2d(image_tensor, label_tensor, dx, dy):

    random_var = tf.random.uniform(maxval=2, dtype=tf.int32, shape=[])
    random_dx = tf.random.uniform(minval=-dx, maxval=dx, dtype=tf.int32, shape=[])
    random_dy = tf.random.uniform(minval=-dy, maxval=dy, dtype=tf.int32, shape=[])

    randomly_translated_img = tf.cond(pred=tf.equal(random_var, 0),
                                      true_fn=lambda: tfa.image.translate(image_tensor, [random_dx, random_dy]),
                                      false_fn=lambda: image_tensor)

    randomly_translated_label = tf.cond(pred=tf.equal(random_var, 0),
                                        true_fn=lambda: tfa.image.translate(label_tensor, [random_dx, random_dy]),
                                        false_fn=lambda: label_tensor)

    randomly_translated_label = one_hot_background_2d(randomly_translated_label)

    return randomly_translated_img, randomly_translated_label

def one_hot_background_2d(label_tensor):

    label_background, label_6ch = tf.split(label_tensor, [1, 6], axis=2)
    label_sum = tf.math.reduce_sum(label_tensor, axis=2)
    label_bool = tf.cast(label_sum, tf.bool)
    label_bool = tf.reshape(label_bool, [-1])
    mask_bool = tf.math.logical_not(label_bool)
    mask = tf.cast(mask_bool, tf.float32)
    mask = tf.reshape(mask, label_background.get_shape())
    new_background = tf.math.add(mask, label_background)
    new_label = tf.concat([new_background, label_6ch], axis=2)

    return new_label
"""