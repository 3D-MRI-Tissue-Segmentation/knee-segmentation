import tensorflow as tf
# import tensorflow_addons as tfa

def flip_randomly_left_right_image_pair_2d(image_tensor, label_tensor):

    random_var = tf.random.uniform(maxval=2, dtype=tf.int32, shape=[])

    randomly_flipped_img = tf.cond(pred=tf.equal(random_var, 0),
                                   true_fn=lambda: tf.image.flip_left_right(image_tensor),
                                   false_fn=lambda: image_tensor)

    randomly_flipped_label = tf.cond(pred=tf.equal(random_var, 0),
                                     true_fn=lambda: tf.image.flip_left_right(label_tensor),
                                     false_fn=lambda: label_tensor)

    return randomly_flipped_img, randomly_flipped_label

# def rotate_randomly_image_pair_2d(image_tensor, label_tensor, min_angle, max_angle):

#     random_var = tf.random.uniform(maxval=2, dtype=tf.int32, shape=[])
#     random_angle = tf.random.uniform(minval=min_angle, maxval=max_angle, dtype=tf.float32, shape=[])

#     randomly_rotated_img = tf.cond(pred=tf.equal(random_var, 0),
#                                    true_fn=lambda: tfa.image.rotate(image_tensor, random_angle),
#                                    false_fn=lambda: image_tensor)

#     randomly_rotated_label = tf.cond(pred=tf.equal(random_var, 0),
#                                      true_fn=lambda: tfa.image.rotate(label_tensor, random_angle),
#                                      false_fn=lambda: label_tensor)

#     randomly_rotated_label = one_hot_background_2d(randomly_rotated_label)

#     return randomly_rotated_img, randomly_rotated_label

# def translate_randomly_image_pair_2d(image_tensor, label_tensor, dx, dy):

#     random_var = tf.random.uniform(maxval=2, dtype=tf.int32, shape=[])
#     random_dx = tf.random.uniform(minval=-dx, maxval=dx, dtype=tf.int32, shape=[])
#     random_dy = tf.random.uniform(minval=-dy, maxval=dy, dtype=tf.int32, shape=[])

#     randomly_translated_img = tf.cond(pred=tf.equal(random_var, 0),
#                                       true_fn=lambda: tfa.image.translate(image_tensor, [random_dx, random_dy]),
#                                       false_fn=lambda: image_tensor)

#     randomly_translated_label = tf.cond(pred=tf.equal(random_var, 0),
#                                         true_fn=lambda: tfa.image.translate(label_tensor, [random_dx, random_dy]),
#                                         false_fn=lambda: label_tensor)

#     randomly_translated_label = one_hot_background_2d(randomly_translated_label)

#     return randomly_translated_img, randomly_translated_label

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


def apply_valid_random_crop_3d(image_tensor, label_tensor, crop_size, depth_crop_size):

    def body(input_image_tensor, input_label_tensor,
             proposed_image_tensor, proposed_label_tensor, 
             crop_size, depth_crop_size):
        # tf.print("------------------------ cropping --------------------------")
        proposed_image_tensor, proposed_label_tensor = get_random_crop_3d(image_tensor, label_tensor, crop_size, depth_crop_size)
        # tf.print(tf.shape(proposed_image_tensor))
        # tf.print(tf.shape(proposed_label_tensor))
        # tf.print("-- cropping done --")
        return input_image_tensor, input_label_tensor, proposed_image_tensor, proposed_label_tensor, crop_size, depth_crop_size

    def condition(input_image_tensor, input_label_tensor,
                  proposed_image_tensor, proposed_label_tensor, *args):
        # tf.print("=================== condition ======================")
        # tf.print("shape:", tf.shape(proposed_label_tensor), len(tf.shape(proposed_label_tensor)))
        # if tf.shape(proposed_label_tensor) == [2]:
        #     tf.print("assign")
        #     red_sum = proposed_label_tensor
        # else:
        #     tf.print("dont assign")
        red_sum = tf.reduce_sum(proposed_label_tensor, [1,2,3,4])
        # tf.print("red sum:", red_sum)
        # tf.print("logic:", tf.reduce_all((red_sum) < 300000))
        # tf.print("===========================")
        return tf.reduce_all((red_sum) < 100)

    crop_image_tensor, crop_label_tensor = get_random_crop_3d(image_tensor, label_tensor, crop_size, depth_crop_size)
    # tf.print("+=============################============+")
    # tf.print("## shape:", tf.shape(crop_image_tensor), tf.shape(crop_label_tensor))
    # tf.print("## red sum:", tf.reduce_sum(crop_label_tensor, [1,2,3,4]))

    _, _, image_tensor, label_tensor, _, _ = tf.while_loop(
        condition,
        body,
        [image_tensor, label_tensor,
         #tf.constant([[[[[0.0, 0.0]]]]], tf.float32), tf.constant([[[[[0.0, 0.0]]]]], tf.float32), 
         # tf.constant([0.0, 0.0], tf.float32), tf.constant([0.0, 0.0], tf.float32),
         crop_image_tensor, crop_label_tensor,
         crop_size, depth_crop_size],
    # )
         shape_invariants=[image_tensor.get_shape(), label_tensor.get_shape(),
                           tf.TensorShape([None, None, None, None, None]), tf.TensorShape([None, None, None, None, None]),
                           tf.TensorShape([]), tf.TensorShape([]),]
    )
    # tf.print("-- shape:", tf.shape(image_tensor), tf.shape(label_tensor))
    # tf.print("-- red sum:", tf.reduce_sum(label_tensor, [1,2,3,4]))
    # tf.print("+============================+")
    return image_tensor, label_tensor

# def apply_valid_random_crop_3d(image_tensor, label_tensor, crop_size, depth_crop_size):
#     image_tensor, label_tensor = get_random_crop_3d(image_tensor, label_tensor, crop_size, depth_crop_size)
#     return image_tensor, label_tensor

def get_random_crop_3d(image_tensor, label_tensor, crop_size, depth_crop_size):
    centre = (tf.cast(tf.math.divide(tf.shape(image_tensor)[1], 2), tf.int32), 
              tf.cast(tf.math.divide(tf.shape(image_tensor)[2], 2), tf.int32),
              tf.cast(tf.math.divide(tf.shape(image_tensor)[3], 2), tf.int32),
              )
    drc = tf.random.normal([], mean=tf.cast(centre[0], tf.float32), stddev=tf.cast(centre[0] / 4, tf.float32))
    hrc = tf.random.normal([], mean=tf.cast(centre[1], tf.float32), stddev=tf.cast(centre[1] / 4, tf.float32))
    wrc = tf.random.normal([], mean=tf.cast(centre[2], tf.float32), stddev=tf.cast(centre[2] / 4, tf.float32))
    drc = tf.clip_by_value(drc, tf.cast(depth_crop_size, tf.float32), tf.cast(tf.shape(image_tensor)[1] - depth_crop_size, tf.float32))
    hrc = tf.clip_by_value(hrc, tf.cast(crop_size, tf.float32), tf.cast(tf.shape(image_tensor)[2] - crop_size, tf.float32))
    wrc = tf.clip_by_value(wrc, tf.cast(crop_size, tf.float32), tf.cast(tf.shape(image_tensor)[3] - crop_size, tf.float32))
    drc = tf.cast(tf.math.round(drc), tf.int32)
    hrc = tf.cast(tf.math.round(hrc), tf.int32)
    wrc = tf.cast(tf.math.round(wrc), tf.int32)
    centre = (drc, hrc, wrc)
    image_tensor, label_tensor = crop_3d(image_tensor, label_tensor, crop_size, depth_crop_size, centre)
    return image_tensor, label_tensor


def apply_centre_crop_3d(image_tensor, label_tensor, crop_size, depth_crop_size):
    centre = (tf.cast(tf.math.divide(tf.shape(image_tensor)[1], 2), tf.int32),
              tf.cast(tf.math.divide(tf.shape(image_tensor)[2], 2), tf.int32),
              tf.cast(tf.math.divide(tf.shape(image_tensor)[3], 2), tf.int32),
              )
    image_tensor, label_tensor = crop_3d(image_tensor, label_tensor, crop_size, depth_crop_size, centre)
    return image_tensor, label_tensor

def crop_3d(image_tensor, label_tensor, crop_size, depth_crop_size, centre):
    dc, hc, wc = centre
    image_tensor = tf.slice(image_tensor, [0, dc - depth_crop_size, hc - crop_size, wc - crop_size, 0], [-1, depth_crop_size * 2, crop_size * 2, crop_size * 2, -1])
    label_tensor = tf.slice(label_tensor, [0, dc - depth_crop_size, hc - crop_size, wc - crop_size, 0], [-1, depth_crop_size * 2, crop_size * 2, crop_size * 2, -1])
    return image_tensor, label_tensor

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
    mean = tf.math.reduce_mean(image_tensor)
    std = tf.math.reduce_std(image_tensor)
    image_tensor = tf.divide(tf.math.subtract(image_tensor, mean), std)
    return image_tensor, label_tensor

def apply_flip_3d_axis(image_tensor, label_tensor, axis):
    do_flip = tf.random.uniform([]) > 0.5
    image_tensor = tf.cond(do_flip, lambda: tf.reverse(image_tensor, [axis]), lambda: image_tensor)
    label_tensor = tf.cond(do_flip, lambda: tf.reverse(label_tensor, [axis]), lambda: label_tensor)
    return image_tensor, label_tensor

def apply_flip_3d(image_tensor, label_tensor):
    image_tensor, label_tensor = apply_flip_3d_axis(image_tensor, label_tensor, axis=-2)
    image_tensor, label_tensor = apply_flip_3d_axis(image_tensor, label_tensor, axis=-3)
    image_tensor, label_tensor = apply_flip_3d_axis(image_tensor, label_tensor, axis=-4)
    return image_tensor, label_tensor

def to_slice(image_tensor, label_tensor):
    dc = tf.cast(tf.math.divide(tf.shape(image_tensor)[1], 2), tf.int32)
    label_tensor = tf.slice(label_tensor, [0, dc, 0, 0, 0], [-1, 1, -1, -1, -1])
    return image_tensor, label_tensor