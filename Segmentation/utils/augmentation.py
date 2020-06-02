import tensorflow as tf
import tensorflow_addons as tfa

def flip_randomly_left_right_image_pair_2d(image_tensor, label_tensor):

    random_var = tf.random.uniform(maxval=2, dtype=tf.int32, shape=[])

    randomly_flipped_img = tf.cond(pred=tf.equal(random_var, 0),
                                   true_fn=lambda: tf.image.flip_left_right(image_tensor),
                                   false_fn=lambda: image_tensor)

    randomly_flipped_label = tf.cond(pred=tf.equal(random_var, 0),
                                     true_fn=lambda: tf.image.flip_left_right(label_tensor),
                                     false_fn=lambda: label_tensor)

    return randomly_flipped_img, randomly_flipped_label

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
