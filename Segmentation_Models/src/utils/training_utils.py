import tensorflow as tf

def cross_entropy_loss(labels, logits, n_classes, loss_mask=None, data_format='channels_last', one_hot_labels=True, name='ce_loss'):
    """
    Cross-entropy loss.
    :param labels: 4D tensor
    :param logits: 4D tensor
    :param n_classes: integer for number of classes
    :param loss_mask: binary 4D tensor, pixels to mask should be marked by 1s
    :param data_format: string
    :param one_hot_labels: bool, indicator for whether labels are to be expected in one-hot representation
    :param name: string
    :return: dict of (pixel-wise) mean and sum of cross-entropy loss
    """
    # permute class channels into last axis
    if data_format == 'channels_first':
        labels = tf.transpose(labels, [0,2,3,1])
        logits = tf.transpose(logits, [0,2,3,1])

    batch_size = tf.cast(tf.shape(labels)[0], tf.float32)

    if one_hot_labels:
        flat_labels = tf.reshape(labels, [-1, n_classes])
    else:
        flat_labels = tf.reshape(labels, [-1])
        flat_labels = tf.one_hot(indices=flat_labels, depth=n_classes, axis=-1)
    flat_logits = tf.reshape(logits, [-1, n_classes])

    # do not compute gradients wrt the labels
    flat_labels = tf.stop_gradient(flat_labels)

    ce_per_pixel = tf.nn.softmax_cross_entropy_with_logits_v2(labels=flat_labels, logits=flat_logits)

    # optional element-wise masking with binary loss mask
    if loss_mask is None:
        ce_sum = tf.reduce_sum(ce_per_pixel) / batch_size
        ce_mean = tf.reduce_mean(ce_per_pixel)
    else:
        loss_mask_flat = tf.reshape(loss_mask, [-1,])
        loss_mask_flat = (1. - tf.cast(loss_mask_flat, tf.float32))
        ce_sum = tf.reduce_sum(loss_mask_flat * ce_per_pixel) / batch_size
        n_valid_pixels = tf.reduce_sum(loss_mask_flat)
        ce_mean = tf.reduce_sum(loss_mask_flat * ce_per_pixel) / n_valid_pixels

    return {'sum': ce_sum, 'mean': ce_mean}