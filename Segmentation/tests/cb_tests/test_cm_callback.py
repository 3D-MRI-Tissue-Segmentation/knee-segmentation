# Testing the confusion matrix callback
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from absl import app
from datetime import datetime
import os

DATASET_SIZE = 20


def get_confusion_matrix_cb(epoch, logs):
    """ Lambda Callback -ready version of get_conusion_matrix """
    train_sample, train_label = train_ds
    val_sample, val_label = validation_ds

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

############# TOY INPUT #############
# toy_sample_3d = np.random.randint(0, 5, size=(3,3,3,7))
# toy_pred_3d = np.random.randint(0, 5, size=(3,3,3,7))
toy_sample_2d = np.random.randint(0, 5, size=(3,3,1))
toy_pred_2d = np.random.randint(0, 5, size=(3,3,1))
toy_sample_3d = np.expand_dims(toy_sample_2d, axis=2)
toy_sample_3d = np.pad(toy_sample_3d, ((0,0),(0,0),(0,2),(0,0)), mode='constant', constant_values=0)
toy_pred_3d = np.expand_dims(toy_pred_2d, axis=2)
toy_pred_3d = np.pad(toy_pred_3d, ((0,0),(0,0),(0,2),(0,0)), mode='constant', constant_values=0)

toy_sample_3d = tf.convert_to_tensor(toy_sample_3d, dtype=tf.float32)
toy_pred_3d = tf.convert_to_tensor(toy_pred_3d, dtype=tf.float32)
toy_sample_2d = tf.convert_to_tensor(toy_sample_2d, dtype=tf.float32)
toy_pred_2d = tf.convert_to_tensor(toy_pred_2d, dtype=tf.float32)

train_examples = [np.random.randint(0, 5, size=(3,3,3,1)).astype('float32') for i in range(DATASET_SIZE//4)]

train_ds = tf.data.Dataset.from_tensor_slices((train_examples, train_examples))
validation_ds = tf.data.Dataset.from_tensor_slices((train_examples, train_examples))

############# TOY INPUT #############


############# MODEL SELECT #############
from Segmentation.tests.flags import FLAGS
from Segmentation.tests.select_model import select_model
from Segmentation.utils.training_utils import LearningRateSchedule

from Segmentation.utils.losses import dice_coef_loss, tversky_loss
from Segmentation.utils.metrics import dice_coef, mIoU
from Segmentation.utils.evaluation_metrics import dice_coef_eval, iou_loss_eval

# epochs=10
# model_name=7
# num_classes=7
# log_dir_now=None
# batch_size=32
# val_batch_size=32
# lr=1e-4
# lr_drop=0.9
# lr_drop_freq=5
# lr_warmup=3
# num_to_visualise=2
# num_channels=[4, 8, 16]
# buffer_size=1000
# run_eager=True
# tfrec_dir='./Data/tfrecords/'
# multi_class=False
# crop_size=288
# depth_crop_size=None
# aug=[]
# use_2d=True
# debug=False
# predict_slice=False
# tpu_name=None
# num_cores=8
# min_lr=1e-7
# custom_loss=None
# use_bfloat16=False
# use_RGB=False
# verbose=True

# if custom_loss is None:
#     loss_func = tversky_loss if multi_class else dice_coef_loss
# elif multi_class and custom_loss == "weighted":
#     loss_func = weighted_cat_cross_entropy
# elif multi_class and custom_loss == "focal":
#     loss_func = focal_tversky
# else:
#     raise NotImplementedError(f"Custom loss: {custom_loss} not implemented.")

# # rewrite a function that takes in model-specific arguments and returns model_fn
# model = select_model(model_name, num_classes, num_channels, use_2d, **model_kwargs)

# batch_size = batch_size * num_cores

# # Fix hard-coding ad check that the lr_drop_freq is a list, not int
# lr = LearningRateSchedule(19200 // batch_size, lr, min_lr, lr_drop, lr_drop_freq, lr_warmup)
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

# if multi_class:
#     metrics = {}
#     for i in range(num_classes):
#         train_dice_name = 'train/dice_' + str(i + 1)
#         train_iou_name = 'train/iou_' + str(i + 1)
#         valid_dice_name = 'valid/dice_' + str(i + 1)
#         valid_iou_name = 'valid/iou_' + str(i + 1)

#         metrics[train_dice_name] = tf.keras.metrics.Mean()
#         metrics[train_iou_name] = tf.keras.metrics.Mean()
#         metrics[valid_dice_name] = tf.keras.metrics.Mean()
#         metrics[valid_iou_name] = tf.keras.metrics.Mean()
# else:
#     metrics['train/dice'] = tf.keras.metrics.Mean()
#     metrics['train/iou'] = tf.keras.metrics.Mean()
#     metrics['valid/dice'] = tf.keras.metrics.Mean()
#     metrics['valid/iou'] = tf.keras.metrics.Mean()


def main(argv):
    if FLAGS.multi_class:
        loss_fn = tversky_loss
        crossentropy_loss_fn = tf.keras.losses.categorical_crossentropy
    else:
        loss_fn = dice_coef_loss
        crossentropy_loss_fn = tf.keras.losses.binary_crossentropy

    if FLAGS.use_bfloat16:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    num_classes = 7 if FLAGS.multi_class else 1
    model_fn, model_args = select_model(FLAGS, num_classes)

    model = model_fn(*model_args)

    steps_per_epoch = DATASET_SIZE // 2
    validation_steps = DATASET_SIZE // 2

    lr_decay_epochs = list(range(FLAGS.lr_warmup_epochs + 1, FLAGS.train_epochs))
    lr_rate = LearningRateSchedule(steps_per_epoch,
                                   FLAGS.base_learning_rate,
                                   FLAGS.min_learning_rate,
                                   FLAGS.lr_drop_ratio,
                                   lr_decay_epochs,
                                   FLAGS.lr_warmup_epochs)

    if FLAGS.optimizer == 'adam':
        optimiser = tf.keras.optimizers.Adam(learning_rate=lr_rate)
    elif FLAGS.optimizer == 'rms-prop':
        optimiser = tf.keras.optimizers.RMSprop(learning_rate=lr_rate)
    elif FLAGS.optimizer == 'sgd':
        optimiser = tf.keras.optimizers.SGD(learning_rate=lr_rate)
    else:
        print('Not a valid input optimizer, using Adam.')
        optimiser = tf.keras.optimizers.Adam(learning_rate=lr_rate)

    # for some reason, if i build the model then it can't load checkpoints. I'll see what I can do about this
    if FLAGS.train:
        if FLAGS.model_architecture != 'vnet':
            if FLAGS.backbone_architecture == 'default':
                model.build((None, FLAGS.crop_size, FLAGS.crop_size, 1))
            else:
                model.build((None, FLAGS.crop_size, FLAGS.crop_size, 3))
        else:
            model.build((None, FLAGS.depth_crop_size, FLAGS.crop_size, FLAGS.crop_size, 1))
        model.summary()

    metrics = [dice_coef, mIoU, crossentropy_loss_fn, 'acc']

    model.compile(optimizer=optimiser,
                    loss=loss_fn,
                    metrics=metrics)

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(FLAGS.logdir, FLAGS.tpu)
    logdir = os.path.join(logdir, time)
    logdir_arch = os.path.join(logdir, FLAGS.model_architecture)

############# MODEL SELECT #############



########################## TEST ##########################
    print('logdir', logdir)
    file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=get_confusion_matrix_cb)

    history = model.fit(train_ds,
                        steps_per_epoch=steps_per_epoch,
                        epochs=FLAGS.train_epochs,
                        validation_data=validation_ds,
                        validation_steps=validation_steps,
                        callbacks=[cm_callback])


if __name__ == '__main__':
    app.run(main)