import tensorflow as tf

import os
from pathlib import Path
from datetime import datetime
from absl import app
from absl import logging

from Segmentation.utils.accelerator import setup_accelerator
from Segmentation.utils.data_loader import load_dataset
from Segmentation.utils.training_utils import LearningRateSchedule
from Segmentation.utils.losses import dice_coef_loss, tversky_loss
from Segmentation.utils.metrics import dice_coef, DiceMetrics, dice_coef_eval

from flags import FLAGS
from select_model import select_model


def main(argv):

    if FLAGS.visual_file:
        assert FLAGS.train is False, "Train must be set to False if you are doing a visual."
    del argv  # unused arg

    tf.random.set_seed(FLAGS.seed)  # set seed

    # set whether to train on GPU or TPU
    strategy = setup_accelerator(use_gpu=FLAGS.use_gpu, num_cores=FLAGS.num_cores, device_name=FLAGS.tpu)

    batch_size = (FLAGS.batch_size * FLAGS.num_cores)

    # set dataset configuration
    train_ds, validation_ds = load_dataset(batch_size=batch_size,
                                           dataset_dir=FLAGS.tfrec_dir,
                                           augmentation=FLAGS.aug_strategy,
                                           use_2d=FLAGS.use_2d,
                                           multi_class=FLAGS.multi_class,
                                           crop_size=FLAGS.crop_size,
                                           buffer_size=FLAGS.buffer_size,
                                           use_bfloat16=FLAGS.use_bfloat16,
                                           use_RGB=False if FLAGS.backbone_architecture == 'default' else True
                                           )

    num_classes = 7 if FLAGS.multi_class else 1
    steps_per_epoch = 19200 // batch_size
    validation_steps = 4480 // batch_size
    # # --------------------------------------------------------------------------------
    # # def set_metrics()
    if FLAGS.multi_class:
        loss_fn = tversky_loss
        crossentropy_loss_fn = tf.keras.losses.categorical_crossentropy
    else:
        loss_fn = dice_coef_loss
        crossentropy_loss_fn = tf.keras.losses.binary_crossentropy

    if FLAGS.use_bfloat16:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
    # # --------------------------------------------------------------------------------

    # set model architecture
    model_fn, model_args = select_model(FLAGS, num_classes)

    # # --------------------------------------------------------------------------------
    # # if FLAGS.train:
    # #     def train()
    with strategy.scope():
        model = model_fn(*model_args)

        if FLAGS.custom_decay_lr:
            lr_decay_epochs = FLAGS.lr_decay_epochs
        else:
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

        if FLAGS.multi_class:
            dice_metrics = [DiceMetrics(idx=idx) for idx in range(num_classes)]            
            metrics = [dice_metrics, dice_coef_eval]
        else:
            metrics = [dice_coef]

        model.compile(optimizer=optimiser,
                      loss=loss_fn,
                      metrics=metrics)

    # FLAGS.train will be outside train()
    if FLAGS.train:
        # define checkpoints
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        training_history_dir = os.path.join(FLAGS.fig_dir, FLAGS.tpu)
        training_history_dir = os.path.join(training_history_dir, time)
        Path(training_history_dir).mkdir(parents=True, exist_ok=True)
        flag_name = os.path.join(training_history_dir, 'test_flags.cfg')
        FLAGS.append_flags_into_file(flag_name)

        logdir = os.path.join(FLAGS.logdir, FLAGS.tpu)
        logdir = os.path.join(logdir, time)
        logdir_arch = os.path.join(logdir, FLAGS.model_architecture)
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(logdir_arch + '_weights.{epoch:03d}.ckpt',
                                                     save_best_only=False,
                                                     save_weights_only=True)
        tb = tf.keras.callbacks.TensorBoard(logdir, update_freq='epoch')
        # file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
        # cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=get_confusion_matrix_cb)

        history = model.fit(train_ds,
                            steps_per_epoch=steps_per_epoch,
                            epochs=FLAGS.train_epochs,
                            validation_data=validation_ds,
                            validation_steps=validation_steps,
                            callbacks=[ckpt_cb, tb],
                            verbose=1)

    else:
        model.load_weights(FLAGS.weights_dir).expect_partial()
        model.evaluate(validation_ds,
                       steps=validation_steps
                       )

if __name__ == '__main__':
    app.run(main)
