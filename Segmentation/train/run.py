import tensorflow as tf
from time import time
import os
from Segmentation.train.build import build_model
from Segmentation.train.train import load_datasets, Train
from Segmentation.train.utils import LearningRateUpdate, Metric
from Segmentation.train.validation import validate_best_model
from Segmentation.utils.losses import dice_loss, tversky_loss, iou_loss
from Segmentation.utils.losses import iou_loss_eval_3d, dice_coef_eval_3d
from Segmentation.utils.losses import dice_loss_weighted_3d, focal_tversky


def main(epochs,
         name,
         log_dir_now=None,
         batch_size=2,
         val_batch_size=2,
         lr=1e-4,
         lr_drop=0.9,
         lr_drop_freq=5,
         lr_warmup=3,
         num_to_visualise=2,
         num_channels=4,
         buffer_size=4,
         enable_function=True,
         tfrec_dir='./Data/tfrecords/',
         multi_class=False,
         crop_size=144,
         depth_crop_size=80,
         aug=[],
         debug=False,
         predict_slice=False,
         tpu_name=None,
         min_lr=1e-7,
         custom_loss=None,
         **model_kwargs,
         ):
    t0 = time()

    if tpu_name:
        tfrec_dir = 'gs://oai-challenge-dataset/tfrecords'

    num_classes = 7 if multi_class else 1

    metrics = {
        'losses': {
            'mIoU': [iou_loss, tf.keras.metrics.Mean(), tf.keras.metrics.Mean(), None, None],
            'dice': [dice_loss, tf.keras.metrics.Mean(), tf.keras.metrics.Mean(), None, None]
        },
    }

    if multi_class:
        metrics['losses']['mIoU-6ch'] = [iou_loss_eval_3d, tf.keras.metrics.Mean(), tf.keras.metrics.Mean(), None, None]
        metrics['losses']['dice-6ch'] = [dice_coef_eval_3d, tf.keras.metrics.Mean(), tf.keras.metrics.Mean(), None, None]

    train_ds, valid_ds = load_datasets(batch_size, buffer_size, tfrec_dir, multi_class,
                                       crop_size=crop_size, depth_crop_size=depth_crop_size, aug=aug,
                                       predict_slice=predict_slice)

    # num_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
    # steps_per_epoch = len(glob(os.path.join(tfrec_dir, 'train_3d/*'))) / (batch_size)

    if tpu_name:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        if custom_loss is None:
            loss_func = tversky_loss if multi_class else dice_loss
        elif multi_class and custom_loss == "weighted":
            loss_func = dice_loss_weighted_3d
        elif multi_class and custom_loss == "focal":
            loss_func = focal_tversky
        else:
            raise NotImplementedError(f"Custom loss: {custom_loss} not implemented.")

        lr_manager = LearningRateUpdate(lr, lr_drop, lr_drop_freq, warmup=lr_warmup, min_lr=min_lr)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model = build_model(num_channels, num_classes, name, predict_slice=predict_slice, **model_kwargs)

        trainer = Train(epochs, batch_size, enable_function,
                        model, optimizer, loss_func, lr_manager, predict_slice, metrics,
                        tfrec_dir=tfrec_dir)

        train_ds = strategy.experimental_distribute_dataset(train_ds)
        valid_ds = strategy.experimental_distribute_dataset(valid_ds)

        if log_dir_now is None:
            log_dir_now = trainer.train_model_loop(train_ds, valid_ds, strategy, multi_class, debug, num_to_visualise)

    train_time = time() - t0
    print(f"Train Time: {train_time:.02f}")
    t1 = time()
    with strategy.scope():
        model = build_model(num_channels, num_classes, name, predict_slice=predict_slice, **model_kwargs)
        model.load_weights(os.path.join(log_dir_now + '/best_weights.tf')).expect_partial()
    print("Validation for:", log_dir_now)

    if not predict_slice:
        total_loss, metric_str = validate_best_model(model,
                                                     log_dir_now,
                                                     val_batch_size,
                                                     buffer_size,
                                                     tfrec_dir,
                                                     multi_class,
                                                     crop_size,
                                                     depth_crop_size,
                                                     predict_slice,
                                                     Metric(metrics))
        print(f"Train Time: {train_time:.02f}")
        print(f"Validation Time: {time() - t1:.02f}")
        print(f"Total Time: {time() - t0:.02f}")
        with open("results/3d_result.txt", "a") as f:
            f.write(f'{log_dir_now}: total_loss {total_loss} {metric_str} \n')


if __name__ == "__main__":
    main(3, 'test')
