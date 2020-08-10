

from Segmentation.utils.data_loader import load_dataset


def main():

    # # --------------------------------------------------------------------------------
    # # --------------------------------------------------------------------------------
    
    # set model architecture
    model_fn, model_args = select_model(FLAGS, num_classes)

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


    # set up accelerator and returns strategy used
    strategy = setup_accelerator(use_gpu=False if tpu_name is None else True,
                                 num_cores=num_cores,
                                 device_name=tpu_name)

    with strategy.scope():

        evaluator = Evaluator(validation_steps,
                              run_eager,
                              target_weights_dir,
                              vis_out_dir,
                              vis_args,
                              vis_weight_name='') 

        # load dataset
        train_ds, valid_ds = load_dataset(batch_size=batch_size,
                                          dataset_dir=tfrec_dir,
                                          augmentation=aug,
                                          use_2d=use_2d,
                                          multi_class=multi_class,
                                          crop_size=crop_size,
                                          buffer_size=buffer_size,
                                          use_bfloat16=use_bfloat16,
                                          use_RGB=use_RGB
                                          )

        train_ds = strategy.experimental_distribute_dataset(train_ds)
        valid_ds = strategy.experimental_distribute_dataset(valid_ds)

        if log_dir_now is None:
            log_dir_now = evaluator.eval_model_loop(train_ds, valid_ds, strategy, multi_class, debug, num_to_visualise)


    # # --------------------------------------------------------------------------------
    # # --------------------------------------------------------------------------------

