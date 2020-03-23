import numpy as np
import math
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from glob import glob
import imageio
import subprocess


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self,
                 steps_per_epoch,
                 initial_learning_rate,
                 drop,
                 epochs_drop,
                 min_lr=1e-7):
        super(LearningRateSchedule, self).__init__()
        self.steps_per_epoch = steps_per_epoch
        self.initial_learning_rate = initial_learning_rate
        self.drop = drop
        self.epochs_drop = epochs_drop
        self._step = 0
        self.min_lr = min_lr

    def __call__(self, step):
        self._step += 1
        lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
        lrate = self.initial_learning_rate * tf.math.pow(self.drop, tf.math.floor((1+lr_epoch)/self.epochs_drop))
        if lrate < self.min_lr:
            lrate = self.min_lr
        return lrate

    def get_config(self):
        lr_epoch = tf.cast(self._step, tf.float32) / self.steps_per_epoch
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'step': self._step,
            'current_learning_rate': (self.initial_learning_rate * tf.math.pow(self.drop, tf.math.floor((1+lr_epoch)/self.epochs_drop))).numpy()
        }

def get_git_file_short_hash(file_path):
    commit_id = None
    try:
        commit_id = subprocess.check_output(['git', 'log', '-1', '--pretty=format:%h', '--', file_path]).decode("utf-8") 
    except Exception as e:
        print(f"Get commit hash for file {file_path} failed: {e}")
        commit_id = "Not Found"
    return commit_id

def train(model, n_classes=1, batch_size=1, sample_shape=(128, 128, 128), epochs=10,
          save_model=True, validate=True, train_name="", custom_train_loop=True, train_debug=False,
          reduce_lr=False, start_lr=5e-4, dataset_load_method=None,
          shuffle_order=True, normalise_input=True, remove_outliers=True,
          transform_angle=False, transform_position="normal",
          skip_empty=True, examples_per_load=1, use_optimizer="adam",
          mean_loss_of_batch="", schedule_drop="", schedule_epochs_drop="", notes="",
          **model_kwargs):
    start_time = time.perf_counter()
    debug = ""
    if train_debug:
        debug = "debug_"
    get_position, get_slice = False, False
    commit_id = ""
    if model == "tiny":
        from Segmentation.model.vnet_tiny import VNet_Tiny
        vnet = VNet_Tiny(1, n_classes, **model_kwargs)
        model_path = "Segmentation/model/vnet_tiny.py"
        commit_id = get_git_file_short_hash(model_path)
    elif model == "small":
        from Segmentation.model.vnet_small import VNet_Small
        vnet = VNet_Small(1, n_classes, **model_kwargs)
        model_path = "Segmentation/model/vnet_small.py"
        commit_id = get_git_file_short_hash(model_path)
    elif model == "small_relative":
        from Segmentation.model.vnet_small_relative import VNet_Small_Relative
        vnet = VNet_Small_Relative(1, n_classes, **model_kwargs)
        get_position = True
        model_path = "Segmentation/model/vnet_small_relative.py"
        commit_id = get_git_file_short_hash(model_path)
    elif model == "slice":
        from Segmentation.model.vnet_slice import VNet_Slice
        vnet = VNet_Slice(1, n_classes, **model_kwargs)
        get_slice = True
        model_path = "Segmentation/model/vnet_slice.py"
        commit_id = get_git_file_short_hash(model_path)
    elif model == "large":
        from Segmentation.model.vnet_large import VNet_Large
        vnet = VNet_Large(1, n_classes, **model_kwargs)
        model_path = "Segmentation/model/vnet_large.py"
        commit_id = get_git_file_short_hash(model_path)
    elif model == "large_relative":
        from Segmentation.model.vnet_large_relative import VNet_Large_Relative
        vnet = VNet_Large_Relative(1, n_classes, **model_kwargs)
        get_position = True
        model_path = "Segmentation/model/vnet_large_relative.py"
        commit_id = get_git_file_short_hash(model_path)
    else:
        raise NotImplementedError()

    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')

    if not os.path.exists('ckpt/checkpoints'):
        os.makedirs('ckpt/checkpoints')

    now = datetime.now()
    now_time = now.strftime('%Y_%m_%d-%H_%M_%S')

    if not os.path.exists(f'ckpt/checkpoints/{debug}train_session_{now_time}_{model}'):
        os.makedirs(f'ckpt/checkpoints/{debug}train_session_{now_time}_{model}')

    if custom_train_loop:
        if not os.path.exists(f'ckpt/checkpoints/{debug}train_session_{now_time}_{model}/progress'):
            os.makedirs(f'ckpt/checkpoints/{debug}train_session_{now_time}_{model}/progress')

    setup_gpu()

    steps = None
    vsteps = None
    if dataset_load_method == "tf":
        get_slice = False
        if model == "slice":
            get_slice = True

        from Segmentation.utils.data_loader_3d_tf import get_dataset
        tdataset, steps = get_dataset(file_path="t",
                                      sample_shape=sample_shape,
                                      remove_outliers=remove_outliers,
                                      transform_angle=False,
                                      transform_position=transform_position,
                                      get_slice=get_slice,
                                      get_position=get_position,
                                      skip_empty=skip_empty)
        vdataset, vsteps = get_dataset(file_path="v",
                                       sample_shape=sample_shape,
                                       remove_outliers=remove_outliers,
                                       transform_angle=False,
                                       transform_position=False,
                                       get_slice=get_slice,
                                       get_position=get_position,
                                       skip_empty=skip_empty)
        tdataset = tdataset.repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        vdataset = vdataset.repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    elif dataset_load_method is None:
        get_slice = False
        if model == "slice":
            get_slice = True

        from Segmentation.utils.data_loader_3d import VolumeGenerator
        tdataset = VolumeGenerator(batch_size, sample_shape, "t",
                                   shuffle_order=shuffle_order,
                                   normalise_input=normalise_input,
                                   remove_outliers=remove_outliers,
                                   transform_angle=False,
                                   transform_position=transform_position,
                                   get_slice=get_slice,
                                   get_position=get_position,
                                   skip_empty=skip_empty,
                                   examples_per_load=examples_per_load,
                                   train_debug=train_debug)
        vdataset = VolumeGenerator(batch_size, sample_shape, "v",
                                   shuffle_order=shuffle_order,
                                   normalise_input=normalise_input,
                                   remove_outliers=remove_outliers,
                                   transform_angle=False,
                                   transform_position=False,
                                   get_slice=get_slice,
                                   get_position=get_position,
                                   skip_empty=skip_empty,
                                   examples_per_load=examples_per_load,
                                   train_debug=train_debug)

    from Segmentation.utils.losses import bce_dice_loss, dice_loss
    from Segmentation.utils.losses import tversky_loss, precision, recall

    loss_name = ""
    if n_classes == 1:
        loss_func = dice_loss
        loss_name = 'dice loss'
        from tensorflow.keras.losses import binary_crossentropy
        met_loss = binary_crossentropy
        met_loss_name = 'binary_crossentropy'
    else:
        loss_func = tversky_loss
        loss_name = "tversky loss"
        from tensorflow.keras.losses import categorical_crossentropy
        met_loss = categorical_crossentropy
        met_loss_name = 'categorical_crossentropy'

    loss_hist = []
    loss_val_hist = []

    metrics = [met_loss, precision, recall]

    metric_hist = [None] * len(metrics)
    metric_val_hist = [None] * len(metrics)
    for i in range(2):  # we only care about 2 metrics
        metric_hist[i] = []
        metric_val_hist[i] = []

    save_models_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'ckpt/checkpoints/{debug}train_session_{now_time}_{model}/chkp/model_' + '{epoch}',
        verbose=1)

    if custom_train_loop:
        if use_optimizer == "adam":
            optimizer = Adam(learning_rate=start_lr)
        elif use_optimizer == "adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=1e-6, rho=0.95)
        elif use_optimizer == "adam_schedule":
            assert dataset_load_method is None, "Need to be using data generator loader"
            assert schedule_drop != "", "Schedule drop needs to be set when using schedule"
            assert schedule_epochs_drop != "", "Schedule epochs drop needs to be set when using schedule"
            steps_per_epoch = int(len(VolumeGenerator.get_paths("t")) / batch_size)
            lr_schedule = LearningRateSchedule(steps_per_epoch=steps_per_epoch,
                                               initial_learning_rate=start_lr,
                                               drop=schedule_drop,
                                               epochs_drop=schedule_epochs_drop)
            optimizer = Adam(learning_rate=lr_schedule)
        if mean_loss_of_batch == "":
            mean_loss_of_batch = False  # assign default value of false
        for epoch in range(epochs):
            epoch_time = time.perf_counter()
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_val_loss_avg = tf.keras.metrics.Mean()
            epoch_met_loss_avg = tf.keras.metrics.Mean()
            epoch_val_met_loss_avg = tf.keras.metrics.Mean()
            epoch_bcedice_loss_avg = tf.keras.metrics.Mean()
            epoch_val_bcedice_loss_avg = tf.keras.metrics.Mean()

            for x, y in tdataset:
                with tf.GradientTape() as tape:
                    y_ = vnet(x, training=True)
                    if mean_loss_of_batch:
                        loss_value = tf.reduce_mean(loss_func(y_true=y, y_pred=y_))
                    else:
                        loss_value = loss_func(y_true=y, y_pred=y_)
                grads = tape.gradient(loss_value, vnet.trainable_variables)

                optimizer.apply_gradients(zip(grads, vnet.trainable_variables))
                epoch_loss_avg(loss_value)

                met_loss_value = met_loss(y_true=y, y_pred=y_)
                epoch_met_loss_avg(met_loss_value)
                epoch_bcedice_loss_avg(bce_dice_loss(y_true=y, y_pred=y_))

            clr = ""
            if use_optimizer == "adam_schedule":
                clr = f", lr: {optimizer.get_config()['learning_rate']['config']['current_learning_rate']: .07f}"

            y_ = vnet(x, training=False)
            if get_position:
                x = x[0]
            slice_idx = int(x.shape[3] / 2)
            store_x = x[0, :, :, slice_idx, 0]
            if get_slice:
                store_y = y[0, :, :, 0]
                store_y_pred = y_[0, :, :, 0]
            else:
                store_y = y[0, :, :, slice_idx, 0]
                store_y_pred = y_[0, :, :, slice_idx, 0]
            vidx = 0
            for x, y in vdataset:
                y_ = vnet(x, training=False)
                val_loss_value = loss_func(y_true=y, y_pred=y_)
                epoch_val_loss_avg(val_loss_value)

                val_met_loss_value = met_loss(y_true=y, y_pred=y_)
                epoch_val_met_loss_avg(val_met_loss_value)

                epoch_val_bcedice_loss_avg(bce_dice_loss(y_true=y, y_pred=y_))

                if vidx == 0:
                    if get_position:
                        x = x[0]
                    store_x_val_0 = x[0, :, :, slice_idx, 0]
                    if get_slice:
                        store_y_val_0 = y[0, :, :, 0]
                        store_y_pred_val_0 = y_[0, :, :, 0]
                    else:
                        store_y_val_0 = y[0, :, :, slice_idx, 0]
                        store_y_pred_val_0 = y_[0, :, :, slice_idx, 0]
                    vidx += 1
            if get_position:
                x = x[0]
            store_x_val = x[0, :, :, slice_idx, 0]
            if get_slice:
                store_y_val = y[0, :, :, 0]
                store_y_pred_val = y_[0, :, :, 0]
            else:
                store_y_val = y[0, :, :, slice_idx, 0]
                store_y_pred_val = y_[0, :, :, slice_idx, 0]

            eloss_str = f" epoch: {epoch:3d}, loss: {epoch_loss_avg.result(): .5f},"
            evalloss_str = f" loss val: {epoch_val_loss_avg.result(): .5f},"
            emetloss_str = f" bce: {epoch_met_loss_avg.result(): .5f},"
            evalmetloss_str = f" bce val: {epoch_val_met_loss_avg.result(): .5f}"
            emetbcediceloss_str = f" bce dice: {epoch_bcedice_loss_avg.result(): .5f},"
            evalbcediceloss_str = f" bce dice val: {epoch_val_bcedice_loss_avg.result(): .5f}"

            print(f"{time.perf_counter() - epoch_time:3.0f} s" + eloss_str + evalloss_str + emetloss_str + evalmetloss_str + emetbcediceloss_str + evalbcediceloss_str)

            loss_hist.append(epoch_loss_avg.result())
            loss_val_hist.append(epoch_val_loss_avg.result())

            metric_hist[0].append(epoch_met_loss_avg.result())
            metric_val_hist[0].append(epoch_val_met_loss_avg.result())
            metric_hist[1].append(epoch_bcedice_loss_avg.result())
            metric_val_hist[1].append(epoch_val_bcedice_loss_avg.result())

            f, axes = plt.subplots(3, 3)

            axes[0, 0].imshow(store_x, cmap="gray")
            axes[0, 0].set_title("train raw image")

            axes[0, 1].imshow(store_y, cmap="gray")
            axes[0, 1].set_title("train y")

            axes[0, 2].imshow(store_y_pred, cmap="gray")
            axes[0, 2].set_title("train y pred")

            axes[1, 0].imshow(store_x_val, cmap="gray")
            axes[1, 0].set_title("val raw image")

            axes[1, 1].imshow(store_y_val, cmap="gray")
            axes[1, 1].set_title("val y")

            axes[1, 2].imshow(store_y_pred_val, cmap="gray")
            axes[1, 2].set_title("val y pred")

            axes[2, 0].imshow(store_x_val_0, cmap="gray")
            axes[2, 0].set_title("val raw image")

            axes[2, 1].imshow(store_y_val_0, cmap="gray")
            axes[2, 1].set_title("val y")

            axes[2, 2].imshow(store_y_pred_val_0, cmap="gray")
            axes[2, 2].set_title("val y pred")

            for a in axes:
                for ax in a:
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)

            f.tight_layout(rect=[0, 0.01, 1, 0.93])
            f.suptitle(f"{model}: {train_name}\nepoch: {epoch}, loss: {epoch_loss_avg.result(): .5f}, val loss: {epoch_val_loss_avg.result(): .5f}{clr}")
            plt.savefig(f"ckpt/checkpoints/{debug}train_session_{now_time}_{model}/progress/train_{epoch:04d}_{now_time}")
            plt.close('all')
            # print(f"plot saved: {time.perf_counter() - epoch_time:.0f}") plots are very fast to save, take ~1 second
            vnet.save_weights(f"ckpt/checkpoints/{debug}train_session_{now_time}_{model}/chkp/ckpt_{epoch:04d}_{now_time}.cktp")
    else:
        vnet.compile(optimizer=Adam(lr=start_lr),
                     loss=loss_func,
                     metrics=metrics,
                     experimental_run_tf_function=True)
        callbacks = [save_models_callback]
        if reduce_lr:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                     patience=10, min_lr=1e-7, verbose=1)
            )
        if validate:
            h = vnet.fit(x=tdataset, validation_data=vdataset,
                         callbacks=callbacks, epochs=epochs, verbose=1,
                         steps_per_epoch=steps, validation_steps=vsteps)
            loss_val_hist = h.history['val_loss']
            metric_val_hist[0] = h.history[f'val_{met_loss_name}']
            metric_val_hist[1] = h.history[f'val_dice_loss']
        else:
            h = vnet.fit(x=tdataset, callbacks=callbacks, epochs=epochs, verbose=1)
        loss_hist = h.history['loss']
        metric_hist[0] = h.history[f'{met_loss_name}']
        metric_hist[1] = h.history[f'dice_loss']

    time_taken = time.perf_counter() - start_time
    roll_period = 5
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(loss_hist, label="loss")
    ax1.plot(running_mean(loss_hist, roll_period), label="loss roll")
    if validate:
        ax1.plot(loss_val_hist, label="val loss")
        ax1.plot(running_mean(loss_val_hist, roll_period), label="val loss roll")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel(loss_name)
    ax1.set_title("loss: dice")
    ax1.legend()
    ax2.plot(metric_hist[0], label=met_loss_name)
    ax2.plot(running_mean(metric_hist[0], roll_period), label=f"{met_loss_name} roll")
    if validate:
        ax2.plot(metric_val_hist[0], label=f"val {met_loss_name}")
        ax2.plot(running_mean(metric_val_hist[0], roll_period), label=f"val {met_loss_name} roll")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("cross entropy")
    ax2.legend()
    ax3.plot(metric_hist[1], label="dice_loss")
    ax3.plot(running_mean(metric_hist[1], roll_period), label=f"dice_loss roll")
    if validate:
        ax3.plot(metric_val_hist[1], label=f"val dice_loss")
        ax3.plot(running_mean(metric_val_hist[1], roll_period), label=f"val dice_loss roll")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("bce + dice loss")
    ax3.legend()
    f.tight_layout(rect=[0, 0.01, 1, 0.93])
    min_loss = min(loss_hist).numpy()
    min_val_loss = min(loss_val_hist).numpy()
    f.suptitle(f"{model}: {train_name}\nTime: {time_taken/60:.1f} mins, Min Loss: {min_loss:.5f}, Min Val Loss: {min_val_loss:.5f}")
    plt.savefig(f"ckpt/checkpoints/{debug}train_session_{now_time}_{model}/train_result_{now_time}")

    if custom_train_loop:
        filenames = glob(f"ckpt/checkpoints/{debug}train_session_{now_time}_{model}/progress/*")
        filenames.sort()
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'ckpt/checkpoints/{debug}train_session_{now_time}_{model}/progress.gif', images)
    print(f"time taken: {time_taken:.1f}")

    train_cols = {
        'Model': [model], 'Input Shape': [sample_shape], 'Learning Rate': [start_lr],
        'Loss': [loss_name], 'Optimizer': [use_optimizer],
        'Num. Epochs': [epochs], 'Examples/Epoch': [examples_per_load],
        'Min Loss': [min_loss], 'Min Val Loss': [min_val_loss], 'Fit': [custom_train_loop],
        'Mean loss of batch': [mean_loss_of_batch],
        'Schedule Drop': [schedule_drop], 'Schedule Epochs Drop': [schedule_epochs_drop], 'Train Duration': [time_taken],
        'Shuffle Order': [shuffle_order], 'Normalise Input': [normalise_input], 'Remove Outliers': [remove_outliers], 'Skip Empty': [skip_empty],
        'transform_angle': [transform_angle], 'transform_position': [transform_position],
        'Commit ID': [commit_id], 'Train Name': [f"{debug}train_session_{now_time}_{model}"], 'Model Params': [vnet.params], 'Notes': [notes],
    }
    df = pd.DataFrame(data=train_cols)
    df.to_csv("vnet_train_experiments.csv", index=False, header=False, mode='a')

    return time_taken


if __name__ == "__main__":

    print("running eagerly:", tf.executing_eagerly())

    import sys
    sys.path.insert(0, os.getcwd())

    if not os.path.exists("vnet_train_experiments.csv"):
        train_cols = {
            'Model': [], 'Input Shape': [], 'Learning Rate': [],
            'Loss': [],'Optimizer': [],
            'Num. Epochs': [], 'Examples/Epoch': [],
            'Min Loss': [], 'Min Val Loss': [], 'Fit': [],
            'Mean loss of batch': [],
            'Schedule Drop': [], 'Schedule Epochs Drop': [], 'Train Duration': [],
            'Shuffle Order': [], 'Normalise Input': [], 'Remove Outliers': [], 'Skip Empty': [],
            'transform_angle': [], 'transform_position': [],
            'Commit ID': [], 'Train Name': [], 'Model Params': [], 'Notes': [],
        }
        df = pd.DataFrame(data=train_cols)
        df.to_csv("vnet_train_experiments.csv", index=False)

    debug = False
    if not debug:

        e = 150
        learn_rate = 5e-4

        # sample_shape = (64, 64, 64)
        # train("small", batch_size=4, sample_shape=sample_shape, epochs=e, examples_per_load=1,
        #       train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
        #       use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.95,
        #       use_stride_2=True, use_res_connect=True,
        #       notes="Training Small VNet for smaller size of (64,64,64)")

        sample_shape = (160, 160, 160)
        # train("small", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1,
        #       train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
        #       use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.95,
        #       use_stride_2=False, use_res_connect=True,
        #       notes="Training Small VNet for smaller size of (160,160,160), use_stride_2=False, use_res_connect=True")
        
        
        sample_shape = (160, 160, 160)
        train("small", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1,
              train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
              use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.95,
              use_stride_2=True, use_res_connect=True,
              notes="Training Small VNet for smaller size of (160,160,160), Repeat #4")
        train("small", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1,
              train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
              use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.95,
              use_stride_2=True, use_res_connect=True,
              notes="Training Small VNet for smaller size of (160,160,160), Repeat #5")

        # train("small", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1,
        #       train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
        #       use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=4, schedule_drop=0.9,
        #       use_stride_2=True, use_res_connect=True, kernel_size=(5, 5, 5),
        #       notes="Training Small VNet for smaller size of (160,160,160), kernel size of (5,5,5)")
        # train("small", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1,
        #       train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
        #       use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=4, schedule_drop=0.9,
        #       use_stride_2=True, use_res_connect=True, kernel_size=(5, 5, 5),
        #       notes="Training Small VNet for smaller size of (160,160,160), kernel size of (5,5,5), Repeat #2")

        # sample_shape = (180, 180, 160)
        # train("small", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1,
        #       train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
        #       use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=4, schedule_drop=0.9,
        #       use_stride_2=True, use_res_connect=True, kernel_size=(5, 5, 5),
        #       notes="Training Small VNet for smaller size of (180,180,160), kernel size of (5,5,5)")
        
        # sample_shape = (240, 240, 160)
        # train("small", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1,
        #       train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
        #       use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.95,
        #       use_stride_2=True, use_res_connect=True,
        #       notes="Training Small VNet for size of (240, 240, 160)")
        # train("small", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1,
        #       train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
        #       use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.95,
        #       use_stride_2=True, use_res_connect=True, kernel_size=(5, 5, 5),
        #       notes="Training Small VNet for size of (240, 240, 160) and kernel size of (5,5,5)")

        # sample_shape = (288, 288, 3)
        # train("slice", batch_size=3, sample_shape=sample_shape, epochs=e, examples_per_load=1,
        #       train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, k=(3,3,1)", custom_train_loop=True,
        #       use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=10, schedule_drop=0.99,
        #       kernel_size=(3, 3, 1),
        #       notes="Baselining slice with kernel (3,3,1) vs (3,3,3), expect to underfit")
        # train("slice", batch_size=3, sample_shape=sample_shape, epochs=e, examples_per_load=1,
        #       train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, k=(3,3,3)", custom_train_loop=True,
        #       use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=10, schedule_drop=0.99,
        #       kernel_size=(3, 3, 3),
        #       notes="Baselining slice with kernel (3,3,1) vs (3,3,3), expect to underfit")

        # learn_rate = 5e-4
        # sample_shape = (160, 160, 160)
        # train("tiny", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1,
        #       train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice", custom_train_loop=True,
        #       use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=10, schedule_drop=0.99,
        #       notes="Comparing tiny vs small vs large")
        # train("small", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1,
        #       train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice", custom_train_loop=True,
        #       use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=10, schedule_drop=0.99,
        #       notes="Comparing tiny vs small vs large")
        # train("large", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1,
        #       train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice", custom_train_loop=True,
        #       use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=10, schedule_drop=0.99,
        #       notes="Comparing tiny vs small vs large")
    else:
        e = 3
        train("tiny", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True)
        train("small", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True)
        train("small", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True,
              use_stride_2=True, use_res_connect=True)
        train("large", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True)
        train("small_relative", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True)
        train("large_relative", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True)
        train("slice", batch_size=1, sample_shape=(160, 160, 5), epochs=e, examples_per_load=1,
              train_name="debug (160,160,5), Adam, dice", custom_train_loop=True, train_debug=True)
