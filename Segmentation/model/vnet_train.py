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
        lrate = self.initial_learning_rate * tf.math.pow(self.drop, tf.math.floor((1 + lr_epoch) / self.epochs_drop))
        if lrate < self.min_lr:
            lrate = self.min_lr
        return lrate

    def get_config(self):
        lr_epoch = tf.cast(self._step, tf.float32) / self.steps_per_epoch
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'step': self._step,
            'current_learning_rate': (self.initial_learning_rate * tf.math.pow(self.drop, tf.math.floor((1 + lr_epoch) / self.epochs_drop))).numpy()
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
        cross_entropy = binary_crossentropy
        cross_entropy_name = 'binary_crossentropy'
    else:
        loss_func = tversky_loss
        loss_name = "tversky loss"
        from tensorflow.keras.losses import categorical_crossentropy
        cross_entropy = categorical_crossentropy
        cross_entropy_name = 'categorical_crossentropy'

    metrics = {}
    metrics['loss'] = {'loss func': loss_func, 'history': [], 'val history': []}
    metrics[cross_entropy_name] = {'loss func': cross_entropy, 'history': [], 'val history': []}
    metrics['bce_dice_loss'] = {'loss func': bce_dice_loss, 'history': [], 'val history': []}
    metrics['precision'] = {'loss func': precision, 'history': [], 'val history': []}
    metrics['recall'] = {'loss func': recall, 'history': [], 'val history': []}

    save_models_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'ckpt/checkpoints/{debug}train_session_{now_time}_{model}/chkp/model_' + '{epoch}',
        verbose=1
    )

    if custom_train_loop:
        if use_optimizer == "adam":
            optimizer = Adam(learning_rate=start_lr)
        elif use_optimizer == "adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=1e-4, rho=0.95)
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
            mean_loss_of_batch = False
        for epoch in range(epochs):
            epoch_time = time.perf_counter()

            metrics_epoch_avg = {}
            for metric in metrics:
                metrics_epoch_avg[metric] = {'loss': tf.keras.metrics.Mean(), 'val_loss': tf.keras.metrics.Mean()}
            for x, y in tdataset:
                with tf.GradientTape() as tape:
                    y_ = vnet(x, training=True)
                    loss_value = loss_func(y_true=y, y_pred=y_)
                    if mean_loss_of_batch:
                        loss_value = tf.reduce_mean(loss_value)
                grads = tape.gradient(loss_value, vnet.trainable_variables)
                optimizer.apply_gradients(zip(grads, vnet.trainable_variables))

                for idx, metric in enumerate(metrics):
                    if idx == 0:  # perform different action for loss_func since it has already been calculated
                        metrics_epoch_avg[metric]['loss'](loss_value)
                        continue
                    value = metrics[metric]['loss func'](y_true=y, y_pred=y_)  # calculate the training metrics value
                    metrics_epoch_avg[metric]['loss'](value)             # store the value in keras.metrics.Mean

            clr = ""
            if use_optimizer == "adam_schedule":
                clr = f", lr: {optimizer.get_config()['learning_rate']['config']['current_learning_rate']: .07f}"

            # should be in seperate function
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

                for metric in metrics:
                    value = metrics[metric]['loss func'](y_true=y, y_pred=y_)  # calculate the training metrics value
                    metrics_epoch_avg[metric]['val_loss'](value)               # store the value in keras.metrics.Mean

                if vidx == 0:  # should be in seperate function
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
            # should be in seperate function
            if get_position:
                x = x[0]
            store_x_val = x[0, :, :, slice_idx, 0]
            if get_slice:
                store_y_val = y[0, :, :, 0]
                store_y_pred_val = y_[0, :, :, 0]
            else:
                store_y_val = y[0, :, :, slice_idx, 0]
                store_y_pred_val = y_[0, :, :, slice_idx, 0]

            eloss_str = f" epoch: {epoch:3d}"
            for metric in metrics:
                epoch_loss_avg = metrics_epoch_avg[metric]['loss'].result()
                epoch_val_loss_avg = metrics_epoch_avg[metric]['val_loss'].result()
                metrics[metric]['history'].append(epoch_loss_avg.numpy())
                metrics[metric]['val history'].append(epoch_val_loss_avg.numpy())
                eloss_str += f", {metric}: {epoch_loss_avg: .5f}, val {metric}: {epoch_val_loss_avg: .5f}"

            print(f"{time.perf_counter() - epoch_time:3.0f} s" + eloss_str)

            f, axes = plt.subplots(3, 3)  # convert into for loop and seperate function
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
            f.suptitle(f"{model}: {train_name}\nepoch: {epoch}, loss: {metrics['loss']['history'][-1]: .5f}, val loss: {metrics['loss']['val history'][-1]: .5f}{clr}")
            plt.savefig(f"ckpt/checkpoints/{debug}train_session_{now_time}_{model}/progress/train_{epoch:04d}_{now_time}")
            plt.close('all')
            # print(f"plot saved: {time.perf_counter() - epoch_time:.0f}") plots are very fast to save, take ~1 second
            vnet.save_weights(f"ckpt/checkpoints/{debug}train_session_{now_time}_{model}/chkp/ckpt_{epoch:04d}_{now_time}.cktp")
    else:
        metric_list = []
        for idx, metric in enumerate(metrics):
            if idx == 0:
                continue
            metric_list.append(metrics[metric]['loss func'])
        vnet.compile(optimizer=Adam(lr=start_lr),
                     loss=loss_func,
                     metrics=metric_list,
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
            metrics['loss']['val history'] = h.history['val_loss']
            metrics[cross_entropy_name]['val history'] = h.history[f'val_{cross_entropy_name}']
            metrics['bce_dice_loss']['val history'] = h.history[f'val_bce_dice_loss']
        else:
            h = vnet.fit(x=tdataset, callbacks=callbacks, epochs=epochs, verbose=1)
        metrics['loss']['history'] = h.history['loss']
        metrics[cross_entropy_name]['history'] = h.history[f'{cross_entropy_name}']
        metrics['bce_dice_loss']['history'] = h.history[f'dice_loss']

    time_taken = time.perf_counter() - start_time
    roll_period = 5
    if roll_period > (epochs - 1):
        roll_period = epochs - 1
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(metrics['loss']['history'], label="loss")
    ax1.plot(running_mean(metrics['loss']['history'], roll_period), label="loss roll")
    if validate:
        ax1.plot(metrics[cross_entropy_name]['val history'], label="val loss")
        ax1.plot(running_mean(metrics[cross_entropy_name]['val history'], roll_period), label="val loss roll")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel(loss_name)
    ax1.set_title("loss: dice")
    ax1.legend()
    ax2.plot(metrics[cross_entropy_name]['history'], label="loss")
    ax2.plot(running_mean(metrics[cross_entropy_name]['history'], roll_period), label=f"roll loss")
    if validate:
        ax2.plot(metrics[cross_entropy_name]['val history'], label=f"val loss")
        ax2.plot(running_mean(metrics[cross_entropy_name]['val history'], roll_period), label=f"val roll loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("cross entropy")
    ax2.legend()
    ax3.plot(metrics['bce_dice_loss']['history'], label="loss")
    ax3.plot(running_mean(metrics['bce_dice_loss']['history'], roll_period), label=f"roll loss")
    if validate:
        ax3.plot(metrics['bce_dice_loss']['val history'], label=f"val loss")
        ax3.plot(running_mean(metrics['bce_dice_loss']['val history'], roll_period), label=f"val roll loss")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("bce + dice loss")
    ax3.legend()
    f.tight_layout(rect=[0, 0.01, 1, 0.93])
    min_loss = min(metrics[cross_entropy_name]['history'])
    min_val_loss = min(metrics[cross_entropy_name]['val history'])
    f.suptitle(f"{model}: {train_name}\nTime: {time_taken/60:.1f} mins, Min Loss: {min_loss:.5f}, Min Val Loss: {min_val_loss:.5f}")
    plt.savefig(f"ckpt/checkpoints/{debug}train_session_{now_time}_{model}/train_result_{now_time}")

    min_roll_loss = min(running_mean(metrics['loss']['history'], roll_period))
    min_roll_val_loss = min(running_mean(metrics['loss']['val history'], roll_period))

    min_ce = min(metrics[cross_entropy_name]['history'])
    min_val_ce = min(metrics[cross_entropy_name]['val history'])
    min_roll_ce = min(running_mean(metrics[cross_entropy_name]['history'], roll_period))
    min_roll_val_ce = min(running_mean(metrics[cross_entropy_name]['val history'], roll_period))

    min_dce = min(metrics['bce_dice_loss']['history'])
    min_val_dce = min(metrics['bce_dice_loss']['val history'])
    min_roll_dce = min(running_mean(metrics['bce_dice_loss']['history'], roll_period))
    min_roll_val_dce = min(running_mean(metrics['bce_dice_loss']['val history'], roll_period))

    if custom_train_loop:
        filenames = glob(f"ckpt/checkpoints/{debug}train_session_{now_time}_{model}/progress/*")
        filenames.sort()
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'ckpt/checkpoints/{debug}train_session_{now_time}_{model}/progress.gif', images)
    print(f"time taken: {time_taken:.1f}")

    df_losses = pd.DataFrame(data={
        'Loss': metrics['loss']['history'],
        'Loss Val': metrics['loss']['val history'],
        'BCE': metrics[cross_entropy_name]['history'],
        'BCE Val': metrics[cross_entropy_name]['val history'],
        'BCE + Dice': metrics['bce_dice_loss']['history'],
        'BCE + Dice Val': metrics['bce_dice_loss']['val history'],
        'Precision': metrics['precision']['history'],
        'Precision Val': metrics['precision']['val history'],
        'Recall': metrics['recall']['history'],
        'Recall Val': metrics['recall']['val history'],
    })
    df_losses.to_csv(f"ckpt/checkpoints/{debug}train_session_{now_time}_{model}/vnet_losses.csv", index_label="epoch")

    train_cols = {
        'Model': [model], 'Input Shape': [sample_shape], 'Learning Rate': [start_lr], 'Debug': [debug],
        'Loss': [loss_name], 'Optimizer': [use_optimizer],
        'Num. Epochs': [epochs], 'Examples/Epoch': [examples_per_load],
        'Min Loss': [min_loss], 'Min Val Loss': [min_val_loss],
        'Min Roll Loss': [min_roll_loss], 'Min Roll Val Loss': [min_roll_val_loss],
        'Min BCE': [min_ce], 'Min Val BCE': [min_val_ce],
        'Min Roll BCE': [min_roll_ce], 'Min Roll Val BCE': [min_roll_val_ce],
        'Min BCE + Dice Loss': [min_dce], 'Min Val BCE + Dice Loss': [min_val_dce],
        'Min Roll BCE + Dice Loss': [min_roll_dce], 'Min Roll Val BCE + Dice Loss': [min_roll_val_dce],
        'Fit': [custom_train_loop],
        'Mean loss of batch': [mean_loss_of_batch],
        'Schedule Drop': [schedule_drop], 'Schedule Epochs Drop': [schedule_epochs_drop], 'Train Duration': [time_taken],
        'Shuffle Order': [shuffle_order], 'Normalise Input': [normalise_input],
        'Remove Outliers': [remove_outliers], 'Skip Empty': [skip_empty],
        'transform_angle': [transform_angle], 'transform_position': [transform_position],
        'Commit ID': [commit_id], 'Train Name': [f"{debug}train_session_{now_time}_{model}"],
        'Model Params': [vnet.params], 'Notes': [notes],
    }
    df_experiments = pd.DataFrame(data=train_cols)
    df_experiments.to_csv("vnet_train_experiments.csv", index=False, header=False, mode='a')

    return time_taken


if __name__ == "__main__":

    print("running eagerly:", tf.executing_eagerly())

    import sys
    sys.path.insert(0, os.getcwd())

    if not os.path.exists("vnet_train_experiments.csv"):
        train_cols = {
            'Model': [], 'Input Shape': [], 'Learning Rate': [], 'Debug': [],
            'Loss': [], 'Optimizer': [],
            'Num. Epochs': [], 'Examples/Epoch': [],
            'Min Loss': [], 'Min Val Loss': [],
            'Min Roll Loss': [], 'Min Roll Val Loss': [],
            'Min BCE': [], 'Min Val BCE': [],
            'Min Roll BCE': [], 'Min Roll Val BCE': [],
            'Min BCE + Dice Loss': [], 'Min Val BCE + Dice Loss': [],
            'Min Roll BCE + Dice Loss': [], 'Min Roll Val BCE + Dice Loss': [],
            'Fit': [],
            'Mean loss of batch': [],
            'Schedule Drop': [], 'Schedule Epochs Drop': [], 'Train Duration': [],
            'Shuffle Order': [], 'Normalise Input': [],
            'Remove Outliers': [], 'Skip Empty': [],
            'transform_angle': [], 'transform_position': [],
            'Commit ID': [], 'Train Name': [],
            'Model Params': [], 'Notes': [],
        }
        df = pd.DataFrame(data=train_cols)
        df.to_csv("vnet_train_experiments.csv", index=False)

    debug = False
    if not debug:

        check_run = False

        e = 150
        if check_run:
            e = 3
        learn_rate = 5e-4
        repeats = 3

        # # for archer
        # for i in range(repeats):
        #     sample_shape = (64, 64, 64)
        #     train("tiny", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1, train_debug=check_run,
        #           train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
        #           use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.9,
        #           use_stride_2=True, use_res_connect=True,
        #           notes=f"Training Tiny VNet {sample_shape}, repeat: {i+1}/{repeats}")

        #     sample_shape = (160, 160, 160)
        #     train("tiny", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1, train_debug=check_run,
        #           train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
        #           use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.9,
        #           use_stride_2=True, use_res_connect=True,
        #           notes=f"Training Tiny VNet {sample_shape}, repeat: {i+1}/{repeats}")

        #     sample_shape = (160, 160, 160)
        #     train("small", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1, train_debug=check_run,
        #           train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
        #           use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.9,
        #           use_stride_2=True, use_res_connect=True,
        #           notes=f"Training Small VNet {sample_shape}, repeat: {i+1}/{repeats}")

        #     sample_shape = (160, 160, 160)
        #     train("small_relative", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1, train_debug=check_run,
        #           train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
        #           use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.9,
        #           use_stride_2=True, use_res_connect=True,
        #           notes=f"Training Small Relative VNet {sample_shape}, repeat: {i+1}/{repeats}")

        #     sample_shape = (160, 160, 5)
        #     train("slice", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1, train_debug=check_run,
        #           train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
        #           use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.9,
        #           use_stride_2=True, use_res_connect=True,
        #           notes=f"Training Small Slice VNet {sample_shape}, repeat: {i+1}/{repeats}")

        # for pompeii
        for i in range(repeats):
            sample_shape = (240, 240, 160)
            train("small", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1, train_debug=check_run,
                  train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
                  use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.9,
                  use_stride_2=True, use_res_connect=True,
                  notes=f"Training Small VNet {sample_shape}, repeat: {i+1}/{repeats}")

            sample_shape = (240, 240, 160)
            train("large", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1, train_debug=check_run,
                  train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
                  use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.9,
                  use_stride_2=True, use_res_connect=True,
                  notes=f"Training Large VNet {sample_shape}, repeat: {i+1}/{repeats}")

            sample_shape = (240, 240, 160)
            train("large_relative", batch_size=1, sample_shape=sample_shape, epochs=e, examples_per_load=1, train_debug=check_run,
                  train_name=f"{sample_shape}, Adam Schedule {learn_rate}, dice, VNet", custom_train_loop=True,
                  use_optimizer="adam_schedule", start_lr=learn_rate, schedule_epochs_drop=5, schedule_drop=0.9,
                  use_stride_2=True, use_res_connect=True,
                  notes=f"Training Large Relative VNet {sample_shape}, repeat: {i+1}/{repeats}")

    else:
        e = 3
        train("tiny", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug 3D UNet (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True)
        train("tiny", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug VNet (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True,
              use_stride_2=True, use_res_connect=True)

        train("small", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug 3D UNet (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True)
        train("small", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug VNet (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True,
              use_stride_2=True, use_res_connect=True)

        train("large", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug 3D UNet (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True)
        train("large", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug VNet (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True,
              use_stride_2=True, use_res_connect=True)

        train("small_relative", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug 3D UNet (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True)
        train("small_relative", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug VNet (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True,
              use_stride_2=True, use_res_connect=True)

        train("large_relative", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug 3D UNet (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True)
        train("large_relative", batch_size=1, sample_shape=(160, 160, 160), epochs=e, examples_per_load=1,
              train_name="debug VNet (160,160,160), Adam, dice", custom_train_loop=True, train_debug=True,
              use_stride_2=True, use_res_connect=True)

        train("slice", batch_size=1, sample_shape=(160, 160, 5), epochs=e, examples_per_load=1,
              train_name="debug 3D UNet (160,160,5), Adam, dice", custom_train_loop=True, train_debug=True)
        train("slice", batch_size=1, sample_shape=(160, 160, 5), epochs=e, examples_per_load=1,
              train_name="debug VNet (160,160,5), Adam, dice", custom_train_loop=True, train_debug=True,
              use_stride_2=True, use_res_connect=True)
