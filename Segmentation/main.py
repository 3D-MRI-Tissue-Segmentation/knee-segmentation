import tensorflow as tf

import os
from pathlib import Path
from datetime import datetime
from absl import app
from absl import logging

from Segmentation.utils.data_loader import read_tfrecord
from Segmentation.utils.data_loader import parse_fn_2d, parse_fn_3d
from Segmentation.utils.losses import dice_coef_loss, tversky_loss, dice_coef, iou_loss  # focal_tversky
from Segmentation.utils.evaluation_metrics import dice_coef_eval, iou_loss_eval
from Segmentation.utils.training_utils import LearningRateSchedule
# from Segmentation.utils.evaluation_utils import plot_and_eval_3D, confusion_matrix, epoch_gif, volume_gif, take_slice
from Segmentation.utils.evaluation_utils import eval_loop
from Segmentation.train.train import Train

from flags import FLAGS
from select_model import select_model


def main(argv):

    if FLAGS.visual_file:
        assert FLAGS.train is False, "Train must be set to False if you are doing a visual."

    del argv  # unused arg
    tf.random.set_seed(FLAGS.seed)  # set seed

    # # ---------------------------------------------------------------------------------
    # set whether to train on GPU or TPU
    # # def setup_accelerator(use_gpu=FLAGS.use_gpu, num_cores=FLAGS.num_cores, tpu_name=FLAGS.tpu)
    # # --------------------------------------------------------------------------------

    # # --------------------------------------------------------------------------------
    # set dataset configuration
    # #  def load_dataset()
    # # --------------------------------------------------------------------------------

    # # --------------------------------------------------------------------------------
    # # def set_metrics()
    # # --------------------------------------------------------------------------------

    # set model architecture
    # model_fn, model_args = select_model(FLAGS, num_classes)

    # # --------------------------------------------------------------------------------
    # # if FLAGS.train:
    # #     def train():
    # # --------------------------------------------------------------------------------

    # # --------------------------------------------------------------------------------
    # else:
    # # def eval():
    # # --------------------------------------------------------------------------------


if __name__ == '__main__':
    app.run(main)
