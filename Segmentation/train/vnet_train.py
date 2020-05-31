import sys
import os
import tensorflow as tf


def train_model(model):
    print(model.name)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())

    from Segmentation.train.utils import setup_gpu

    setup_gpu()

    num_channels = 16
    num_classes = 1  # binary segmentation problem

    from Segmentation.model.vnet import VNet
    model = VNet(num_channels, num_classes)

    train_model(model)
