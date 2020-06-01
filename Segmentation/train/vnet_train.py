import sys
import os
import tensorflow as tf


def train_model(model):
    print(model.name)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())

    from Segmentation.train.utils import setup_gpu
    from Segmentation.utils.losses import dice_loss

    setup_gpu()

    num_channels = 16
    num_classes = 1  # binary segmentation problem
    buffer_size = 3
    batch_size = 2
    tfrec_dir = './Data/tfrecords/'

    from Segmentation.model.vnet import VNet
    model = VNet(num_channels, num_classes)

    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(optimizer=optimiser,
                  loss=dice_loss,
                  metrics=['binary_crossentropy', 'acc'])

    train_model(model)

    from Segmentation.utils.data_loader import read_tfrecord, parse_fn_3d

    train_ds = read_tfrecord(tfrecords_dir=os.path.join(tfrec_dir, 'train_3d/'),
                             batch_size=batch_size,
                             buffer_size=buffer_size,
                             parse_fn=parse_fn_3d,
                             multi_class=False,
                             is_training=True)
    valid_ds = read_tfrecord(tfrecords_dir=os.path.join(tfrec_dir, 'valid_3d/'),
                             batch_size=batch_size,
                             buffer_size=buffer_size,
                             parse_fn=parse_fn_3d,
                             multi_class=False,
                             is_training=False)

    # history = model.fit(train_ds,
    #                     steps_per_epoch=20,
    #                     epochs=10,
    #                     validation_data=valid_ds,
    #                     validation_steps=10)

    import matplotlib.pyplot as plt

    for idx, i in enumerate(train_ds):
        print(type(i), idx)
        print(i[0].shape)
        print(i[1].shape)
        print(i[2])
        print(i[3])
        print(i[4])
        print(i[5])
        print("==========")
        break
