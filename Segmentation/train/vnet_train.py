import sys
import os
from glob import glob
import datetime
import tensorflow as tf

<<<<<<< HEAD

def load_datasets(batch_size, buffer_size,
                  tfrec_dir='./Data/tfrecords/', multi_class=False):
    """
    Loads tf records datasets for 3D models.
    """
    from Segmentation.utils.data_loader import read_tfrecord, parse_fn_3d
    train_ds = read_tfrecord(tfrecords_dir=os.path.join(tfrec_dir, 'train_3d/'),
                             batch_size=batch_size,
                             buffer_size=buffer_size,
                             parse_fn=parse_fn_3d,
                             multi_class=multi_class,
                             is_training=True)
    valid_ds = read_tfrecord(tfrecords_dir=os.path.join(tfrec_dir, 'valid_3d/'),
                             batch_size=batch_size,
                             buffer_size=buffer_size,
                             parse_fn=parse_fn_3d,
                             multi_class=multi_class,
                             is_training=False)
    return train_ds, valid_ds


def build_model(num_channels, num_classes, lr=1e-4):
    """
    Builds standard vnet for 3D
    """
    from Segmentation.model.vnet import VNet
    from Segmentation.utils.losses import dice_loss

    model = VNet(num_channels, num_classes)

    optimiser = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimiser,
                  loss=dice_loss,
                  metrics=['binary_crossentropy', 'acc'])
    return model

def train_model(model, train_ds, valid_ds, epochs, batch_size,
                tfrec_dir='./Data/tfrecords/'):
    """
    Trains 3D model with keras fit
    """
    train_size = len(glob(os.path.join(tfrec_dir, 'train_3d/*')))
    valid_size = len(glob(os.path.join(tfrec_dir, 'valid_3d/*')))

    log_dir = "logs/vnet/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_ds,
                        steps_per_epoch=train_size // batch_size,
                        epochs=epochs,
                        validation_data=valid_ds,
                        validation_steps=valid_size // batch_size,
                        callbacks=[tensorboard_callback],
                        verbose=1)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())

    from Segmentation.train.utils import setup_gpu

    setup_gpu()

    num_channels = 1
    num_classes = 1  # binary segmentation problem
    buffer_size = 4
    epochs = 3
    batch_size = 1
    tfrec_dir = './Data/tfrecords/'

    model = build_model(num_channels, num_classes)
    train_ds, valid_ds = load_datasets(batch_size, buffer_size, tfrec_dir)
    train_model(model, train_ds, valid_ds, tfrec_dir, epochs, batch_size)
=======
def train_model(model):
    print(model.name)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())

    from Segmentation.train.utils import setup_gpu
    from Segmentation.utils.losses import dice_loss

    setup_gpu()

    num_channels = 4
    num_classes = 1  # binary segmentation problem
    buffer_size = 4
    batch_size = 1
    tfrec_dir = './Data/tfrecords/'

    for i in range(4):

        from Segmentation.model.vnet import VNet
        model = VNet(num_channels, num_classes)

        optimiser = tf.keras.optimizers.Adam(learning_rate=1e-4)

        model.compile(optimizer=optimiser,
                      loss=dice_loss,
                      metrics=['binary_crossentropy'])

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

        train_size = len(glob(os.path.join(tfrec_dir, 'train_3d/*')))
        valid_size = len(glob(os.path.join(tfrec_dir, 'valid_3d/*')))

        log_dir = "logs/vnet/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)

        history = model.fit(train_ds,
                            steps_per_epoch=train_size // batch_size,
                            epochs=25,
                            validation_data=valid_ds,
                            validation_steps=valid_size // batch_size,
                            callbacks=[tensorboard_callback],
                            verbose=1)
>>>>>>> 4bbf767be637556ea94342fbf611743830aefde0
