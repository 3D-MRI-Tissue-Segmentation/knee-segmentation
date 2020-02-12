import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from translate_image import pad_image, shift_image_and_label, rotate_image_and_label


def load_3d_mnist(visulise=False):
    import pandas as pd
    df = pd.read_csv("./Data/Tests_data/mnist_3d_centre_test_data.csv")
    centres = df.to_numpy()
    print(centres)
    import h5py
    with h5py.File("./Data/Tests_data/full_dataset_vectors.h5") as hf:
        data = hf["X_train"][:]
    data = np.reshape(data, (data.shape[0], 16, 16, 16, 1))
    x_train = []
    region_radiuses = []
    y_train = []
    for centres_row in centres:
        blank = np.zeros(data.shape[1:])
        x, y, z = int(centres_row[2]), int(centres_row[3]), int(centres_row[4])
        blank[x, y, z] = 1
        y_train.append(blank)
        x_train.append(data[int(centres_row[0]), :, :, :])
        region_radiuses.append(int(centres_row[1]))
    if visulise:
        plt.hist(region_radiuses)
        plt.show()
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)
    return x_train, y_train, max(region_radiuses)


def load_2d_mnist(visulise=False):
    import pandas as pd
    df = pd.read_csv("./Data/Tests_data/mnist_2d_centre_test_data_.csv")
    centres = df.to_numpy()
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    data = x_train
    data = np.reshape(data, (data.shape[0], 28, 28, 1))
    x_train = []
    region_radiuses = []
    y_train = []
    for centres_row in centres:
        blank = np.zeros(data.shape[1:])
        x, y = int(centres_row[2]), int(centres_row[3])
        blank[x, y] = 1
        y_train.append(blank)
        x_train.append(data[int(centres_row[0]), :, :])
        region_radiuses.append(int(centres_row[1]))
    if visulise:
        plt.hist(region_radiuses)
        plt.show()
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)
    return x_train, y_train, max(region_radiuses)


def build_3d_model(x_train_shape, depth=2):
    assert depth >= 1
    assert x_train_shape[1] == x_train_shape[2] == x_train_shape[3], "All dimensions need to be the same"
    assert (x_train_shape[1] % (2 * 2**depth)) == 0, "Shape / 2**(depth + 1) needs to be an integer"
    model = tf.keras.Sequential()
    # input
    model.add(tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3),  # (original dim)
                                     activation="relu", padding="same", input_shape=x_train_shape[1:]))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling3D())  # (original dim / 2)
    # downsample
    for i in range(depth):
        model.add(tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3),  # original dim / (2 * 2**i))
                                         activation="relu", padding="same"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling3D())
    # middle convolution
    model.add(tf.keras.layers.Convolution3DTranspose(32, kernel_size=(3, 3, 3),  # original dim / (2 ** (depth+1))
                                                     activation="relu", padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    # upsample
    for i in range(depth):
        model.add(tf.keras.layers.Convolution3DTranspose(32, kernel_size=(3, 3, 3), strides=2,  # original dim / (2 ** (depth - i))
                                                         activation="relu", padding="same"))
        model.add(tf.keras.layers.BatchNormalization())
    # output
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Convolution3DTranspose(1, kernel_size=(3, 3, 3), strides=2,
                                                     activation="sigmoid", padding="same"))  # original dim
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    return model


def build_2d_model(x_train_shape, depth=1):
    assert depth >= 1
    assert x_train_shape[1] == x_train_shape[2], "All dimensions need to be the same"
    assert (x_train_shape[1] % (2 * 2**depth)) == 0, "Shape / 2**(depth + 1) needs to be an integer"
    model = tf.keras.Sequential()
    # input
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),  # (original dim)
                                     activation="relu", padding="same", input_shape=x_train_shape[1:]))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())  # (original dim / 2)
    # downsample
    for i in range(depth):
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),  # original dim / (2 * 2**i))
                                         activation="relu", padding="same"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D())
    # middle convolution
    model.add(tf.keras.layers.Convolution2DTranspose(32, kernel_size=(3, 3),  # original dim / (2 ** (depth+1))
                                                     activation="relu", padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    # upsample
    for i in range(depth):
        model.add(tf.keras.layers.Convolution2DTranspose(32, kernel_size=(3, 3), strides=2,  # original dim / (2 ** (depth - i))
                                                         activation="relu", padding="same"))
        model.add(tf.keras.layers.BatchNormalization())
    # output
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Convolution2DTranspose(1, kernel_size=(3, 3), strides=2,
                                                     activation="sigmoid", padding="same"))  # original dim
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def main_3d():
    x_train, y_train, max_rad = load_3d_mnist()
    model = build_3d_model(x_train.shape)
    model.fit(x_train, y_train,
              batch_size=5, epochs=10, shuffle=True)
    y_pred = model.predict(x_train)
    for i, i_true in zip(y_pred, y_train):
        print(np.unravel_index(i.argmax(), i.shape), np.unravel_index(i_true.argmax(), i_true.shape))


def main_2d():
    x_train, y_train, max_rad = load_2d_mnist()
    model = build_2d_model(x_train.shape)
    model.fit(x_train, y_train,
              batch_size=5, epochs=100, shuffle=True)
    y_pred = model.predict(x_train)
    for i, i_true in zip(y_pred, y_train):
        print(np.unravel_index(i.argmax(), i.shape), np.unravel_index(i_true.argmax(), i_true.shape))


if __name__ == "__main__":
    main_2d()
