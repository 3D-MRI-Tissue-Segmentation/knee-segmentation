import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from translate_image import pad_image, shift_image_and_label, rotate_image_and_label


if __name__=="__main__":
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    print(train_images.shape)

    example_image = train_images[0]
    plt.imshow(example_image)
    plt.show()
