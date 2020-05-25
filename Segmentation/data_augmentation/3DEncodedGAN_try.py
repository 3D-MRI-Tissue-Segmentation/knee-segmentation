import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import h5py

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 16

with h5py.File("../../Data/Tests_data/3d-mnist/full_dataset_vectors.h5", "r") as hf:
    print(hf)   
    X_train = hf["X_train"][:]
    y_train = hf["y_train"][:]  
    X_test = hf["X_test"][:]
    y_test = hf["y_test"][:]

#Reshaping the 3D mnist
X_train = np.reshape(X_train, (X_train.shape[0], 16, 16, 16))
X_test = np.reshape(X_test, (X_test.shape[0], 16, 16, 16))
assert X_train.shape == (X_train.shape[0], 16, 16, 16), f"X_train's shape is {X_train.shape} != ({X_train.shape[0]}, 16, 16, 16)"

print(f"\n the images in the train dataset are of size {X_train.shape}")

#Create 2D dataset to train the encoder
plt.imshow(X_train[0,:,:,6])
plt.show()

X_train_2d = X_train[:,:,:,6]
assert X_train.shape == (X_train.shape[0], 16, 16, 16), f"X_train's shape is {X_train_2d.shape} != ({X_train_2d.shape[0]}, 16, 16)"

dim = (1, 1, 1, batch_size) #Set the dimensions of the first input to the generator onvolution

def encoder_model():
    model = tf.keras.Sequential()

    #Fist convolutional layer
    model.add(layers.Conv2D(64, (11,11), strides=4, padding='same', input_shape=[16,16,1]))
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    #Second covolutional layer
    model.add(layers.Conv2D(128, (5,5), strides=4, padding='same'))
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    #Third convolutional layer
    model.add(layers.Conv2D(256, (5,5), strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    #Fourth covolutional layer
    model.add(layers.Conv2D(512, (5,5), strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    #Fifth covolutional layer
    model.add(layers.Conv2D(400, (8,8), strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    #First dense layer
    model.add(layers.Dense(200))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #First dense layer
    model.add(layers.Dense(200))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    return model

def generator_model(noise_len):
    model = tf.keras.Sequential()

    #Adding first dense layer
    model.add(layers.Dense(2048, use_bias = False, input_shape=(noise_len, )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((dim[1], dim[2], dim[3])))
    assert model.output_shape == (None, dim[1], dim[2], dim[3])

    #First transpose convolutional layer
    model.add(layers.Conv3DTranspose(256, (4,4,4), strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #Second transpose convolutional layer
    model.add(layers.Conv3DTranspose(128, (4,4,4), strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #Third transpose convolutional layer
    model.add(layers.Conv3DTranspose(64, (4,4,4), strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #Last transpose convolutional layer
    model.add(layers.Conv3DTranspose(1, (4,4,4), strides=2, padding='same', use_bias=False, activation='tanh'))
    model.add(layers.tanh())
    assert model.output_shape == (None, 16, 16, 16, 1)

    return model

def discriminator_model():
    model = tf.keras.Sequential()

    #Fist convolutional layer
    model.add(layers.Conv3D(32, (4,4,4), strides=2, padding='same', input_shape=[28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    #Second covolutional layer
    model.add(layers.Conv3D(64, (4,4,4), strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    #Third convolutional layer
    model.add(layers.Conv3D(128, (4,4,4), strides=2, padding='same', input_shape=[28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    #Fourth covolutional layer
    model.add(layers.Conv3D(256, (4,4,4), strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    #Dense Layer for output --> therefore flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

#Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)  #Defining the cross entropy loss function

def discriminator_loss(real_ouput, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_ouput), real_ouput)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

#Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#Training
epochs = 3
number_of_generated_output = 2
noise_length = 100
seed = tf.random.normal([number_of_generated_output, noise_length])

generator = generator_model(noise_length)
discriminator = discriminator_model()

