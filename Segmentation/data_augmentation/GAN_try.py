import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

BATCH_SIZE = 256

#load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_train = x_train/255
x_test = x_test/255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BATCH_SIZE).batch(BATCH_SIZE)

#define metrics
gen_loss_label = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
d_loss_label = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
d_accurcy_real = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy_real')
d_accurcy_fake = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy_fake')

# #showing image in mnist
# plt.imshow(x_train[0,:,:,0],cmap='gray')
# plt.show()

def generator_model(noise_len):
    model = tf.keras.Sequential()

    #Adding first dense layer
    model.add(layers.Dense(7*7*BATCH_SIZE, use_bias = False, input_shape=(noise_len, )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, BATCH_SIZE)))
    assert model.output_shape == (None, 7, 7, BATCH_SIZE)

    #First transpose convolutional layer
    model.add(layers.Conv2DTranspose(128, (5,5), strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #Second transpose convolutional layer
    model.add(layers.Conv2DTranspose(64, (5,5), strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #Last transpose convolutional layer
    model.add(layers.Conv2DTranspose(1, (5,5), strides=2, padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

#Test the genrator with random noise
noise_length = 100

generator_tryingOut = generator_model(noise_length)
generator_tryingOut.summary()

noise = tf.random.normal([1, noise_length])
generated_image = generator_tryingOut(noise, training=False)

# plt.imshow(generated_image[0,:,:,0], cmap='gray')
# plt.show()

def discriminator_model():
    model = tf.keras.Sequential()

    #Fist convolutional layer
    model.add(layers.Conv2D(64, 5, strides=2, padding='same', input_shape=[28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    #Second covolutional layer
    model.add(layers.Conv2D(128, 5, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    #Dense Layer for output --> therefore flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# discriminator_tryingOut = discriminator_model()
# decision = discriminator_tryingOut(generated_image)
# print(decision)

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

#Accuracy
def discriminator_accuracy(real_pred, fake_pred):
    real_true = tf.ones_like(real_pred)
    fake_true = tf.zeros_like(fake_pred)
    d_accurcy_real.update_state(real_true, tf.math.round(real_pred))
    d_accurcy_fake.update_state(fake_true, tf.math.round(fake_pred))
    #print (tf.shape(real_pred))
    #d_accurcy.update_state(tf.concat(values=[real_true, fake_true], axis=1), tf.concat(values=[real_pred, fake_pred], axis=1))


#Training
epochs = 3
number_of_generated_output = 2
noise_length = 100
seed = tf.random.normal([number_of_generated_output, noise_length])

generator = generator_model(noise_length)
discriminator = discriminator_model()

@tf.function #This will run the function as a graph which will have a faster execution and will be using the GPU
def training_epoch(images):
    noise = tf.random.normal([BATCH_SIZE, noise_length])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as d_tape:
        generated_images = generator(noise, training=True)

        real_out = discriminator(images, training=True)
        fake_out = discriminator(generated_images, training=True)

        #Calculating the loss
        gen_loss = generator_loss(fake_out)
        d_loss = discriminator_loss(real_out, fake_out)

    #Gettng the gradient of the weights based on the loss
    gen_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    d_gradient = d_tape.gradient(d_loss, discriminator.trainable_variables)

    #Applying the gradients through the optimizer
    generator_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(d_gradient, discriminator.trainable_variables))

    gen_loss_label.update_state(gen_loss)
    d_loss_label.update_state(d_loss)
   # print (real_out.numpy())
    discriminator_accuracy(real_out, fake_out)
    
    # show_im(generated_images.numpy())

def training(n_epochs, dataset):
    for epoch in range(epochs):
        for batch_im in dataset:
            training_epoch(batch_im)
            
        template = 'Generator Loss: {}, Discrimintor Loss: {}, Discriminator Accuracy_real: {}, Discriminator Accuracy_fake: {}'
        print(f"----------------\t{(epoch+1)}\t----------------")
        print (template.format(gen_loss_label.result(), 
                                d_loss_label.result(),
                                d_accurcy_real.result()*100,
                                d_accurcy_fake.result()*100) + "\n")
        
        gen_loss_label.reset_states()
        d_loss_label.reset_states()
        d_accurcy_real.reset_states()
        d_accurcy_fake.reset_states()

        show_evolution()

# def generate_showIm():
#     prediction = generator(seed, training=False)
#     plt.imshow(prediction[0,:,:,0], cmap='gray')
#     plt.show()

def show_evolution():
    prediction = generator(seed, training=False)
    plt.imshow(prediction[0,:,:,0], cmap='gray')
    plt.draw()
    #plt.pause(0.001)

training(epochs, x_dataset)
# generate_showIm()
