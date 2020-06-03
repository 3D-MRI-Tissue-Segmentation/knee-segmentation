import tensorflow as tf
from glob import glob

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

@tf.function
def train_step(model, loss_func, optimizer, x_train, y_train, train_loss, visualise=False):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_func(y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)
    if visualise:
        return predictions
    return

@tf.function
def test_step(model, loss_func, x_test, y_test, test_loss, visualise=False):
    predictions = model(x_test, training=False)
    loss = loss_func(y_test, predictions)
    test_loss(loss)
    if visualise:
        return predictions
    return
