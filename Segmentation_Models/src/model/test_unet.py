import tensorflow as tf
from unet import UNet, build_unet # local file import
from src.utils.training_utils import dice_coef_loss #local file import

class UNetTest(tf.test.TestCase):

    def testVanillaUNet(self):
        tf.random.set_seed(83922)

        dataset_size = 10
        batch_size = 5
        input_shape = (256, 256, 1)
        num_classes = 3
        output_shape = (256,256, num_classes)
        num_filters = 64 #number of filters at the start

        features = tf.random.normal((dataset_size,) + input_shape)
        labels = tf.random.normal((dataset_size,) + output_shape)

        features, labels = self.evaluate([features, labels])
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

        model = UNet(num_filters, num_classes)
        model.compile('adam', loss=dice_coef_loss)
        history = model.fit(dataset,
                        steps_per_epoch=dataset_size // batch_size,
                        epochs=2)
        loss_history = history.history['loss']

        loss_history = history.history['loss']
        self.assertAllGreaterEqual(loss_history, 0.)
        self.assertGreater(loss_history[0], loss_history[-1])

if __name__ == '__main__':
  tf.test.main()
