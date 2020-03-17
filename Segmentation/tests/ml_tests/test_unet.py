import tensorflow as tf


class UNetTest(tf.test.TestCase):

    def testVanillaUNet(self):
        import json
        import os

        config = 'config.json'
        if os.path.isfile(config):
            with open(config) as json_data_file:
                data = json.load(json_data_file)
            if data['train'] == 'false':
                return
        tf.random.set_seed(83922)

        dataset_size = 10
        batch_size = 5
        input_shape = (256, 256, 1)
        num_classes = 3
        output_shape = (256, 256, num_classes)
        num_filters = 64  # number of filters at the start

        features = tf.random.normal((dataset_size,) + input_shape)
        labels = tf.random.normal((dataset_size,) + output_shape)

        features, labels = self.evaluate([features, labels])
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

        from Segmentation.model.unet import UNet  # local file import
        model = UNet(num_filters, num_classes)
        from Segmentation.utils.training_utils import dice_coef_loss  # local file import
        model.compile('adam', loss=dice_coef_loss)
        history = model.fit(dataset,
                            steps_per_epoch=dataset_size // batch_size,
                            epochs=2)


if __name__ == '__main__':
    import sys
    from os import getcwd
    sys.path.insert(0, getcwd())

    tf.test.main()
