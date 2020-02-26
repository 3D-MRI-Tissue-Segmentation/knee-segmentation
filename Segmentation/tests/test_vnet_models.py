import tensorflow as tf
from absl.testing import parameterized


class Test_VNet(parameterized.TestCase, tf.test.TestCase):

    # def setUp(self):
    #     assert True

    def create_test_volume(self, n_volumes, n_reps, n_classes,
                           width, height, depth, colour_channels):
        from Segmentation.data_gen.toy_volume_gen import get_test_volumes
        return get_test_volumes(n_volumes, n_reps, n_classes,
                                width, height, depth, colour_channels)

    @parameterized.named_parameters(
        {"testcase_name": "tiny_28-28-16-1-merge", "model": "tiny", "height": 28, "width": 28, "depth": 16, "colour_channels": 1, "merge_connections": True},
        {"testcase_name": "tiny_28-28-16-1-no-merge", "model": "tiny", "height": 28, "width": 28, "depth": 16, "colour_channels": 1, "merge_connections": False},
        {"testcase_name": "tiny_28-28-16-3-merge", "model": "tiny", "height": 28, "width": 28, "depth": 16, "colour_channels": 3, "merge_connections": True},
        {"testcase_name": "tiny_48-48-16-1-merge", "model": "tiny", "height": 48, "width": 48, "depth": 16, "colour_channels": 1, "merge_connections": True},

        {"testcase_name": "small_28-28-16-1-merge", "model": "small", "height": 28, "width": 28, "depth": 16, "colour_channels": 1, "merge_connections": True},
        {"testcase_name": "small_28-28-16-1-no-merge", "model": "small", "height": 28, "width": 28, "depth": 16, "colour_channels": 1, "merge_connections": False},
        {"testcase_name": "small_28-28-16-3-merge", "model": "small", "height": 28, "width": 28, "depth": 16, "colour_channels": 3, "merge_connections": True},
        {"testcase_name": "small_48-48-16-1-merge", "model": "small", "height": 48, "width": 48, "depth": 16, "colour_channels": 1, "merge_connections": True},
        {"testcase_name": "small_16-16-16-1-merge", "model": "small", "height": 16, "width": 16, "depth": 16, "colour_channels": 1, "merge_connections": True},
    )
    def test_vnet(self, model, height, width, depth, colour_channels, merge_connections,
                  epochs=2, n_volumes=3, n_reps=2, n_classes=2):
        volumes, one_hots = self.create_test_volume(n_volumes, n_reps, n_classes,
                                                    width, height, depth, colour_channels)

        def build_vnet(model, colour_channels, n_classes, merge_connections):
            if model == "tiny":
                from Segmentation.model.vnet_tiny import VNet_Tiny
                return VNet_Tiny(colour_channels, n_classes, merge_connections)
            elif model == "small":
                from Segmentation.model.vnet_small import VNet_Small
                return VNet_Small(colour_channels, n_classes, merge_connections)

        vnet = build_vnet(model, colour_channels, n_classes, merge_connections)

        def vnet_feedforward(vnet, volumes, one_hots):
            from tensorflow.keras.optimizers import Adam
            from Segmentation.utils.training_utils import tversky_loss, dice_loss
            vnet.compile(optimizer=Adam(0.00001),
                         loss=dice_loss,
                         metrics=['categorical_crossentropy'],
                         experimental_run_tf_function=True)
            output = vnet(volumes)
            assert output.shape == one_hots.shape

        vnet_feedforward(vnet, volumes, one_hots)

        def vnet_fit(vnet, volumes, one_hots, epochs):
            history = vnet.fit(volumes, one_hots, epochs=epochs)
            loss_history = history.history['loss']
            loss_history = history.history['loss']
            pred = vnet.predict(volumes)
            assert pred.shape == one_hots.shape

        vnet_fit(vnet, volumes, one_hots, epochs)


if __name__ == '__main__':
    import sys
    from os import getcwd
    sys.path.insert(0, getcwd())

    tf.test.main()
