import tensorflow as tf
import numpy as np
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
        {"testcase_name": "tiny_28-28-16-1_merge", "model": "tiny", "shape": (28, 28, 16, 1), "merge_connections": True},
        {"testcase_name": "tiny_28-28-16-1_no-merge", "model": "tiny", "shape": (28, 28, 16, 1), "merge_connections": False},
        {"testcase_name": "tiny_28-28-16-3_merge", "model": "tiny", "shape": (28, 28, 16, 3), "merge_connections": True},
        {"testcase_name": "tiny_48-48-16-1_merge", "model": "tiny", "shape": (48, 48, 16, 1), "merge_connections": True},

        {"testcase_name": "small_28-28-16-1_merge", "model": "small", "shape": (28, 28, 16, 1), "merge_connections": True},
        {"testcase_name": "small_28-28-16-1_no-merge", "model": "small", "shape": (28, 28, 16, 1), "merge_connections": False},
        {"testcase_name": "small_28-28-16-3_merge", "model": "small", "shape": (28, 28, 16, 3), "merge_connections": True},
        {"testcase_name": "small_48-48-16-1_merge", "model": "small", "shape": (48, 48, 16, 1), "merge_connections": True},
        {"testcase_name": "small_16-16-16-1_merge", "model": "small", "shape": (16, 16, 16, 1), "merge_connections": True},

        {"testcase_name": "small_relative_96-96-96-1_merge_multi", "model": "small_relative", "shape": (96, 96, 96, 1), "merge_connections": True, "relative": True, 'relative_action': "multiply"},
        {"testcase_name": "small_relative_96-96-96-1_merge_add", "model": "small_relative", "shape": (96, 96, 96, 1), "merge_connections": True, "relative": True, 'relative_action': "add"},
        {"testcase_name": "small_relative_128-128-128-1_merge_multi", "model": "small_relative", "shape": (128, 128, 128, 1), "merge_connections": True, "relative": True, 'relative_action': "multiply"},
        {"testcase_name": "small_relative_128-128-128-1_no-merge_multi", "model": "small_relative", "shape": (128, 128, 128, 1), "merge_connections": False, "relative": True, 'relative_action': "multiply"},
    )
    def test_run(self, model, shape, merge_connections,
                 epochs=2, n_volumes=3, n_reps=2, n_classes=2, relative=False, relative_action=None):
        height, width, depth, colour_channels = shape
        self.run_vnet(model, height, width, depth,
                      colour_channels, merge_connections,
                      epochs, n_volumes, n_reps, n_classes,
                      relative, relative_action)

    def run_vnet(self, model, height, width, depth, colour_channels, merge_connections,
                 epochs=2, n_volumes=3, n_reps=2, n_classes=2, relative=False, relative_action=None):
        volumes, one_hots = self.create_test_volume(n_volumes, n_reps, n_classes,
                                                    width, height, depth, colour_channels)
        inputs = volumes
        if relative:
            pos = np.array([[1.0, 0.0, -0.5]], dtype=np.float32)

            pos = np.repeat(pos, volumes.shape[0], axis=0)
            assert volumes.shape[0] == pos.shape[0]
            inputs = [volumes, pos]

        def build_vnet(model, colour_channels, n_classes, merge_connections):
            if model == "tiny":
                from Segmentation.model.vnet_tiny import VNet_Tiny
                return VNet_Tiny(colour_channels, n_classes, merge_connections=merge_connections)
            elif model == "small":
                from Segmentation.model.vnet_small import VNet_Small
                return VNet_Small(colour_channels, n_classes, merge_connections=merge_connections)
            elif model == "small_relative":
                from Segmentation.model.vnet_small_relative import VNet_Small_Relative
                return VNet_Small_Relative(colour_channels, n_classes, merge_connections=merge_connections,
                                           action=relative_action)
            else:
                raise NotImplementedError(f"no model named: {model}")

        vnet = build_vnet(model, colour_channels, n_classes, merge_connections)

        def vnet_feedforward(vnet, inputs, one_hots):
            output = vnet(inputs)
            assert output.shape == one_hots.shape
            output = vnet.predict(inputs)
            assert output.shape == one_hots.shape

        vnet_feedforward(vnet, inputs, one_hots)

        def vnet_fit(vnet, inputs, one_hots, epochs):
            from tensorflow.keras.optimizers import Adam
            from Segmentation.utils.training_utils import tversky_loss, dice_loss

            # grads = tape.gradient(loss, model.trainable_variables)
            # grads = [grad if grad is not None else tf.zeros_like(var)
            # for var, grad in zip(model.trainable_variables, grads)]

            vnet.compile(optimizer=Adam(0.00001),
                         loss=dice_loss,
                         metrics=['categorical_crossentropy'],
                         experimental_run_tf_function=True)

            history = vnet.fit(x=inputs, y=one_hots, epochs=epochs, verbose=1)
            loss_history = history.history['loss']
            loss_history = history.history['loss']
            pred = vnet.predict(inputs)
            assert pred.shape == one_hots.shape

        vnet_fit(vnet, inputs, one_hots, epochs)

if __name__ == '__main__':
    import sys
    from os import getcwd
    sys.path.insert(0, getcwd())

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    tf.test.main()

    # tv = Test_VNet()
    # tv.run_vnet(model="small_relative",
    #             height=96, width=96, depth=96, colour_channels=1,
    #             merge_connections=False, relative=True, relative_action="add", epochs=20)
