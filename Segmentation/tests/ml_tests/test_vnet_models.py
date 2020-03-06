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
                 epochs=2, n_volumes=1, n_reps=2, n_classes=2, relative=False, relative_action=None, custom_fit=False):
        height, width, depth, colour_channels = shape
        self.run_vnet(model, height, width, depth,
                      colour_channels, merge_connections,
                      epochs, n_volumes, n_reps, n_classes,
                      relative, relative_action)

    def run_vnet(self, model, height, width, depth, colour_channels, merge_connections,
                 epochs=2, n_volumes=5, n_reps=2, n_classes=1, relative=False, relative_action=None, custom_fit=False):
        volumes, one_hots = self.create_test_volume(n_volumes, n_reps, n_classes,
                                                    width, height, depth, colour_channels)
        print("=============================")
        print(one_hots.shape)
        print(one_hots[:, :, :, 3].shape)

        if model == "slice":
            one_hots = one_hots[:, :, :, 3]
        print("=============================")

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
                return VNet_Small(colour_channels, n_classes, merge_connections=merge_connections, use_batchnorm=False)
            elif model == "small_relative":
                from Segmentation.model.vnet_small_relative import VNet_Small_Relative
                return VNet_Small_Relative(colour_channels, n_classes, merge_connections=merge_connections,
                                           action=relative_action)
            elif model == "slice":
                from Segmentation.model.vnet_slice import VNet_Slice
                return VNet_Slice(colour_channels, n_classes, merge_connections=merge_connections)
            else:
                raise NotImplementedError(f"no model named: {model}")

        vnet = build_vnet(model, colour_channels, n_classes, merge_connections)

        target_shape = one_hots.shape
        if n_classes > 1:
            target_shape = (n_volumes, target_shape[1] * target_shape[2] * target_shape[3], n_classes)

        def vnet_feedforward(vnet, inputs, one_hots):
            output = vnet(inputs, training=False)
            assert output.shape == one_hots.shape
            output = vnet.predict(inputs)
            assert output.shape == one_hots.shape

        vnet_feedforward(vnet, inputs, one_hots)

        def vnet_fit(vnet, inputs, one_hots, epochs, custom_fit, n_classes):
            from tensorflow.keras.optimizers import Adam
            from Segmentation.utils.losses import dice_loss, tversky_loss, bce_dice_loss, focal_tversky
            from tensorflow.keras.losses import MSE

            if n_classes == 1:
                loss_func = dice_loss
            else:
                loss_func = tversky_loss

            if custom_fit:

                def loss(model, x, y, training, loss_func):
                    y_ = model(x, training=training)
                    return loss_func(y_true=y, y_pred=y_)

                print(inputs.shape)
                print(one_hots.shape)

                current_l = loss(vnet, inputs, one_hots, False, loss_func)
                print(f"Loss test: {current_l.shape}, {current_l}")

                def grad(model, inputs, targets, loss_func):
                    with tf.GradientTape() as tape:
                        loss_value = loss(model, inputs, targets, True, loss_func)
                    return loss_value, tape.gradient(loss_value, model.trainable_variables)

                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                loss_value, grads = grad(vnet, inputs, one_hots, loss_func)

                for var, grad in zip(vnet.trainable_variables, grads):
                    assert grad is not None, f"gradient must not be none for {var.name}"

                optimizer.apply_gradients(zip(grads, vnet.trainable_variables))

                current_l = loss(vnet, inputs, one_hots, False, loss_func)
                print(f"Loss test: {current_l.shape}")

            else:
                metrics = ['categorical_crossentropy']
                if n_classes == 1:
                    metrics.append(dice_coef_loss)
                vnet.compile(optimizer=Adam(0.001),
                             loss=loss_func,
                             metrics=metrics,
                             experimental_run_tf_function=True)

                history = vnet.fit(x=inputs, y=one_hots, epochs=epochs, verbose=1)
                loss_history = history.history['loss']

        # vnet_fit(vnet, inputs, one_hots, epochs, custom_fit, n_classes)

if __name__ == '__main__':
    import sys
    from os import getcwd
    sys.path.insert(0, getcwd())

    gpus = tf.config.experimental.list_physical_devices('GPU')
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

    # tf.test.main()

    tv = Test_VNet()
    tv.run_vnet(model="slice", n_volumes=3,
                height=96, width=96, depth=5, colour_channels=1,
                merge_connections=False, relative=False, n_classes=1,
                epochs=2, custom_fit=True)
